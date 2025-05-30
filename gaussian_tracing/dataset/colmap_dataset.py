import os

import numpy as np
import torch
from einops import repeat
from PIL import Image

from arguments import ModelParams
from gaussian_tracing.dataset.colmap_parser import ColmapParser
from gaussian_tracing.utils.depth_utils import (
    project_pointcloud_to_depth_map,
    ransac_linear_fit,
    transform_depth_to_position_image,
    transform_normals_to_world,
    transform_points,
)
from gaussian_tracing.utils.graphics_utils import BasicPointCloud, focal2fov
from gaussian_tracing.utils.tonemapping import untonemap

from .camera_info import CameraInfo
from .colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from .image_utils import from_pil_image


class ColmapDataset:
    def __init__(
        self,
        model_params: ModelParams,
        data_dir: str,
        split: str = "train",
    ):
        self.model_params = model_params
        self.data_dir = data_dir
        self.split = split

        self.colmap_parser = ColmapParser(data_dir)
        self.point_cloud = BasicPointCloud(
            points=self.colmap_parser.points,
            colors=self.colmap_parser.points_rgb,
            normals=np.zeros_like(self.colmap_parser.points),
        )

        self.buffers_dir = os.path.join(self.data_dir, "priors")
        self.llffhold = 8
        try:
            cameras_extrinsic_file = os.path.join(data_dir, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(data_dir, "sparse/0", "cameras.bin")
            self.cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            self.cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except Exception:
            cameras_extrinsic_file = os.path.join(data_dir, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(data_dir, "sparse/0", "cameras.txt")
            self.cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            self.cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        keys = list(sorted(list(self.cam_extrinsics.keys())))[
            : self.model_params.max_images
        ]
        if model_params.eval:
            if split == "train":
                self.keys = [
                    key for i, key in enumerate(keys) if i % self.llffhold != 0
                ]
            else:
                self.keys = [
                    key for i, key in enumerate(keys) if i % self.llffhold == 0
                ]
        else:
            if split == "train":
                self.keys = keys
            else:
                self.keys = []

        if "MANUAL_FILTER" in os.environ:
            self.best_frames = (
                open(os.path.join(data_dir, "best_frames.txt"), "r")
                .read()
                .strip()
                .split(" ")
            )
            self.keys = [k for k in self.keys if k in self.best_frames]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> CameraInfo:
        key = self.keys[idx]
        extr = self.cam_extrinsics[key]
        intr = self.cam_intrinsics[extr.camera_id]
        image_name = extr.name
        frame_name = os.path.splitext(image_name)[0]
        image_path = os.path.join(self.data_dir, "images", frame_name + ".jpg")

        image = self._get_buffer(frame_name, "image")
        albedo_image = self._get_buffer(frame_name, "albedo")
        diffuse_image = self._get_buffer(frame_name, "diffuse")
        glossy_image = self._get_buffer(frame_name, "glossy")
        roughness_image = self._get_buffer(frame_name, "roughness")
        metalness_image = self._get_buffer(frame_name, "metalness")
        depth_image = self._get_buffer(frame_name, "depth")
        normal_image = self._get_buffer(frame_name, "normal")
        specular_image = torch.ones_like(image) * 0.5
        brdf_image = torch.zeros_like(image)
        base_color_image = albedo_image * (1.0 - metalness_image) + metalness_image

        roughness_image = self._get_buffer(frame_name, "roughness")
        metalness_image = self._get_buffer(frame_name, "metalness")
        if "REAL_SCENE" in os.environ:
            # original_metalness_image = metalness_image
            # original_roughness_image = roughness_image
            if "SKIP_MIRROR_METALS" not in os.environ:
                roughness_image = roughness_image * (1.0 - metalness_image)
            if "SKIP_WHITE_METALS" not in os.environ:
                albedo_image = albedo_image * (1.0 - metalness_image) + metalness_image
                diffuse_image = diffuse_image * (1.0 - metalness_image)
            # if "SKIP_THRESHOLD_ROUGHNESS" not in os.environ:
            #     upsized_roughness = torch.nn.functional.interpolate(roughness_image.moveaxis(-1, 0)[None], scale_factor=4, mode='bicubic', antialias=True)
            #     upsized_roughness[upsized_roughness < 0.25] = 0.0
            #     roughness_image = torch.nn.functional.interpolate(upsized_roughness, scale_factor=1/4, mode="area")[0].moveaxis(0, -1)
            # if "SKIP_SPECULAR_FROM_METALNESS" not in os.environ:
            #     specular_image = original_roughness_image / 2 + 0.5

        # Camera intrinsics
        height = intr.height
        width = intr.width
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            fovy = focal2fov(focal_length_x, height)
            fovx = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            fovy = focal2fov(focal_length_y, height)
            fovx = focal2fov(focal_length_x, width)
        else:
            assert False, (
                "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )

        # Camera extrinsics
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(extr.qvec)
        w2c[:3, 3] = extr.tvec
        c2w = np.linalg.inv(w2c)
        # R is stored transposed due to 'glm' in CUDA code
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        # Postprocess normal_image
        R_tensor = torch.tensor(R, dtype=torch.float32)
        normal_image = transform_normals_to_world(normal_image, R_tensor)

        # Postprocess depth_image to position_image
        depth_image = depth_image[:, :, 0]
        c2w_tensor = torch.tensor(c2w, dtype=torch.float32)
        w2c_tensor = torch.tensor(w2c, dtype=torch.float32)
        points_tensor = torch.tensor(
            self.colmap_parser.points[self.colmap_parser.point_indices[image_name]],
            dtype=torch.float32,
        )
        points_tensor = transform_points(points_tensor, w2c_tensor)
        depth_points_image = project_pointcloud_to_depth_map(
            points_tensor, fovx, fovy, depth_image.shape[:2]
        )
        valid_mask = depth_points_image != 0
        x = depth_image[valid_mask].float()
        y = depth_points_image[valid_mask]
        # a, b = linear_least_squares_1d(x, y)
        (a, b), _ = ransac_linear_fit(x, y)
        depth_image = depth_image * a + b
        position_image = transform_depth_to_position_image(depth_image, fovx, fovy)
        position_image = transform_points(position_image, c2w_tensor)

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            diffuse_image=diffuse_image,
            glossy_image=glossy_image,
            position_image=position_image,
            normal_image=normal_image,
            roughness_image=roughness_image,
            metalness_image=metalness_image,
            base_color_image=base_color_image,
            brdf_image=brdf_image,
            specular_image=specular_image,
        )
        return cam_info

    def _get_buffer(self, frame_name: str, buffer_name: str):
        file_name = frame_name.split("/")[-1]
        buffer_path = os.path.join(self.buffers_dir, buffer_name, file_name + ".png")

        buffer_image = Image.open(buffer_path)
        buffer_height = self.model_params.resolution
        buffer_width = int(
            buffer_height * (buffer_image.size[0] / buffer_image.size[1])
        )
        buffer_image = buffer_image.resize((buffer_width, buffer_height))
        buffer = from_pil_image(buffer_image)

        if buffer_name in ["image", "irradiance", "diffuse", "glossy"]:
            buffer = untonemap(buffer)
            buffer /= 3.5  # Align exposure
        elif buffer_name == "albedo":
            pass
        elif buffer_name in ["roughness", "metalness", "depth"]:
            buffer = repeat(buffer, "h w 1 -> h w 3")
        elif buffer_name in ["normal"]:
            buffer = buffer * 2.0 - 1.0
        else:
            raise ValueError(f"Buffer name not recognized: {buffer_name}")
        buffer = torch.tensor(buffer)
        return buffer
