import json
import os
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from PIL import Image

from arguments import ModelParams
from scene.gaussian_model import BasicPointCloud
from scene.tonemapping import untonemap
from utils.depth_utils import (
    linear_least_squares_1d,
    project_pointcloud_to_depth_map,
    transform_depth_to_position_image,
    transform_distance_to_position_image,
    transform_normals_to_world,
    transform_points,
)
from utils.graphics_utils import focal2fov, fov2focal

from .camera_info import CameraInfo
from .image_utils import from_pil_image


class BlenderPriorDataset:
    def __init__(
        self,
        model_params: ModelParams,
        data_dir: str,
        point_cloud: BasicPointCloud,
        split: str = "train",
        dirname: str = None,
    ):
        self.model_params = model_params
        self.data_dir = data_dir
        self.point_cloud = point_cloud
        self.split = split
        self.dirname = split if dirname is None else dirname
        self.buffers_dir = os.path.join(self.data_dir, self.dirname)
        transform_path = os.path.join(data_dir, f"transforms_{split}.json")
        with open(transform_path) as json_file:
            self.contents = json.load(json_file)
        self.frames = sorted(self.contents["frames"], key=lambda x: x["file_path"])
        self.frames = self.frames[: self.model_params.max_images]
        assert len(self.frames) != 0, "Dataset is empty"

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> CameraInfo:
        frame = self.frames[idx]
        frame_name = frame["file_path"]
        image_name = Path(frame_name).stem
        image_path = os.path.join(self.data_dir, frame_name + ".png")

        image = self._get_buffer(frame_name, "image")
        albedo_image = self._get_buffer(frame_name, "albedo")
        # irradiance_image = self._get_buffer(frame_name, "irradiance")
        diffuse_image = self._get_buffer(frame_name, "diffuse")
        glossy_image = self._get_buffer(frame_name, "glossy")
        roughness_image = self._get_buffer(frame_name, "roughness")
        if "SKIP_THRESHOLD_ROUGHNESS" not in os.environ:
            roughness_image[roughness_image < 0.25] = 0.0
            upsized_roughness = torch.nn.functional.interpolate(
                roughness_image.moveaxis(-1, 0)[None],
                scale_factor=4,
                mode="bicubic",
                antialias=True,
            )
            upsized_roughness[upsized_roughness < 0.25] = 0.0
            roughness_image = torch.nn.functional.interpolate(
                upsized_roughness, scale_factor=1 / 4, mode="area"
            )[0].moveaxis(0, -1)
        metalness_image = self._get_buffer(frame_name, "metalness")
        if "SKIP_THRESHOLD_METALNESS" not in os.environ:
            upsized_metal = torch.nn.functional.interpolate(
                metalness_image.moveaxis(-1, 0)[None],
                scale_factor=4,
                mode="bicubic",
                antialias=True,
            )
            metalness_image = torch.nn.functional.interpolate(
                (upsized_metal > 0.4).float(), scale_factor=1 / 4, mode="area"
            )[0].moveaxis(0, -1)

        specular_image = torch.zeros_like(image) + 0.5
        depth_image = self._get_buffer(frame_name, "depth_moge")
        normal_image = self._get_buffer(frame_name, "normal_stable")
        brdf_image = torch.zeros_like(image)

        # Camera intrinsics
        height, width = image.shape[0], image.shape[1]
        if "camera_angle_y" in self.contents:
            fovy = self.contents["camera_angle_y"]
            fovx = self.contents["camera_angle_x"]
        else:
            fovx = self.contents["camera_angle_x"]
            fovy = focal2fov(fov2focal(fovx, width), height)

        # Camera extrinsics
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        # R is stored transposed due to 'glm' in CUDA code
        R = np.transpose(w2c[:3, :3])

        T = w2c[:3, 3]

        # Postprocess normal_image
        R_tensor = torch.tensor(R, dtype=torch.float32)
        normal_image = transform_normals_to_world(normal_image, R_tensor)

        # Postprocess position_image
        c2w_tensor = torch.tensor(c2w, dtype=torch.float32)
        w2c_tensor = torch.tensor(w2c, dtype=torch.float32)
        points_tensor = torch.tensor(self.point_cloud.points, dtype=torch.float32)
        points_tensor = transform_points(points_tensor, w2c_tensor)
        depth_image = depth_image[:, :, 0]
        depth_points_image = project_pointcloud_to_depth_map(
            points_tensor, fovx, fovy, depth_image.shape
        )
        a, b = linear_least_squares_1d(
            depth_image[depth_points_image != 0],
            depth_points_image[depth_points_image != 0],
        )
        depth_image = depth_image * a + b
        # camera_center = torch.tensor([0.007455553859472275, -1.6538097858428955, 0.9788005352020264])
        position_image = transform_depth_to_position_image(depth_image, fovx, fovy)
        position_image = transform_points(position_image, c2w_tensor)

        # def save_position_image_to_ply(position_image, path):
        #     import numpy as np
        #     from plyfile import PlyData, PlyElement

        #     H, W, _ = position_image.shape
        #     points = position_image.reshape(-1, 3).astype(np.float32)

        #     verts = np.array(
        #         [(p[0], p[1], p[2]) for p in points],
        #         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        #     )

        #     ply = PlyData([PlyElement.describe(verts, 'vertex')], text=True)
        #     ply.write(path)
        # save_position_image_to_ply(position_image.numpy(), "distance_converted2.ply")

        if "ABLATION" in os.environ:
            resolution = image.shape[0]
            scene_name = (
                os.path.basename(self.data_dir)
                .replace("_128", "")
                .replace("_256", "")
                .replace("_512", "")
                .replace("_768", "")
            )
            assert int(image_name.split("_")[-1]) == idx
            (
                rendered_image,
                rendered_diffuse_image,
                rendered_glossy_image,
                rendered_normal_image,
                rendered_position_image,
                rendered_roughness_image,
                rendered_specular_image,
                rendered_metalness_image,
                rendered_base_color_image,
                rendered_brdf_image,
            ) = torch.load(
                f"cache/{resolution}/{scene_name}/{self.split}/render/{idx:04d}.pt".replace(
                    "/render/", "/"
                )
            )

            ablated_passes = os.environ["ABLATION"].split(",")

            if "image" not in ablated_passes:
                image = rendered_image
            if "diffuse" not in ablated_passes:
                diffuse_image = rendered_diffuse_image
            if "glossy" not in ablated_passes:
                glossy_image = rendered_glossy_image
            if "normals" not in ablated_passes and "normal" not in ablated_passes:
                normal_image = rendered_normal_image
            if "position" not in ablated_passes:
                position_image = rendered_position_image
            if "roughness" not in ablated_passes:
                roughness_image = rendered_roughness_image
            if "specular" not in ablated_passes:
                specular_image = rendered_specular_image
            if "metalness" not in ablated_passes:
                metalness_image = rendered_metalness_image
            if "base_color" not in ablated_passes:
                base_color_image = rendered_base_color_image
            else:
                base_color_image = albedo_image
        else:
            base_color_image = albedo_image

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
            #
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
        buffer = from_pil_image(buffer_image)
        if buffer_name in ["image", "irradiance", "diffuse", "glossy"]:
            buffer = untonemap(buffer)
        elif buffer_name == "albedo":
            pass
        elif buffer_name in ["roughness", "metalness", "depth", "depth_moge"]:
            buffer = repeat(buffer, "h w 1 -> h w 3")
        elif buffer_name in ["normal", "normal_stable"]:
            buffer = buffer * 2.0 - 1.0
        else:
            raise ValueError(f"Buffer name not recognized: {buffer_name}")
        buffer = torch.tensor(buffer)

        if buffer_name in ["image", "irradiance", "diffuse", "glossy"]:
            buffer /= 3.5

        return buffer
