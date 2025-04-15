import os

import numpy as np
import torch
from einops import repeat
from PIL import Image

from arguments import ModelParams
from scene.gaussian_model import BasicPointCloud
from utils.depth_utils import (
    linear_least_squares_1d,
    project_pointcloud_to_depth_map,
    transform_depth_to_position_image,
    transform_normals_to_world,
    transform_points,
)
from utils.graphics_utils import focal2fov, fov2focal

from .camera_info import CameraInfo
from .colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)


class ColmapDataset:
    def __init__(self, model_params: ModelParams, data_dir: str, split: str = "train"):
        self.model_params = model_params
        self.data_dir = data_dir
        self.split = split
        self.llffhold = 8
        self.do_eval = model_params.eval
        self._point_cloud = self.get_point_cloud()

        try:
            cameras_extrinsic_file = os.path.join(data_dir, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(data_dir, "sparse/0", "cameras.bin")
            self.cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            self.cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(data_dir, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(data_dir, "sparse/0", "cameras.txt")
            self.cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            self.cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        keys = sorted(list(self.cam_extrinsics.keys()))
        if self.do_eval:
            if split == "train":
                self.keys = [
                    key for idx, key in enumerate(keys) if idx % self.llffhold != 0
                ]
            else:
                self.keys = [
                    key for idx, key in enumerate(keys) if idx % self.llffhold == 0
                ]
        else:
            if split == "train":
                self.keys = keys
            else:
                self.keys = []

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> CameraInfo:
        key = self.keys[idx]
        extr = self.cam_extrinsics[key]
        intr = self.cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        frame_name = os.path.splitext(extr.name)[0]
        uid = intr.id

        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(extr.qvec)
        w2c[:3, 3] = extr.tvec
        c2w = np.linalg.inv(w2c)

        # R is stored transposed due to 'glm' in CUDA code
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

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

        image = self._get_buffer(frame_name, "image")
        albedo_image = self._get_buffer(frame_name, "albedo")
        # irradiance_image = self._get_buffer(frame_name, "irradiance")
        diffuse_image = self._get_buffer(frame_name, "diffuse")
        glossy_image = self._get_buffer(frame_name, "glossy")
        roughness_image = self._get_buffer(frame_name, "roughness")
        metalness_image = self._get_buffer(frame_name, "metalness")
        depth_image = self._get_buffer(frame_name, "depth")
        normal_image = self._get_buffer(frame_name, "normal")
        specular_image = torch.zeros_like(image)
        brdf_image = torch.zeros_like(image)

        # Manually adjust exposure
        image /= 3.5
        diffuse_image /= 3.5
        glossy_image /= 3.5

        # Postprocess normal_image
        R_tensor = torch.tensor(R, dtype=torch.float32)
        normal_image = transform_normals_to_world(normal_image, R_tensor)

        # Postprocess position_image
        c2w_tensor = torch.tensor(c2w, dtype=torch.float32)
        w2c_tensor = torch.tensor(w2c, dtype=torch.float32)
        points_tensor = torch.tensor(self._point_cloud.points, dtype=torch.float32)
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
        position_image = transform_depth_to_position_image(depth_image, fovx, fovy)
        position_image = transform_points(position_image, c2w_tensor)

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image=image,
            image_path=os.path.join(self.data_dir, frame_name + ".png"),
            image_name=frame_name,
            width=width,
            height=height,
            diffuse_image=diffuse_image,
            glossy_image=glossy_image,
            position_image=position_image,
            normal_image=normal_image,
            roughness_image=roughness_image,
            metalness_image=metalness_image,
            base_color_image=albedo_image,
            brdf_image=brdf_image,
            specular_image=specular_image,
        )
        return cam_info

    def _get_buffer(self, frame_name: str, buffer_name: str):
        buffer_file_name = frame_name.split("/")[-1]
        buffer_path = os.path.join(
            self.data_dir, "priors", buffer_name, buffer_file_name + ".png"
        )
        buffer = _from_pil_image(Image.open(buffer_path))

        if buffer_name == "image":
            buffer = buffer**2.2
        elif buffer_name == "albedo":
            pass
        elif buffer_name in ["irradiance", "diffuse", "glossy"]:
            buffer = buffer / (1.0 - buffer + 1e-6)
        elif buffer_name in ["roughness", "metalness", "depth"]:
            buffer = repeat(buffer, "h w 1 -> h w 3")
        elif buffer_name == "normal":
            buffer = buffer * 2.0 - 1.0
        else:
            raise ValueError(f"Buffer name not recognized: {buffer_name}")
        buffer = torch.tensor(buffer)
        return buffer

    def get_point_cloud(self) -> BasicPointCloud:
        bin_path = os.path.join(self.data_dir, "sparse/0/points3D.bin")
        txt_path = os.path.join(self.data_dir, "sparse/0/points3D.txt")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        pcd = BasicPointCloud(
            points=xyz,
            colors=rgb / 255.0,
            normals=np.zeros_like(xyz),
        )
        return pcd


def _from_pil_image(img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr / (2**8 - 1)
    elif arr.ndim == 2:
        arr = arr / (2**16 - 1)
        arr = arr[:, :, None]
    return arr
