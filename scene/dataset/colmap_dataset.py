import os

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams
from scene.gaussian_model import BasicPointCloud
from utils.depth_utils import transform_normals_to_world
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
        self.images_folder = os.path.join(self.data_dir, "images")
        self.priors_folder = os.path.join(self.data_dir, "priors")
        assert model_params.linear_space

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

        frame_name = extr.name
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, (
                "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )

        image = self._get_buffer(frame_name, "image")
        albedo_image = self._get_buffer(frame_name, "albedo")
        irradiance_image = self._get_buffer(frame_name, "irradiance")
        normal_image = self._get_buffer(frame_name, "normal")
        diffuse_image = (albedo_image * irradiance_image).clip(0.0, 1.0)
        glossy_image = (image - diffuse_image).clip(0.0, 1.0)
        position_image = torch.zeros_like(image)
        roughness_image = torch.zeros_like(image)
        specular_image = torch.zeros_like(image)
        metalness_image = torch.zeros_like(image)
        brdf_image = torch.zeros_like(image)

        # Postprocess buffers
        normal_image = transform_normals_to_world(
            normal_image, torch.tensor(R, dtype=torch.float32)
        )
        diffuse_image = diffuse_image * self.model_params.exposure
        glossy_image = glossy_image * self.model_params.exposure

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=os.path.join(self.images_folder, frame_name),
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

    def _get_buffer(self, frame_name: str, buffer_name: str, R=None):
        buffer_file_name = os.path.splitext(frame_name.split("/")[-1])[0]
        buffer_path = os.path.join(
            self.priors_folder, buffer_name, buffer_file_name + ".png"
        )
        buffer = np.array(Image.open(buffer_path), dtype=np.float32) / 255.0

        if buffer_name in ["image", "albedo"]:
            buffer = buffer**2.2
            return torch.tensor(buffer)
        elif buffer_name in ["roughness", "metalness"]:
            return torch.tensor(buffer)
        elif buffer_name == "normal":
            normal = 2.0 * buffer - 1.0
            normal = transform_normals_to_world(normal, R)
            return torch.tensor(normal)
        elif buffer_name == "irradiance":
            irradiance = 1.0 / (1.0 - buffer + 1e-6) - 1.0
            return torch.tensor(irradiance)
        else:
            raise ValueError(f"Buffer name not recognized: {buffer_name}")

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
