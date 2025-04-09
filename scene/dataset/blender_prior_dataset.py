import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
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
    read_points3D_binary,
    read_points3D_text,
)


class BlenderPriorDataset:
    def __init__(self, model_params: ModelParams, data_dir: str, split: str = "train"):
        self.model_params = model_params
        self.data_dir = data_dir
        self.split = split
        self.do_fallback = True
        self.fallback_dir = f"./data/renders/{self.data_dir.split('/')[-1]}"
        self._point_cloud = self.get_point_cloud()

        transform_path = os.path.join(data_dir, f"transforms_{split}.json")
        with open(transform_path) as json_file:
            contents = json.load(json_file)
        self.frames = sorted(contents["frames"], key=lambda x: x["file_path"])
        assert len(self.frames) != 0, "Dataset is empty"

        # Read first image to get height and width
        first_frame_name = self.frames[0]["file_path"]
        first_image = self._get_buffer(first_frame_name, "image")
        self.height, self.width = first_image.shape[:2]
        self.fovx = contents["camera_angle_x"]
        self.fovy = focal2fov(fov2focal(self.fovx, self.width), self.height)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> CameraInfo:
        frame = self.frames[idx]
        frame_name = frame["file_path"]
        image_name = Path(frame_name).stem

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        # R is stored transposed due to 'glm' in CUDA code
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        image = self._get_buffer(frame_name, "image")
        albedo_image = self._get_buffer(frame_name, "albedo")
        irradiance_image = self._get_buffer(frame_name, "irradiance")
        normal_image = self._get_buffer(frame_name, "normal")
        depth_image = self._get_buffer(frame_name, "depth")
        diffuse_image = (albedo_image * irradiance_image).clip(0.0, 1.0)
        glossy_image = (image - diffuse_image).clip(0.0, 1.0)
        if self.do_fallback:
            roughness_image = self._get_buffer_fallback(frame_name, "roughness")
            specular_image = self._get_buffer_fallback(frame_name, "specular")
            metalness_image = self._get_buffer_fallback(frame_name, "metalness")
            brdf_image = self._get_buffer_fallback(frame_name, "glossy_brdf")
        else:
            roughness_image = torch.zeros_like(image)
            metalness_image = torch.zeros_like(image)
            specular_image = torch.zeros_like(image)
            brdf_image = torch.zeros_like(image)

        # Postprocess buffers
        R_tensor = torch.tensor(R, dtype=torch.float32)
        c2w_tensor = torch.tensor(c2w, dtype=torch.float32)
        w2c_tensor = torch.tensor(w2c, dtype=torch.float32)
        normal_image = transform_normals_to_world(normal_image, R_tensor)

        points_tensor = torch.tensor(self._point_cloud.points, dtype=torch.float32)
        points_tensor = transform_points(points_tensor, w2c_tensor)
        points_image = project_pointcloud_to_depth_map(
            points_tensor, self.fovx, self.fovy, depth_image.shape
        )
        valid_mask = points_image != 0
        xs = depth_image[valid_mask]
        ys = points_image[valid_mask]
        a, b = linear_least_squares_1d(xs, ys)
        depth_image = depth_image * a + b
        position_image = transform_depth_to_position_image(
            depth_image, self.fovx, self.fovy
        )
        position_image = transform_points(position_image, c2w_tensor)

        # Manually adjust exposure
        image *= 0.5
        diffuse_image *= 0.5
        glossy_image *= 0.5

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=self.fovy,
            FovX=self.fovx,
            image=image,
            image_path=os.path.join(self.data_dir, frame_name + ".png"),
            image_name=image_name,
            width=self.width,
            height=self.height,
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
            self.data_dir, self.split, buffer_name, buffer_file_name + ".png"
        )
        buffer = np.array(Image.open(buffer_path), dtype=np.float32) / 255.0
        buffer = torch.tensor(buffer)

        if buffer_name in ["image", "albedo"]:
            buffer = buffer**2.2
        elif buffer_name in ["roughness", "metalness"]:
            pass
        elif buffer_name == "depth":
            buffer /= 255.0
        elif buffer_name == "normal":
            buffer = 2.0 * buffer - 1.0
        elif buffer_name == "irradiance":
            buffer = 1.0 / (1.0 - buffer + 1e-6) - 1.0
        else:
            raise ValueError(f"Buffer name not recognized: {buffer_name}")
        return buffer

    def _get_buffer_fallback(self, frame_name: str, buffer_name: str, R=None):
        buffer_filename = frame_name.replace("render", buffer_name)
        buffer_path = os.path.join(self.fallback_dir, buffer_filename + ".exr")
        assert os.path.exists(buffer_path), f"{buffer_name} not found at {buffer_path}"
        image = cv2.imread(buffer_path, cv2.IMREAD_UNCHANGED)
        image = torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image

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
