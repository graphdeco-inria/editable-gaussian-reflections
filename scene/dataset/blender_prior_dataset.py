import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from arguments import ModelParams
from utils.graphics_utils import focal2fov, fov2focal, transform_normals_to_world

from .camera_info import CameraInfo


class BlenderPriorDataset:
    def __init__(self, model_params: ModelParams, data_dir: str, split: str = "train"):
        self.model_params = model_params
        self.data_dir = data_dir
        self.split = split
        self.do_fallback = True
        self.fallback_dir = f"./data/renders/{self.data_dir.split('/')[-1]}"

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
        diffuse_image = (albedo_image * irradiance_image).clip(0.0, 1.0)
        glossy_image = (image - diffuse_image).clip(0.0, 1.0)
        normal_image = self._get_buffer(frame_name, "normal", R=R)

        if self.do_fallback:
            position_image = self._get_buffer_fallback(frame_name, "position")
            roughness_image = self._get_buffer_fallback(frame_name, "roughness")
            specular_image = self._get_buffer_fallback(frame_name, "specular")
            metalness_image = self._get_buffer_fallback(frame_name, "metalness")
            brdf_image = self._get_buffer_fallback(frame_name, "glossy_brdf")
        else:
            position_image = torch.zeros_like(image)
            roughness_image = torch.zeros_like(image)
            specular_image = torch.zeros_like(image)
            metalness_image = torch.zeros_like(image)
            brdf_image = torch.zeros_like(image)

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=self.fovy,
            FovX=self.fovx,
            image=image,
            image_path=os.path.join(self.data_dir, frame_name + ".png"),
            image_name=Path(frame_name).stem,
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

    def _get_buffer(self, frame_name: str, buffer_name: str, R=None):
        buffer_file_name = frame_name.split("/")[-1]
        buffer_path = os.path.join(
            self.data_dir, self.split, buffer_name, buffer_file_name + ".png"
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

    def _get_buffer_fallback(self, frame_name: str, buffer_name: str, R=None):
        buffer_filename = frame_name.replace("render", buffer_name)
        buffer_path = os.path.join(self.fallback_dir, buffer_filename + ".exr")
        assert os.path.exists(buffer_path), f"{buffer_name} not found at {buffer_path}"
        image = cv2.imread(buffer_path, cv2.IMREAD_UNCHANGED)
        image = torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image
