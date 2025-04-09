import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from arguments import ModelParams
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal

from .camera_info import CameraInfo
from .colmap_loader import (
    read_points3D_binary,
    read_points3D_text,
)


class BlenderDataset:
    def __init__(self, model_params: ModelParams, data_dir: str, split: str = "train"):
        self.model_params = model_params
        self.data_dir = data_dir
        self.split = split
        self.cache_dir = data_dir.replace("/renders/", "/cache/")
        assert model_params.linear_space

        transform_path = os.path.join(data_dir, f"transforms_{split}.json")
        with open(transform_path) as json_file:
            self.contents = json.load(json_file)

        self.frames = sorted(
            self.contents["frames"], key=lambda frame: frame["file_path"]
        )
        self.frames = self.frames[: self.model_params.max_images]

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

        if "LOAD_FROM_IMAGE_FILES" not in os.environ:
            cache_name = frame_name.replace("render/render_", "")
            cache_path = os.path.join(self.cache_dir, cache_name + ".pt")
            image_tensor = torch.load(cache_path)
            (
                image,
                diffuse_image,
                glossy_image,
                normal_image,
                position_image,
                roughness_image,
                metalness_image,
                base_color_image,
                brdf_image,
                specular_image,
            ) = torch.unbind(image_tensor, dim=0)
            height, width = image.shape[0], image.shape[1]
        else:
            image = self._get_buffer(frame_name, "render")
            diffuse_image = self._get_buffer(frame_name, "diffuse")
            glossy_image = self._get_buffer(frame_name, "glossy")
            normal_image = self._get_buffer(frame_name, "normal")
            position_image = self._get_buffer(frame_name, "position")
            roughness_image = self._get_buffer(frame_name, "roughness")
            specular_image = self._get_buffer(frame_name, "specular")
            metalness_image = self._get_buffer(frame_name, "metalness")
            base_color_image = self._get_buffer(frame_name, "base_color")
            brdf_image = self._get_buffer(frame_name, "glossy_brdf")
            height, width = image.shape[0], image.shape[1]

        diffuse_image = diffuse_image * self.model_params.exposure
        glossy_image = glossy_image * self.model_params.exposure

        fovx = self.contents["camera_angle_x"]
        fovy = focal2fov(fov2focal(fovx, width), height)

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image=image,
            image_path=os.path.join(self.data_dir, frame_name + ".png"),
            image_name=Path(frame_name).stem,
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
        buffer_filename = frame_name.replace("render", buffer_name)
        buffer_path = os.path.join(self.data_dir, buffer_filename + ".exr")
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
