import json
import os
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from einops import rearrange
from torchvision.io import read_image

from editable_gauss_refl.utils.graphics_utils import focal2fov, fov2focal

from .camera_info import CameraInfo


class BlenderDataset:
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        resolution: int | None = None,
        max_images: int | None = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.resolution = resolution
        self.max_images = max_images

        downsampled_cache_dir = data_dir.replace("/renders/", f"/cache/{self.resolution}/")
        if os.path.exists(downsampled_cache_dir):
            self.cache_dir = downsampled_cache_dir
        else:
            self.cache_dir = data_dir.replace("/renders/", "/cache/fullres/")
        transform_path = os.path.join(data_dir, f"transforms_{split}.json")
        with open(transform_path) as json_file:
            self.contents = json.load(json_file)
        self.frames = sorted(self.contents["frames"], key=lambda x: x["file_path"])
        if self.max_images is not None:
            self.frames = self.frames[: self.max_images]
        assert len(self.frames) != 0, "Dataset is empty"

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> CameraInfo:
        frame = self.frames[idx]
        frame_name = frame["file_path"]
        image_name = Path(frame_name).stem + ".png"
        image_path = os.path.join(self.data_dir, image_name)

        if "safetensors" in self.data_dir:
            import safetensors.torch

            st_path = os.path.join(self.data_dir, frame_name.replace("render/render_", "buffers_") + ".safetensors")
            st = safetensors.torch.load_file(st_path)
            image = st["render"].moveaxis(0, -1)
            diffuse_image = st["diffuse"].moveaxis(0, -1)
            glossy_image = st["glossy"].moveaxis(0, -1)
            normal_image = st["normal"].moveaxis(0, -1)
            depth_image = st["depth"].moveaxis(0, -1)
            roughness_image = st["roughness"].moveaxis(0, -1)
            f0_image = st["f0"].moveaxis(0, -1)
        else:
            image = self._get_buffer(frame_name, "render")
            diffuse_image = self._get_buffer(frame_name, "diffuse")
            glossy_image = self._get_buffer(frame_name, "glossy")
            roughness_image = self._get_buffer(frame_name, "roughness")
            normal_image = self._get_buffer(frame_name, "normal")
            depth_image = self._get_buffer(frame_name, "depth")
            f0_image = self._get_buffer(frame_name, "f0")

        # Camera intrinsics
        height, width = image.shape[0], image.shape[1]
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
            depth_image=depth_image,
            normal_image=normal_image,
            roughness_image=roughness_image,
            f0_image=f0_image,
        )
        return cam_info

    def _get_buffer(self, frame_name: str, buffer_name: str):
        buffer_filename = frame_name.replace("render", buffer_name)
        buffer_path = os.path.join(self.data_dir, buffer_filename + ".exr")
        if os.path.exists(buffer_path):
            image = cv2.imread(buffer_path, cv2.IMREAD_UNCHANGED)
            image = torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif os.path.exists(buffer_path.replace(".exr", ".tiff")):
            buffer_path = buffer_path.replace(".exr", ".tiff")
            image = torch.tensor(tifffile.imread(buffer_path))
        elif os.path.exists(buffer_path.replace(".exr", ".png")):
            buffer_path = buffer_path.replace(".exr", ".png")
            image = read_image(buffer_path)
            image = rearrange(image, "c h w -> h w c")
        assert image.shape[0] != 1
        if image.ndim == 2:
            image = image.unsqueeze(-1)
        if self.resolution is not None and image.shape[0] != self.resolution:
            image = _resize_image_tensor(image, self.resolution)
        return image


def _resize_image_tensor(image, resolution):
    height = image.shape[0]
    width = image.shape[1]
    aspect_ratio = width / height
    image = rearrange(image, "h w c -> 1 c h w")
    image = torch.nn.functional.interpolate(
        image,
        (resolution, int(resolution * aspect_ratio)),
        mode="area",
    )
    image = rearrange(image, "1 c h w -> h w c")
    return image
