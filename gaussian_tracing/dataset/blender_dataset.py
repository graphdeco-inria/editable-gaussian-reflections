import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange

from gaussian_tracing.dataset.colmap_parser import ColmapParser
from gaussian_tracing.utils.depth_utils import transform_depth_to_position_image
from gaussian_tracing.utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal

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

        self.colmap_parser = ColmapParser(data_dir)
        self.point_cloud = BasicPointCloud(
            points=self.colmap_parser.points,
            colors=self.colmap_parser.points_rgb,
            normals=np.zeros_like(self.colmap_parser.points),
        )

        downsampled_cache_dir = data_dir.replace(
            "/renders/", f"/cache/{self.resolution}/"
        )
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
                specular_image,
                metalness_image,
                base_color_image,
                brdf_image,
            ) = torch.unbind(image_tensor, dim=0)
            albedo_image = None
            depth_image = None
        else:
            image = self._get_buffer(frame_name, "render")
            albedo_image = self._get_buffer(frame_name, "albedo")
            diffuse_image = self._get_buffer(frame_name, "diffuse")
            glossy_image = self._get_buffer(frame_name, "glossy")
            roughness_image = self._get_buffer(frame_name, "roughness")
            metalness_image = self._get_buffer(frame_name, "metalness")
            normal_image = self._get_buffer(frame_name, "normal")
            depth_image = self._get_buffer(frame_name, "depth")
            specular_image = self._get_buffer(frame_name, "specular")
            brdf_image = self._get_buffer(frame_name, "glossy_brdf")
            base_color_image = self._get_buffer(frame_name, "base_color")
            # specular_image = torch.ones_like(image) * 0.5
            # brdf_image = torch.zeros_like(image)
            # base_color_image = albedo_image * (1.0 - metalness_image) + metalness_image

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

        # Convert to depth to distance image
        if depth_image is not None:
            position_image = transform_depth_to_position_image(
                depth_image[:, :, 0], fovx, fovy
            )
            distance_image = torch.norm(position_image, dim=-1)
        else:
            distance_image = (position_image - torch.from_numpy(c2w[:3, 3])).norm(dim=-1)

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
            albedo_image=albedo_image,
            diffuse_image=diffuse_image,
            glossy_image=glossy_image,
            depth_image=distance_image,
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
        if self.resolution is not None:
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
