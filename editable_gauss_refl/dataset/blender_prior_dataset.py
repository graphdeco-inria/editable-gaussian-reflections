import json
import os
from pathlib import Path
import sys 

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from editable_gauss_refl.dataset.colmap_parser import ColmapParser
from editable_gauss_refl.utils.depth_utils import (
    compute_primary_ray_directions,
    project_pointcloud_to_depth_map,
    ransac_linear_fit,
    transform_depth_to_position_image,
    transform_normals_to_world,
    transform_points,
)
from editable_gauss_refl.utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from editable_gauss_refl.utils.tonemapping import untonemap

from .camera_info import CameraInfo
from .image_utils import from_pil_image


class BlenderPriorDataset:
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

        self.buffer_names = [
            "render",
            "diffuse",
            "specular",
            "roughness",
            "metalness",
            "depth",
            "normal"
        ]
        self.colmap_parser = ColmapParser(data_dir)
        self.point_cloud = BasicPointCloud(
            points=self.colmap_parser.points,
            colors=self.colmap_parser.points_rgb,
            normals=np.zeros_like(self.colmap_parser.points),
        )
        self.buffers_dir = os.path.join(self.data_dir, split)
        transform_path = os.path.join(data_dir, f"transforms_{split}.json")
        with open(transform_path) as json_file:
            self.contents = json.load(json_file)
        self.frames = sorted(self.contents["frames"], key=lambda x: x["file_path"])
        if self.max_images is not None:
            self.frames = self.frames[: self.max_images]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> CameraInfo:
        frame = self.frames[idx]
        frame_name = frame["file_path"]
        image_name = Path(frame_name).stem + ".png"
        image_path = os.path.join(self.data_dir, image_name)

        buffers = {buffer_name: self._get_buffer(frame_name, buffer_name) for buffer_name in self.buffer_names}

        # Resize all buffers
        if self.resolution is not None:
            for k, v in buffers.items():
                buffers[k] = _resize_image_tensor(v, self.resolution)

        # Camera intrinsics
        height, width = buffers["render"].shape[:2]
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

        R_tensor = torch.tensor(R, dtype=torch.float32)
        w2c_tensor = torch.tensor(w2c, dtype=torch.float32)

        # Postprocess normal_image
        buffers["normal"] = transform_normals_to_world(buffers["normal"], R_tensor)

        # Postprocess depth_image
        # if self.do_depth_fit:
        points_tensor = torch.tensor(
            self.colmap_parser.points[self.colmap_parser.point_indices[image_name]],
            dtype=torch.float32,
        )
        points_tensor = transform_points(points_tensor, w2c_tensor)
        depth_points_image = project_pointcloud_to_depth_map(points_tensor, fovx, fovy, buffers["depth"].shape[:2])
        valid_mask = depth_points_image != 0
        x = buffers["depth"][:, :, 0][valid_mask].float()
        y = depth_points_image[valid_mask]
        # a, b = linear_least_squares_1d(x, y)
        (a, b), _ = ransac_linear_fit(x, y)
        buffers["depth"] = (buffers["depth"] * a + b)

        # Convert to depth to distance image
        position_image = transform_depth_to_position_image(buffers["depth"].squeeze(-1), fovx, fovy) 
        
        # Save position_image as a PLY file
        point_cloud = position_image.reshape(-1, 3).cpu().numpy()
        valid_points = ~np.isnan(point_cloud).any(axis=1)
        point_cloud = point_cloud[valid_points]

        buffers["distance"] = torch.norm(position_image, dim=-1, keepdim=True)

        # Convert metalness to f0 base reflectance image
        f0_image = (0.04 * (1.0 - buffers["metalness"]) + buffers["metalness"]).repeat(1, 1, 3)

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image=buffers["render"],
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            #
            diffuse_image=buffers["diffuse"],
            specular_image=buffers["specular"],
            depth_image=buffers["distance"],
            normal_image=buffers["normal"],
            roughness_image=buffers["roughness"],
            f0_image=f0_image
        )
        return cam_info

    def _get_buffer(self, frame_name: str, buffer_name: str) -> Tensor:
        file_name = frame_name.split("/")[-1]
        frame_id = file_name.split("_")[-1]
        buffer_path = os.path.join(self.buffers_dir, buffer_name, f"{buffer_name}_{frame_id}" + ".png")

        buffer_image = Image.open(buffer_path)
        buffer = from_pil_image(buffer_image)
        if buffer_name in ["render", "irradiance", "diffuse", "specular"]:
            buffer = untonemap(buffer)
        elif buffer_name in ["depth", "roughness", "metalness"]:
            pass
        elif buffer_name in ["normal"]:
            buffer = buffer * 2.0 - 1.0
        else:
            raise ValueError(f"Buffer name not recognized: {buffer_name}")
        buffer = torch.tensor(buffer)
        return buffer


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
