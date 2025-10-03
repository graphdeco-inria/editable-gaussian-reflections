import json
import os
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from gaussian_tracing.dataset.colmap_parser import ColmapParser
from gaussian_tracing.utils.depth_utils import (
    project_pointcloud_to_depth_map,
    ransac_linear_fit,
    transform_depth_to_position_image,
    transform_normals_to_world,
    transform_points,
)
from gaussian_tracing.utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from gaussian_tracing.utils.tonemapping import untonemap

from .camera_info import CameraInfo
from .image_utils import from_pil_image


class BlenderPriorDataset:
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        resolution: int | None = None,
        max_images: int | None = None,
        do_eval: bool = True,
        do_depth_fit: bool = False,
    ):
        self.data_dir = data_dir
        self.split = split
        self.resolution = resolution
        self.max_images = max_images
        self.do_eval = do_eval
        self.do_depth_fit = do_depth_fit

        self.buffer_names = [
            "render",
            "albedo",
            "base_color",
            "diffuse",
            "glossy",
            "roughness",
            "metalness",
            "depth",
            "normal",
            "specular",
            "brdf",
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

        buffers = {
            buffer_name: self._get_buffer(frame_name, buffer_name)
            for buffer_name in self.buffer_names
        }
        if buffers["brdf"] is None:
            buffers["brdf"] = torch.zeros_like(buffers["render"])
        if buffers["specular"] is None:
            buffers["specular"] = torch.ones_like(buffers["render"]) * 0.5
        if buffers["base_color"] is None:
            buffers["base_color"] = (
                buffers["albedo"] * (1.0 - buffers["metalness"]) + buffers["metalness"]
            )

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
        if self.do_depth_fit:
            points_tensor = torch.tensor(
                self.colmap_parser.points[self.colmap_parser.point_indices[image_name]],
                dtype=torch.float32,
            )
            points_tensor = transform_points(points_tensor, w2c_tensor)
            depth_points_image = project_pointcloud_to_depth_map(
                points_tensor, fovx, fovy, buffers["depth"].shape[:2]
            )
            valid_mask = depth_points_image != 0
            x = buffers["depth"][:, :, 0][valid_mask].float()
            y = depth_points_image[valid_mask]
            # a, b = linear_least_squares_1d(x, y)
            (a, b), _ = ransac_linear_fit(x, y)
        else:
            a, b = (4.0, 0.0)
        buffers["depth"] = buffers["depth"] * a + b

        # Convert to depth to distance image
        position_image = transform_depth_to_position_image(
            buffers["depth"][:, :, 0], fovx, fovy
        )
        buffers["distance"] = torch.norm(position_image, dim=-1)

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
            albedo_image=buffers["albedo"],
            diffuse_image=buffers["diffuse"],
            glossy_image=buffers["glossy"],
            depth_image=buffers["distance"],
            normal_image=buffers["normal"],
            roughness_image=buffers["roughness"],
            metalness_image=buffers["metalness"],
            base_color_image=buffers["base_color"],
            brdf_image=buffers["brdf"],
            specular_image=buffers["specular"],
        )
        return cam_info

    def _get_buffer(self, frame_name: str, buffer_name: str) -> Tensor:
        file_name = frame_name.split("/")[-1]
        frame_id = file_name.split("_")[-1]
        buffer_path = os.path.join(
            self.buffers_dir, buffer_name, f"{buffer_name}_{frame_id}" + ".png"
        )
        if not os.path.isfile(buffer_path):
            return None

        buffer_image = Image.open(buffer_path)
        buffer = from_pil_image(buffer_image)
        if buffer_name in ["render", "irradiance", "diffuse", "glossy"]:
            buffer = untonemap(buffer)
            buffer /= 3.5  # Align exposure
        elif buffer_name in ["albedo", "base_color", "brdf"]:
            pass
        elif buffer_name in ["roughness", "metalness", "specular", "depth"]:
            buffer = repeat(buffer, "h w 1 -> h w 3")
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
