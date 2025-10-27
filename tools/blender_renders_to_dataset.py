import sys, os
sys.path.append(os.getcwd())

import shutil 
import os 
import glob
import imageio.v3 as iio
import json
import numpy as np
import tqdm
import torch
from dataclasses import dataclass, field
from typing import *
import tyro
import torch.nn.functional as F
import cv2 
import tifffile
import safetensors.torch
from editable_gauss_refl.utils.tonemapping import tonemap

ALWAYS_PRESERVE = ["depth"]

@dataclass
class ExtractDatasetCLI:
    scene: str # *e.g. "shiny_kitchen"
    src_root: str = "data/blender_renders/{scene}"
    dst_root: str = "data/renders_{format}_{precision}bits{extra}/{scene}"
    resolution: int = 768
    format: str = "png"
    precision: Literal[8, 16, 32] = 8
    preserve: List[Literal["images", "depth", "normals"]] = field(default_factory=lambda: ALWAYS_PRESERVE)
    collapsed: bool = False
    exposure: float = 3.5 
    
    def __post_init__(self):
        assert self.format in ["safetensors", "exr", "tiff", "png", "st_mixed_depth"]
        assert self.precision in [8, 16, 32, "mixed_depth", "mixed_images", "mixed_both"]
        if self.format == "png":
            assert self.precision == 8, "PNG format only supports 8-bit precision"
        elif self.format == "tiff":
            assert self.precision == 16, "TIFF format only supports 16-bit precision"
        elif self.format == "exr":
            assert self.precision == 32, "EXR format only supports 32-bit precision"
        if self.preserve != ALWAYS_PRESERVE:
            extra = "_preserve_" + "_".join(self.preserve)
        else:
            extra = ""
        self.src_root = self.src_root.format(scene=self.scene)
        self.dst_root = self.dst_root.format(format=self.format, precision=self.precision, scene=self.scene, extra=extra)

# * Parse cli
cli = tyro.cli(ExtractDatasetCLI)

# * Delete dst if exists
if os.path.exists(cli.dst_root):
    shutil.rmtree(cli.dst_root)
os.makedirs(cli.dst_root, exist_ok=True)

# * Copy point cloud 
pc_src = os.path.join(cli.src_root, "point_cloud_dense.ply")
if os.path.exists(pc_src):
    shutil.copy2(pc_src, cli.dst_root)

# * Copy all .mp4 and .json files
for pattern in ("*.mp4", "*.json"):
    for src_path in glob.glob(os.path.join(cli.src_root, pattern)):
        shutil.copy2(src_path, cli.dst_root)

# * Parse trnasforms
transforms = {}
transforms["train"] = json.load(open(os.path.join(cli.src_root, "transforms_train.json")))
transforms["test"] = json.load(open(os.path.join(cli.src_root, "transforms_test.json")))

# * Prep and copy all images
subdirs = [
    "base_color",
    "diffuse",
    "depth",
    "glossy",
    "metalness",
    "normal",
    "position",
    "roughness",
    "specular",
    "render"
]
for split in ["train", "test"]:
    src_dir = os.path.join(cli.src_root, split)
    dst_dir = os.path.join(cli.dst_root, split)
    os.makedirs(dst_dir, exist_ok=True) 

    for i in tqdm.tqdm(range({"train": 200, "test": 100}[split])):
        # * Load images for viewpoint
        images = {}
        for subdir in subdirs:
            src_subdir = os.path.join(src_dir, subdir)
            dst_subdir = os.path.join(dst_dir, subdir)
            img = cv2.cvtColor(cv2.imread(os.path.join(src_subdir, f"{subdir}_{i:04d}.exr"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            images[subdir] = img

        # * Reformat images
        dtype = np.float32
        render = images["render"].astype(dtype)
        diffuse = images["diffuse"].astype(dtype)
        specular = images["glossy"].astype(dtype)
        depth = np.linalg.norm(images["position"] - np.array(transforms[split]["frames"][i]["transform_matrix"])[:3, 3], axis=-1, keepdims=True).astype(dtype)
        f0 = ((1.0 - images["metalness"]) * 0.08 * images["specular"] + images["metalness"] * images["base_color"]).astype(dtype)
        normal = images["normal"].astype(dtype)
        roughness = np.mean(images["roughness"].astype(dtype), axis=-1, keepdims=True)

        # * Exposure adjust
        render = render * cli.exposure
        diffuse = diffuse * cli.exposure
        specular = specular * cli.exposure  

        # * Resize images
        def resize(arr):
            resized = cv2.resize(
                arr, 
                dsize=(int(1.5 * cli.resolution), cli.resolution), 
                interpolation=cv2.INTER_AREA
            )
            if resized.ndim == 2:
                resized = resized[:, :, None]
            return resized
        
        render = resize(render)
        diffuse = resize(diffuse)
        specular = resize(specular)
        depth = resize(depth)
        f0 = resize(f0)
        normal = resize(normal)
        roughness = resize(roughness)

        # * If 8 bit, restrict values to [0, 1]
        if cli.precision == 8:
            if "images" not in cli.preserve:
                render = tonemap(torch.from_numpy(render)).numpy()
                diffuse = tonemap(torch.from_numpy(diffuse)).numpy()
                specular = tonemap(torch.from_numpy(specular)).numpy()
            if "depth" not in cli.preserve:
                depth = (depth - 1) / (3 - 1)
            if "normals" not in cli.preserve:
                normal = normal / 2 + 0.5

        # * Save outputs
        if cli.format == "safetensors":
            if cli.precision == 32:
                dtype = torch.float32
                dtype_images = torch.float32
                dtype_depth = torch.float32
                dtype_normals = torch.float32
            elif cli.precision == 16:
                dtype = torch.float16
                dtype_images = torch.float16
                dtype_depth = torch.float16
                dtype_normals = torch.float16
            elif cli.precision == 8:
                if "images" in cli.preserve:
                    dtype_images = torch.float16
                else:
                    render = np.clip(render * 255, 0, 255)
                    diffuse = np.clip(diffuse * 255, 0, 255)
                    specular = np.clip(specular * 255, 0, 255)
                    dtype_images = torch.uint8
                if "depth" in cli.preserve:
                    dtype_depth = torch.float16
                else:
                    depth = np.clip(depth * 255, 0, 255)
                    dtype_depth = torch.uint8
                f0 = np.clip(f0 * 255, 0, 255)
                if "normals" in cli.preserve:
                    dtype_normals = torch.float16
                else:
                    normal = np.clip(normal * 255, 0, 255)
                    dtype_normals = torch.uint8
                roughness = np.clip(roughness * 255, 0, 255)
                dtype = torch.uint8
            safetensors.torch.save_file({
                "render": torch.from_numpy(render).permute(2, 0, 1).to(dtype_images).contiguous(),
                "diffuse": torch.from_numpy(diffuse).permute(2, 0, 1).to(dtype_images).contiguous(),
                "specular": torch.from_numpy(specular).permute(2, 0, 1).to(dtype_images).contiguous(),
                "depth": torch.from_numpy(depth).permute(2, 0, 1).to(dtype_depth).contiguous(),
                "f0": torch.from_numpy(f0).permute(2, 0, 1).to(dtype).contiguous(),
                "normal": torch.from_numpy(normal).permute(2, 0, 1).to(dtype_normals).contiguous(),
                "roughness": torch.from_numpy(roughness).permute(2, 0, 1).to(dtype).contiguous(),
            }, os.path.join(dst_dir, f"buffers_{i:04d}.safetensors"))
        else:
            os.makedirs(os.path.join(dst_dir, "render"), exist_ok=True)
            os.makedirs(os.path.join(dst_dir, "diffuse"), exist_ok=True)
            os.makedirs(os.path.join(dst_dir, "specular"), exist_ok=True)
            os.makedirs(os.path.join(dst_dir, "depth"), exist_ok=True)
            os.makedirs(os.path.join(dst_dir, "f0"), exist_ok=True)
            os.makedirs(os.path.join(dst_dir, "normal"), exist_ok=True)
            os.makedirs(os.path.join(dst_dir, "roughness"), exist_ok=True)
            if cli.format == "exr":
                cv2.imwrite(
                    os.path.join(dst_dir, "render", f"render_{i:04d}.exr"),
                    cv2.cvtColor(render, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(dst_dir, "diffuse", f"diffuse_{i:04d}.exr"),
                    cv2.cvtColor(diffuse, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(dst_dir, "specular", f"specular_{i:04d}.exr"),
                    cv2.cvtColor(specular, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(dst_dir, "depth", f"depth_{i:04d}.exr"),
                    np.squeeze(depth),
                )
                cv2.imwrite(
                    os.path.join(dst_dir, "f0", f"f0_{i:04d}.exr"),
                    cv2.cvtColor(f0, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(dst_dir, "normal", f"normal_{i:04d}.exr"),
                    cv2.cvtColor(normal, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(dst_dir, "roughness", f"roughness_{i:04d}.exr"),
                    np.squeeze(roughness),
                )
            elif cli.format == "tiff":
                tifffile.imwrite(
                    os.path.join(dst_dir, "render", f"render_{i:04d}.tiff"),
                    render.astype(np.float16),
                )
                tifffile.imwrite(
                    os.path.join(dst_dir, "diffuse", f"diffuse_{i:04d}.tiff"),
                    diffuse.astype(np.float16),
                )
                tifffile.imwrite(
                    os.path.join(dst_dir, "specular", f"specular_{i:04d}.tiff"),
                    specular.astype(np.float16),
                )
                tifffile.imwrite(
                    os.path.join(dst_dir, "depth", f"depth_{i:04d}.tiff"),
                    np.squeeze(depth).astype(np.float16),
                )
                tifffile.imwrite(
                    os.path.join(dst_dir, "f0", f"f0_{i:04d}.tiff"),
                    f0.astype(np.float16),
                )
                tifffile.imwrite(
                    os.path.join(dst_dir, "normal", f"normal_{i:04d}.tiff"),
                    normal.astype(np.float16),
                )
                tifffile.imwrite(
                    os.path.join(dst_dir, "roughness", f"roughness_{i:04d}.tiff"),
                    np.squeeze(roughness).astype(np.float16),
                )
            elif cli.format == "png":
                if "images" in cli.preserve:
                    tifffile.imwrite(
                        os.path.join(dst_dir, "render", f"render_{i:04d}.tiff"),
                        render.astype(np.float16),
                    )
                    tifffile.imwrite(
                        os.path.join(dst_dir, "diffuse", f"diffuse_{i:04d}.tiff"),
                        diffuse.astype(np.float16),
                    )
                    tifffile.imwrite(
                        os.path.join(dst_dir, "specular", f"specular_{i:04d}.tiff"),
                        specular.astype(np.float16),
                    )
                else:
                    iio.imwrite(os.path.join(dst_dir, "render", f"render_{i:04d}.png"), (np.clip(render, 0, 1) * 255).astype(np.uint8))
                    iio.imwrite(os.path.join(dst_dir, "diffuse", f"diffuse_{i:04d}.png"), (np.clip(diffuse, 0, 1) * 255).astype(np.uint8))
                    iio.imwrite(os.path.join(dst_dir, "specular", f"specular_{i:04d}.png"), (np.clip(specular, 0, 1) * 255).astype(np.uint8))
                if "depth" in cli.preserve:
                    tifffile.imwrite(
                        os.path.join(dst_dir, "depth", f"depth_{i:04d}.tiff"),
                        np.squeeze(depth).astype(np.float16),
                    )
                else:
                    iio.imwrite(os.path.join(dst_dir, "depth", f"depth_{i:04d}.png"), (np.clip(np.squeeze(depth), 0, 1) * 255).astype(np.uint8))
                iio.imwrite(os.path.join(dst_dir, "f0", f"f0_{i:04d}.png"), (np.clip(f0, 0, 1) * 255).astype(np.uint8))
                if "normals" in cli.preserve:
                    tifffile.imwrite(
                        os.path.join(dst_dir, "normal", f"normal_{i:04d}.tiff"),
                        normal.astype(np.float16),
                    )
                else:
                    iio.imwrite(os.path.join(dst_dir, "normal", f"normal_{i:04d}.png"), (np.clip(normal, 0, 1) * 255).astype(np.uint8))
                iio.imwrite(os.path.join(dst_dir, "roughness", f"roughness_{i:04d}.png"), (np.clip(np.squeeze(roughness), 0, 1) * 255).astype(np.uint8))