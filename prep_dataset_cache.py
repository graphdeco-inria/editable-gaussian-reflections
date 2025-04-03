import sys
import glob
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import torch
import numpy as np
from tqdm import tqdm

path = sys.argv[1]

input_passes = [
    "render",
    "diffuse",
    "glossy",
    "normal",
    "position",
    "roughness",
    "specular",
    "metalness",
    "base_color",
    "glossy_brdf",
]


def imread(image_path, render_pass_name):
    path = (
        image_path.replace("/images/", "/render/")
        .replace("/colmap/", "/renders/")
        .replace("/render_", f"/{render_pass_name}_")
        .replace("/render/", f"/{render_pass_name}/")
    )
    path = path.replace(".png", ".exr")
    assert os.path.exists(path), f"{render_pass_name} render pass not found at {path}"
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


assert path.startswith("renders/")
cache_path = path.replace("renders/", "cache/")

if WRITE := True:
    for split in ["train", "test"]:
        os.makedirs(os.path.join(cache_path, split), exist_ok=True)
        for i in tqdm(range(200 if split == "train" else 100)):
            frame = []
            for render_pass in input_passes:
                image_path = f"{path}/{split}/{render_pass}/{render_pass}_{i:04d}.png"
                image = imread(image_path, render_pass)
                frame.append(image)
            torch.save(
                torch.stack(frame, dim=0).half(), f"{cache_path}/{split}/{i:04d}.pt"
            )
else:
    for split in ["test"]:
        for i in tqdm(range(200 if split == "train" else 100)):
            torch.load(f"{cache_path}/{split}/{i:04d}.pt")
