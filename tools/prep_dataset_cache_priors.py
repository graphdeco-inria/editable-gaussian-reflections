import sys
import glob
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image 

path = sys.argv[1]

if len(sys.argv) > 2:
    target_height = int(sys.argv[2])
else: 
    target_height = None

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


def resize_by_height(image: torch.Tensor, target_height: int) -> torch.Tensor:
    w, h = image.shape[1], image.shape[0]
    scale = target_height / h
    target_width = int(w * scale)
    resized = torch.nn.functional.interpolate(
        image.moveaxis(-1, 0)[None],
        (target_height, target_width),
        mode="area",
    )[0].moveaxis(0, -1)
    return torch.tensor(np.array(resized))

def imread(image_path, render_pass_name):
    path = (
        image_path.replace("/images/", "/render/")
        .replace("/colmap/", "/priors/")
        .replace("/render_", f"/{render_pass_name}_")
        .replace("/render/", f"/{render_pass_name}/")
    )
    path = path.replace(".png", ".exr")
    assert os.path.exists(path), f"{render_pass_name} render pass not found at {path}"
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if target_height is not None:
        image = resize_by_height(image, target_height)
    return image


assert path.startswith("priors/")
cache_path = path.replace("priors/", f"cache_{target_height}/") if target_height is not None else path.replace("priors/", "cache/")

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
