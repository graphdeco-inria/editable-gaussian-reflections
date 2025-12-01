import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.name == "nt":
    GAUSS_TRACER_PATH = os.path.join(
        BASE_DIR,
        "cuda", "build", "Release", "raytracer.dll"
    )
else:
    GAUSS_TRACER_PATH = os.path.join(
        BASE_DIR,
        "cuda", "build", "libraytracer.so"
    )
LOADED = False

def make_raytracer(
    image_width: int,
    image_height: int,
    num_gaussians: int,
    ppll_forward_size: int = 180_000_000,
    ppll_backward_size: int = 120_000_000,
):
    global LOADED
    if not LOADED:
        torch.classes.load_library(GAUSS_TRACER_PATH)
        LOADED = True

    return torch.classes.raytracer.Raytracer(image_width, image_height, num_gaussians, ppll_forward_size, ppll_backward_size)


