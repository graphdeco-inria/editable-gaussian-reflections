import os

import torch

GAUSS_TRACER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "libgausstracer.so"
)
LOADED = False


def make_raytracer(image_width: int, image_height: int, num_gaussians: int):
    global LOADED
    if not LOADED:
        torch.classes.load_library(GAUSS_TRACER_PATH)
        LOADED = True

    return torch.classes.gausstracer.Raytracer(image_width, image_height, num_gaussians)
