import os

import torch

GAUSS_TRACER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda", "build", "libraytracer.so")
LOADED = False


def make_raytracer(
    image_width: int,
    image_height: int,
    num_gaussians: int,
    ppll_forward_size: int = 200_000_000,
    ppll_backward_size: int = 150_000_000,
):
    global LOADED
    if not LOADED:
        torch.classes.load_library(GAUSS_TRACER_PATH)
        LOADED = True

    return torch.classes.raytracer.Raytracer(
        image_width, image_height, num_gaussians, ppll_forward_size, ppll_backward_size
    )
