import os

import torch

if os.name == 'nt':
    __PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GAUSS_TRACER_PATH = os.path.join(__PARENT_DIR, "build\\Release\\gray.dll")
else:
    GAUSS_TRACER_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
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

    if os.name == 'nt':
        GAUSS_TRACER_PATH = os.path.dirname(os.path.dirname(GAUSS_TRACER_PATH))

    return torch.classes.raytracer.Raytracer(image_width, image_height, num_gaussians, ppll_forward_size, ppll_backward_size)


