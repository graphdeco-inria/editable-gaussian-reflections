import torch


def test_raytracer_init():
    build_dir = "./gaussian_tracing/cuda/build"
    torch.classes.load_library(f"{build_dir}/libgausstracer.so")

    image_width = 1536
    image_height = 1024
    num_gaussians = 1
    raytracer = torch.classes.gausstracer.Raytracer(
        image_width, image_height, num_gaussians
    )

    rot = torch.eye(3)
    pos = torch.ones(3)
    znear = 0.0001
    zfar = 100.0
    max_lod = 0.05
    raytracer.set_camera(
        rot,
        pos,
        0.4710899591445923,
        znear,
        zfar,
        max_lod,
    )
