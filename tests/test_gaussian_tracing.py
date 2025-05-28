import torch

from gaussian_tracing import make_raytracer


def test_raytracer_init():
    image_width = 1536
    image_height = 1024
    num_gaussians = 1
    raytracer = make_raytracer(image_width, image_height, num_gaussians)

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
