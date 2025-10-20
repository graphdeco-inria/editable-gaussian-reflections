import torch


def test_raytracer_init():
    build_dir = "./gaussian_tracing/cuda/build"
    torch.classes.load_library(f"{build_dir}/libraytracer.so")

    image_width = 1536
    image_height = 1024
    num_gaussians = 1
    ppll_forward_size = 300_000_000
    ppll_backward_size = 200_000_000
    raytracer = torch.classes.raytracer.Raytracer(
        image_width, image_height, num_gaussians, ppll_forward_size, ppll_backward_size
    )

    rot = torch.eye(3)
    pos = torch.ones(3)
    camera = raytracer.get_camera()
    camera.znear.fill_(0.0001)
    camera.zfar.fill_(100.0)
    camera.set_pose(pos, rot)
