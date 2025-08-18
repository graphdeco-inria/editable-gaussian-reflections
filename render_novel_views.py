#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from copy import deepcopy

import numpy as np
import torch
import torchvision
import tyro
from einops import rearrange
from tqdm import tqdm

from gaussian_tracing.arguments import (
    TyroConfig,
)
from gaussian_tracing.renderer import GaussianRaytracer, render
from gaussian_tracing.scene import GaussianModel, Scene
from gaussian_tracing.utils.cam_utils import generate_spiral_path
from gaussian_tracing.utils.general_utils import safe_state
from gaussian_tracing.utils.tonemapping import tonemap


@torch.no_grad()
def render_set(
    cfg,
    cameras,
    background,
    raytracer,
    save_dir,
):
    pipe_params = cfg.pipe_params

    for idx, camera in enumerate(tqdm(cameras, desc="Rendering progress")):
        raytracer.cuda_module.denoise.copy_(not cfg.skip_denoiser)

        if cfg.spp > 1:
            raytracer.cuda_module.accumulate.copy_(True)
            raytracer.cuda_module.accumulated_rgb.zero_()
            raytracer.cuda_module.accumulated_normal.zero_()
            raytracer.cuda_module.accumulated_sample_count.zero_()
            for i in range(cfg.spp):
                package = render(
                    camera, raytracer, pipe_params, background
                )
        else:
            package = render(
                camera, raytracer, pipe_params, background
            )

        if cfg.supersampling > 1:
            for key, value in package.__dict__.items():
                batched = value.ndim == 4
                resized = torch.nn.functional.interpolate(
                    value[None] if not batched else value,
                    scale_factor=1.0 / cfg.supersampling,
                    mode="area",
                )
                setattr(package, key, resized[0] if not batched else resized)

        diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
        glossy_image = tonemap(package.rgb[1:-1].sum(dim=0)).clamp(0, 1)
        pred_image = tonemap(package.rgb[-1]).clamp(0, 1)
        ray_origin = raytracer.cuda_module.output_ray_origin[0].moveaxis(-1, 0).abs()
        ray_direction = raytracer.cuda_module.output_ray_direction[0].moveaxis(-1, 0)

        # Match normal image with EnvGS visualization
        R_tensor = torch.tensor(camera.R.T, dtype=torch.float32)
        normal_image = package.normal[0].cpu()
        normal_image = rearrange(normal_image, "c h w -> h w c")
        normal_image = normal_image / torch.norm(normal_image, dim=-1, keepdim=True)
        normal_image = torch.einsum("ij,...j->...i", R_tensor, normal_image)
        normal_image = rearrange(normal_image, "h w c -> c h w")
        normal_image *= -1
        normal_image[0, :, :] *= -1  # To match EnvGS visualization.

        result = {
            "render": pred_image,
            "glossy": glossy_image,
            "diffuse": diffuse_image,
            "depth": package.depth[0] / package.depth[0].amax(),
            "normal": normal_image * 0.5 + 0.5,
            "ray_origin": ray_origin / 5,
            "ray_direction": ray_direction * 0.5 + 0.5,
            "roughness": package.roughness[0],
            "F0": package.F0[0],
        }
        for k, v in result.items():
            save_path = os.path.join(save_dir, k, "{0:05d}".format(idx) + f"_{k}.png")
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchvision.utils.save_image(v, save_path)


def main(cfg: TyroConfig):
    # Initialize system state (RNG)
    safe_state(cfg.quiet)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

    gaussians = GaussianModel(cfg)
    scene = Scene(cfg, gaussians, load_iteration=cfg.iteration, shuffle=False)
    views = scene.getTrainCameras()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    raytracer = GaussianRaytracer(
        gaussians, views[0].image_width, views[0].image_height
    )
    if cfg.spp > 1:
        raytracer.cuda_module.denoise.fill_(False)

    # Create spiral path from EnvGS
    camtoworlds_all = []
    for view in views:
        w2c = np.eye(4)
        w2c[:3, :3] = view.R.T
        w2c[:3, 3] = view.T
        c2w = np.linalg.inv(w2c)
        camtoworlds_all.append(c2w)
    camtoworlds_all = np.array(camtoworlds_all)
    camtoworlds_all = generate_spiral_path(camtoworlds_all)
    camtoworlds_all = np.concatenate(
        [
            camtoworlds_all,
            np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0),
        ],
        axis=1,
    )  # [N, 4, 4]

    # Create cameras from camera path
    cameras = []
    for c2w in camtoworlds_all:
        camera = deepcopy(views[0])
        w2c = np.linalg.inv(c2w)
        camera.R = np.transpose(w2c[:3, :3])
        camera.T = w2c[:3, 3]
        camera.update()
        cameras.append(camera)

    save_dir = os.path.join(cfg.model_path, "novel_views", f"ours_{scene.loaded_iter}")
    render_set(
        cfg,
        cameras,
        background,
        raytracer,
        save_dir,
    )


if __name__ == "__main__":
    cfg = tyro.cli(TyroConfig)

    # TODO: Remove this custom config modification.
    cfg.resolution *= cfg.supersampling

    main(cfg)
