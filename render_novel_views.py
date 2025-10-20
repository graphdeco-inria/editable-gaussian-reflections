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
from gaussian_tracing.utils.general_utils import set_seeds
from gaussian_tracing.utils.tonemapping import tonemap


@torch.no_grad()
def render_set(
    cfg,
    cameras,
    raytracer,
    save_dir,
):
    for idx, camera in enumerate(tqdm(cameras, desc="Rendering progress")):
        config = raytracer.cuda_module.get_config()
        if cfg.max_bounces > -1:
            config.num_bounces.copy_(cfg.max_bounces)

        if cfg.spp > 1:
            config.accumulate_samples.copy_(True)
            raytracer.cuda_module.reset_accumulators()
            for i in range(cfg.spp):
                package = render(
                    camera,
                    raytracer,
                    denoise=cfg.denoise,
                )
        else:
            package = render(
                camera,
                raytracer,
                denoise=cfg.denoise,
            )

        diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
        glossy_image = tonemap(package.rgb[1:].sum(dim=0)).clamp(0, 1)
        pred_image = tonemap(package.final.squeeze(0)).clamp(0, 1)

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
            "roughness": package.roughness[0],
            "f0": package.f0[0],
        }
        for k, v in result.items():
            save_path = os.path.join(save_dir, k, "{0:05d}".format(idx) + f"_{k}.png")
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchvision.utils.save_image(v, save_path)


def main(cfg: TyroConfig):
    # Initialize system state (RNG)
    set_seeds()

    gaussians = GaussianModel(cfg)
    scene = Scene(cfg, gaussians, load_iteration=cfg.iteration, shuffle=False)
    views = scene.getTrainCameras()

    raytracer = GaussianRaytracer(gaussians, views[0].image_width, views[0].image_height)

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
    if cfg.max_images is not None:
        cameras = cameras[: cfg.max_images]

    save_dir = os.path.join(cfg.model_path, "novel_views", f"ours_{scene.loaded_iter}")
    render_set(
        cfg,
        cameras,
        raytracer,
        save_dir,
    )


if __name__ == "__main__":
    cfg = tyro.cli(TyroConfig)
    main(cfg)
