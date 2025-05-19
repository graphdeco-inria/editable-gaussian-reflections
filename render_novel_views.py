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
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, GaussianRaytracer, render
from scene import Scene
from scene.tonemapping import tonemap
from utils.cam_utils import generate_spiral_path
from utils.general_utils import safe_state


@torch.no_grad()
def render_set(
    cameras,
    pipeline,
    background,
    raytracer,
    save_dir,
):
    for idx, camera in enumerate(tqdm(cameras, desc="Rendering progress")):
        raytracer.cuda_module.denoise.copy_(not args.skip_denoiser)

        if args.spp > 1:
            raytracer.cuda_module.accumulate.copy_(True)
            raytracer.cuda_module.accumulated_rgb.zero_()
            raytracer.cuda_module.accumulated_normal.zero_()
            raytracer.cuda_module.accumulated_sample_count.zero_()
            for i in range(args.spp):
                package = render(
                    camera, raytracer, pipeline, background, blur_sigma=None
                )
        else:
            package = render(camera, raytracer, pipeline, background, blur_sigma=None)

        if args.supersampling > 1:
            for key, value in package.__dict__.items():
                batched = value.ndim == 4
                resized = torch.nn.functional.interpolate(
                    value[None] if not batched else value,
                    scale_factor=1.0 / args.supersampling,
                    mode="area",
                )
                setattr(package, key, resized[0] if not batched else resized)

        diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
        glossy_image = tonemap(package.rgb[1:-1].sum(dim=0)).clamp(0, 1)
        pred_image = tonemap(package.rgb[-1]).clamp(0, 1)
        result = {
            "render": pred_image,
            "glossy": glossy_image,
            "diffuse": diffuse_image,
            "position": package.position[0],
            "depth": package.depth[0] / package.depth[0].amax(),
            "normal": package.normal[0] / 2 + 0.5,
            "ray_origin": raytracer.cuda_module.output_ray_origin[0].moveaxis(-1, 0).abs() / 5,
            "ray_direction": raytracer.cuda_module.output_ray_direction[0].moveaxis(-1, 0) / 2 + 0.5,
            "roughness": package.roughness[0],
            "F0": package.F0[0],
        }
        for k, v in result.items():
            save_path = os.path.join(save_dir, k, "{0:05d}".format(idx) + f"_{k}.png")
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchvision.utils.save_image(v, save_path)


@torch.no_grad()
def render_sets(model_params: ModelParams, iteration: int, pipeline: PipelineParams):
    gaussians = GaussianModel(model_params)
    scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)
    views = scene.getTrainCameras()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    raytracer = GaussianRaytracer(
        gaussians, views[0].image_width, views[0].image_height
    )
    if args.spp > 1:
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

    save_dir = os.path.join(
        model_params.model_path, "novel_views", f"ours_{scene.loaded_iter}"
    )
    render_set(
        cameras,
        pipeline,
        background,
        raytracer,
        save_dir,
    )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    _ = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    # Rendering args
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--spp", default=8, type=int)
    parser.add_argument("--supersampling", default=1, type=int)
    parser.add_argument("--skip_denoiser", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    model_params = model.extract(args)
    model_params.resolution *= args.supersampling
    render_sets(model_params, args.iteration, pipeline.extract(args))
