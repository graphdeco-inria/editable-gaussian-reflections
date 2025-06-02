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

import torch
from gaussian_tracing.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from gaussian_tracing.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_tracing.arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel, GaussianRaytracer, render
import copy
import imageio
import shutil
import math
import numpy as np
import torch.nn.functional as F
from gaussian_tracing.utils.tonemapping import tonemap, untonemap
from torchvision.utils import save_image

background = torch.tensor([0.0, 0.0, 0.0]).cuda()

if __name__ == "__main__":
    parser = ArgumentParser(description="Edit script parameters")
    model = ModelParams(parser, sentinel=False)
    _ = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--mask_name")
    parser.add_argument("--load_iteration", default=-1, type=int)
    parser.add_argument("--selection_threshold", default=0.5, type=float)
    parser.add_argument(
        "--edit",
        type=str,
        choices=["make_mirror", "make_bright", "delete", "make_rougher", "turn_pink"],
        default="make_mirror",
    )
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    model_params = model.extract(args)
    pipe_params = pipeline.extract(args)

    gaussians = GaussianModel(model_params)
    scene = Scene(
        model_params, gaussians, load_iteration=args.load_iteration, shuffle=False
    )
    raytracer = GaussianRaytracer(gaussians, scene.getTestCameras()[0])
    raytracer.rebuild_bvh()

    cams = scene.getTestCameras()
    mask = (
        torch.load(model_params.model_path + f"/select_{args.mask_name}.pt")
        > args.selection_threshold
    ).squeeze(-1)

    with torch.no_grad():
        if args.edit == "make_bright":
            gaussians._diffuse[mask] *= 0.0
            gaussians._diffuse[mask] += 1.0
            gaussians._roughness[mask] *= 0.0
            gaussians._roughness[mask] -= 1000.0
        elif args.edit == "make_rougher":
            gaussians._roughness[mask] += 0.9
        elif args.edit == "delete":
            gaussians._opacity[mask] *= 0.0
            gaussians._opacity[mask] -= 1000.0
        elif args.edit == "make_mirror":
            pass
            gaussians._diffuse[mask] *= 0.0
            gaussians._diffuse[mask] -= 1000.0
            gaussians._f0[mask] *= 0.0
            gaussians._f0[mask] += 0.5
            gaussians._roughness[mask] *= 0.0
            gaussians._roughness[mask] -= 1000.0
        elif args.edit == "turn_pink":
            gaussians._diffuse[mask] *= 0
            gaussians._diffuse[mask, 0:1] = 0.7
            gaussians._diffuse[mask, 1:2] = -100.0  # account for softplus

        mask = (
            torch.load(model_params.model_path + f"/select_shelf.pt")
            > args.selection_threshold
        ).squeeze(-1)
        gaussians._diffuse[mask, 2:3] += 1.0
        gaussians._diffuse[mask, 0:2] = -5.0  # account for softplus

    safe_state(args.quiet)

    with torch.no_grad():
        package = render(scene.getTestCameras()[0], raytracer, pipeline, background)

    save_image(tonemap(package.rgb[-1]), "edit.ignore.jpg")
