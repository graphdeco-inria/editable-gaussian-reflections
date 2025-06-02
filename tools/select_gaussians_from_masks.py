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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_tracing.arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel, GaussianRaytracer, render
import copy
import imageio
import shutil
import math
import numpy as np
import torch.nn.functional as F
from scene.tonemapping import *
import torchvision.transforms.functional as TF
from PIL import Image
import random
import os
from torchvision.utils import save_image

os.environ["DIFFUSE_LOSS_WEIGHT"] = str(0.0)
os.environ["GLOSSY_LOSS_WEIGHT"] = str(0.0)
os.environ["NORMAL_LOSS_WEIGHT"] = str(0.0)
os.environ["POSITION_LOSS_WEIGHT"] = str(0.0)
os.environ["F0_LOSS_WEIGHT"] = str(0.0)
os.environ["ROUGHNESS_LOSS_WEIGHT"] = str(1.0)

bg = torch.tensor([0.0, 0.0, 0.0]).cuda()

if __name__ == "__main__":
    parser = ArgumentParser(description="Selection script parameters")
    model = ModelParams(parser, sentinel=False)
    _ = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--mask_name")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--load_iteration", default=-1, type=int)

    args = get_combined_args(parser)
    model_params = model.extract(args)
    pipe_params = pipeline.extract(args)

    gaussians = GaussianModel(model_params)
    scene = Scene(
        model_params, gaussians, load_iteration=args.load_iteration, shuffle=False
    )
    raytracer = GaussianRaytracer(gaussians, scene.getTestCameras()[0])
    raytracer.rebuild_bvh()
    gaussians.init_empty_grads()

    cams = scene.getTestCameras()
    masks_path = (
        model_params.source_path.replace("/colmap/", "/renders/")
        + "/test/mask_"
        + args.mask_name
        + "/"
    )
    masks = list(
        tqdm(
            [
                TF.to_tensor(
                    Image.open(
                        masks_path + f"/mask_{args.mask_name}_{i:04d}.png"
                    ).convert("RGB")
                ).cuda()
                for i in range(len(cams))
            ],
            desc="Loading masks",
        )
    )

    with torch.no_grad():
        gaussians._roughness *= 0
        gaussians._roughness -= 5
        gaussians._opacity.clamp_(max=0)

    adam = torch.optim.Adam([gaussians._roughness], lr=1e-1)
    for i in range(args.iters):
        j = random.randint(0, len(cams) - 1)
        viewpoint = cams[j]
        mask = masks[j]

        downsized_mask = F.interpolate(
            mask[None], size=viewpoint.roughness_image.shape[1:], mode="area"
        )[0]
        viewpoint._roughness_image.copy_(downsized_mask)

        package = render(viewpoint, raytracer, pipe_params, bg)

        adam.step()
        adam.zero_grad(set_to_none=False)
        raytracer.zero_grad()

        if i % 100 == 0:
            print(".")
            save_image(
                torch.stack([package.roughness[0], downsized_mask]), f"mask.ignore.png"
            )

    torch.save(
        gaussians._roughness, model_params.model_path + f"/select_{args.mask_name}.pt"
    )
    print("Saved selection for mask:", args.mask_name)
