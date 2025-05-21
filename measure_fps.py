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

import copy
import math
import os
import shutil
from argparse import ArgumentParser
from os import makedirs

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, GaussianRaytracer, render
from scene import Scene
from scene.tonemapping import *
from utils.general_utils import safe_state
from utils.image_utils import psnr


@torch.no_grad()
def render_set(
    scene,
    model_params,
    model_path,
    split,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    raytracer,
):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        package = render(view, raytracer, pipeline, background, force_update_bvh=False, targets_available=False)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    elapsed_sec = elapsed_ms / 1000.0
    fps = len(views) / elapsed_sec
    print(f"{fps:.2f} FPS")

    if "SKIP_WRITE" not in os.environ:
        with open(os.path.join(model_path, "fps.txt"), "w") as f:
            f.write(f"{fps:.2f}\n")


@torch.no_grad()
def render_sets(model_params: ModelParams, iteration: int, pipeline: PipelineParams):
    gaussians = GaussianModel(model_params)
    scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    raytracer = GaussianRaytracer(
        gaussians, viewpoint_stack[0].image_width, viewpoint_stack[0].image_height
    )

    if args.train_views or "REAL_SCENE" in os.environ:
        render_set(
            scene,
            model_params,
            model_params.model_path,
            "train",
            scene.loaded_iter,
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
            raytracer,
        )
    else:
        render_set(
            scene,
            model_params,
            model_params.model_path,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipeline,
            background,
            raytracer,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    _ = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    # Dummy repeat training args
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--flip_camera", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[
            1,
            100,
            500,
            1_000,
            2_500,
            5_000,
            10_000,
            20_000,
            30_000,
            60_000,
            90_000,
        ],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[1, 1_000, 2_500, 7_000, 15_000, 30_000, 60_000, 90_000],
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    # Rendering args
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--train_views", action="store_true")
    parser.add_argument("--skip_denoise", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--modes",
        type=str,
        choices=[
            "regular",
            "lod",
            "env_rot_1",
            "env_rot_2",
            "env_move_1",
            "env_move_2",
        ],
        default=["regular", "env_rot_1", "env_move_1", "env_move_2"],
        nargs="+",
    )  # env_rot_1 is at the scene's origin, env_rot_2 it somewhere in the far-field, env_move_1 dollys forward, env_move_2 trucks sideways # ["regular", "lod", "env_rot_1", "env_move_1", "env_move_2"]
    parser.add_argument(
        "--blur_sigmas", type=float, default=[None], nargs="+"
    )  # [None, 4.0, 16.0]
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--red_region", action="store_true")
    parser.add_argument("--skip_save_frames", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # if not args.train_views:
    #     args.max_images = min(100, args.max_images)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    model_params = model.extract(args)
    render_sets(model_params, args.iteration, pipeline.extract(args))
