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
from utils.general_utils import colormap
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, GaussianRaytracer
import sys
from scene import Scene, GaussianModel, SurfelModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
import copy 
from utils.graphics_utils import BasicPointCloud
from datetime import datetime

# Set up command line argument parser
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument('--flip_camera', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 100, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 30_000, 60_000, 90_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 1_000, 2_500, 7_000, 15_000, 30_000, 60_000, 90_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--viewer", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default = None)
args = parser.parse_args(sys.argv[1:])
args.save_iterations.append(args.iterations)
torch.autograd.set_detect_anomaly(args.detect_anomaly)
model_params = lp.extract(args)
opt_params = op.extract(args)
pipe_params = pp.extract(args)

# print("Optimizing " + args.model_path)

# # Initialize system state (RNG)
# safe_state(args.quiet)

# # Start GUI server, configure and run training
# if args.viewer:
#     network_gui.init(args.ip, args.port)

# first_iter = 0
# tb_writer = prepare_output_and_logger(model_params) 

# gaussians = GaussianModel(model_params)
# scene = Scene(model_params, gaussians)
# # gaussians.training_setup(opt_params)

# viewpoint_stack = scene.getTrainCameras().copy()
# raytracer = GaussianRaytracer(gaussians, viewpoint_stack[0])

model_params = model_params
gaussians = GaussianModel(model_params)
scene = Scene(model_params, gaussians)
raytracer = GaussianRaytracer(gaussians, scene.getTrainCameras()[0])


with torch.no_grad():
    view = scene.getTrainCameras()[0]
    package = render(view, gaussians, pipe_params, None, raytracer=raytracer)
print("wat2:", package.glossy.render.amax())
