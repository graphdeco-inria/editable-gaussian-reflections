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
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, GaussianRaytracer, render
import copy
import imageio
import shutil
import math 
import numpy as np 
import torch.nn.functional as F

# Set up command line argument parser
parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser)
pipeline = PipelineParams(parser)
# parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--train_views", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--mode", type=str, choices=["normal", "env_rot_1", "env_rot_2", "env_move_1", "env_move_2"], default="normal")
parser.add_argument("--skip_video", action="store_true")
parser.add_argument("--red_region", action="store_true")
args = get_combined_args(parser)
print("Rendering " + args.model_path)

model_params = model.extract(args)
gaussians = GaussianModel(model_params)
scene = Scene(model_params, gaussians)
raytracer = GaussianRaytracer(gaussians, scene.getTrainCameras()[0])

view = scene.getTrainCameras()[0]
package = render(view, gaussians, pipeline, None, raytracer=raytracer)
print(package.glossy.render.amax())
