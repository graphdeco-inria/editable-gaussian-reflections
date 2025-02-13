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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import contextlib
import io 
from utils.point_utils import depth_to_normal
import cv2 
import torch.nn.functional as F 
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer.gaussian_raytracer import GaussianRaytracer
from scene.cameras import Camera
import numpy as np 
import os 


def render(camera: Camera, raytracer: GaussianRaytracer, pipe_params: PipelineParams, bg_color: torch.Tensor, iteration=None):
    target = camera.original_image
    target_diffuse = camera.diffuse_image
    target_glossy = camera.glossy_image
    target_position = camera.sample_position_image()
    target_normal = camera.sample_normal_image()
    target_f0 = camera.sample_F0_image()
    target_roughness = camera.sample_roughness_image().mean(dim=0, keepdim=True) # todo do this averaging during image load
    target_brdf = camera.sample_brdf_image()

    if iteration is not None:
        do_backprop = torch.is_grad_enabled() and iteration > raytracer.pc.model_params.warmup
    else:
        do_backprop = torch.is_grad_enabled()

    with torch.set_grad_enabled(do_backprop):
        raytracing_pkg = raytracer(camera, pipe_params, bg_color, target=target, target_diffuse=target_diffuse, target_glossy=target_glossy, target_position=target_position, target_normal=target_normal, target_roughness=target_roughness, target_f0=target_f0, target_brdf=target_brdf)

    class package:
        "All of these results are reshaped to (C, H, W)"

        rgb = raytracer.cuda_raytracer.output_rgb.clone().detach().moveaxis(-1, 1)

        if raytracer.cuda_raytracer.output_position is not None:
            position = raytracer.cuda_raytracer.output_position.clone().detach().moveaxis(-1, 1)
        else:
            position = torch.zeros_like(rgb)

        if raytracer.cuda_raytracer.output_normal is not None:
            normal = raytracer.cuda_raytracer.output_normal.clone().detach().moveaxis(-1, 1)
        else:
            normal = torch.zeros_like(rgb)

        if raytracer.cuda_raytracer.output_roughness is not None:
            roughness = raytracer.cuda_raytracer.output_roughness.clone().detach().moveaxis(-1, 1).repeat(1, 3, 1, 1)
        else:
            roughness = torch.zeros_like(rgb)

        if raytracer.cuda_raytracer.output_f0 is not None:
            F0 = raytracer.cuda_raytracer.output_f0.clone().detach().moveaxis(-1, 1)
        else:
            F0 = torch.zeros_like(rgb)
        
        if raytracer.cuda_raytracer.output_brdf is not None:
            brdf = raytracer.cuda_raytracer.output_brdf.clone().detach().moveaxis(-1, 1)
        else:
            brdf = torch.zeros_like(rgb)
        
    return package