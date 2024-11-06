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
# from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer #as SurfelRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from bvh import RayTracer
import contextlib
import io 
from utils.point_utils import depth_to_normal
import cv2 
import torch.nn.functional as F 
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer.gaussian_raytracer import GaussianRaytracer
from scene.cameras import Camera

import nerfacc
from gauss_render import * 
import numpy as np 

import os 


brdf_lut_path = "data/ibl_brdf_lut.png"
brdf_lut = cv2.imread(brdf_lut_path)
brdf_lut = cv2.cvtColor(brdf_lut, cv2.COLOR_BGR2RGB)
brdf_lut = brdf_lut.astype(np.float32)
brdf_lut /= 255.0
brdf_lut = torch.tensor(brdf_lut).to("cuda")
brdf_lut = brdf_lut.permute((2, 0, 1))


def render(camera: Camera, gaussians: GaussianModel, raytracer: GaussianRaytracer, pipe_params: PipelineParams, bg_color: torch.Tensor,  is_diffuse_pass=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    if gaussians.model_params.use_masks and not is_diffuse_pass:
        mask_map = (camera.glossy_image.sum(0) > 0).cuda()
    else:
        mask_map = torch.ones_like(camera.glossy_image.sum(0))

    refl_ray_o_flat = camera.position_image.cuda()
    refl_ray_d_flat = camera.reflection_ray_image.cuda()
    roughness_map_flat = camera.roughness_image.cuda().mean(dim=0, keepdim=True)
    
    if gaussians.model_params.brdf:
        input_brdf_map = camera.brdf_image.cuda()
    else:
        input_brdf_map = torch.ones_like(roughness_map_flat)

    if is_diffuse_pass:
        target = camera.diffuse_image.cuda()
        target_position = camera.position_image.cuda()
        target_normal = camera.normal_image.cuda()
        target_F0 = camera.F0_image.cuda()
        target_roughness = camera.roughness_image.cuda().mean(dim=0, keepdim=True)
        target_brdf_params = torch.cat([target_F0, target_roughness], dim=0)
    else:
        target = camera.glossy_image.cuda()
        target_position = None
        target_normal = None
        target_F0 = None
        target_roughness = None
        target_brdf_params = None

    raytracing_pkg = raytracer(refl_ray_o_flat, refl_ray_d_flat, mask_map, roughness_map_flat, input_brdf_map, camera, gaussians, pipe_params, bg_color,  target=target, rays_from_camera=is_diffuse_pass, target_position=target_position, target_normal=target_normal, target_brdf_params=target_brdf_params)

    class package:
        "All of these results are reshaped to (C, H, W)"
        render = raytracing_pkg["render"].moveaxis(-1, 0)
        roughness = raytracer.output_brdf_params[..., 3:4].clone().detach().reshape(*camera.position_image.shape[1:3], 1).moveaxis(-1, 0).repeat(3, 1, 1)
        F0 = raytracer.output_brdf_params[..., :3].clone().detach().reshape(*camera.position_image.shape[1:3], 3).moveaxis(-1, 0)
        position = raytracer.output_position_buffer.clone().detach().reshape(*camera.position_image.shape[1:3], 3).moveaxis(-1, 0)
        normal = raytracer.output_normal_buffer.clone().detach().reshape(*camera.normal_image.shape[1:3], 3).moveaxis(-1, 0)
        mask = mask_map

        "These results are reshaped to (num_gaussians, _)"
        visibility_filter = raytracing_pkg["visibility_filter"]

        # todo also include losses


    return package 


def render_multipass(camera: Camera, gaussians: GaussianModel, pipe_params: PipelineParams, bg_color: torch.Tensor, raytracer: GaussianRaytracer):
    class package:
        diffuse = render(camera, gaussians, raytracer, pipe_params, bg_color, is_diffuse_pass=True)
        glossy = render(camera, gaussians, raytracer, pipe_params, bg_color)

    return package