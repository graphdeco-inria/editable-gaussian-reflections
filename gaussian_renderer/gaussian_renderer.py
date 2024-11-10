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
import numpy as np 

import os 

def render_pass(camera: Camera, gaussians: GaussianModel, raytracer: GaussianRaytracer, pipe_params: PipelineParams, bg_color: torch.Tensor,  is_diffuse_pass=False, diffuse_package=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if not is_diffuse_pass:
        assert diffuse_package is not None

    if gaussians.model_params.use_masks and not is_diffuse_pass:
        mask_map = (camera.glossy_image.sum(0) > 0).cuda()
    else:
        mask_map = torch.ones_like(camera.glossy_image.sum(0))

    refl_ray_o_flat = camera.position_image.cuda() # todo don't depend on g.t.
    refl_ray_d_flat = camera.reflection_ray_image.cuda() # todo don't depend on g.t.
    refl_ray_o_flat = refl_ray_o_flat + gaussians.model_params.ray_offset * refl_ray_d_flat
    roughness_map_flat = camera.roughness_image.cuda().mean(dim=0, keepdim=True)
    
    if is_diffuse_pass or gaussians.model_params.brdf_mode == "disabled":
        input_brdf_map = torch.ones_like(camera.F0_image.cuda())
    elif gaussians.model_params.brdf_mode == "gt":
        input_brdf_map = camera.brdf_image.cuda()
    else:
        assert "lut" in gaussians.model_params.brdf_mode
        used_normals = diffuse_package.normal if gaussians.model_params.use_attached_brdf else camera.normal_image.cuda() # / 2 + 0.5
        used_roughness = diffuse_package.roughness if gaussians.model_params.use_attached_brdf else camera.roughness_image.cuda().mean(dim=0, keepdim=True)
        used_F0 = diffuse_package.F0 if gaussians.model_params.use_attached_brdf else camera.F0_image.cuda()

        incident = -F.normalize(refl_ray_o_flat.moveaxis(0, -1) - camera.camera_center, dim=-1).moveaxis(-1, 0) # todo don't depend on g.t.
        n_dot_v = (incident * used_normals).sum(dim=0).clamp(0) #*** was missing clamp 0
        uv = torch.stack([2 * n_dot_v - 1, 2 * used_roughness[0] - 1], -1)
        lut_values = F.grid_sample(gaussians.get_brdf_lut[None], uv[None], align_corners=True)
        input_brdf_map = lut_values[:, 0] * used_F0 + lut_values[:, 1]

    if is_diffuse_pass:
        target = camera.diffuse_image.cuda()
        target_position = camera.position_image.cuda()
        target_normal = camera.normal_image.cuda()
        target_F0 = camera.F0_image.cuda()
        target_roughness = camera.roughness_image.cuda().mean(dim=0, keepdim=True) # todo do this averaging during image load
        target_brdf_params = torch.cat([target_F0, target_roughness], dim=0)
    else:
        target = camera.glossy_image.cuda()
        target_position = None
        target_normal = None
        target_F0 = None
        target_roughness = None
        target_brdf_params = None

    raytracing_pkg = raytracer(refl_ray_o_flat, refl_ray_d_flat, mask_map, roughness_map_flat, input_brdf_map, camera, gaussians, pipe_params, bg_color,  target=target, rays_from_camera=is_diffuse_pass, target_position=target_position, target_normal=target_normal, target_brdf_params=target_brdf_params)

    if not is_diffuse_pass and gaussians.model_params.brdf_mode == "finetune_lut" and torch.is_grad_enabled(): 
        gaussians._brdf_lut_residual.grad = torch.autograd.grad(input_brdf_map, gaussians._brdf_lut_residual, input_brdf_map.grad)[0]


    class package:
        "All of these results are reshaped to (C, H, W)"
        render = raytracing_pkg["render"].moveaxis(-1, 0)
        roughness = raytracer.output_brdf_params[..., 3:4].clone().detach().reshape(*camera.position_image.shape[1:3], 1).moveaxis(-1, 0).repeat(3, 1, 1)
        F0 = raytracer.output_brdf_params[..., :3].clone().detach().reshape(*camera.position_image.shape[1:3], 3).moveaxis(-1, 0)
        position = raytracer.output_position_buffer.clone().detach().reshape(*camera.position_image.shape[1:3], 3).moveaxis(-1, 0)
        normal = raytracer.output_normal_buffer.clone().detach().reshape(*camera.normal_image.shape[1:3], 3).moveaxis(-1, 0)
        mask = mask_map
        brdf = input_brdf_map

        "These results are reshaped to (num_gaussians, _)"
        visibility_filter = raytracing_pkg["visibility_filter"]

    return package 


def render(camera: Camera, gaussians: GaussianModel, pipe_params: PipelineParams, bg_color: torch.Tensor, raytracer: GaussianRaytracer):
    class package:
        diffuse = render_pass(camera, gaussians, raytracer, pipe_params, bg_color, is_diffuse_pass=True)
        glossy = render_pass(camera, gaussians, raytracer, pipe_params, bg_color, diffuse_package=diffuse)

    return package