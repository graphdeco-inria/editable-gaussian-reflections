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
# from bvh import RayTracer
import contextlib
import io 
from utils.point_utils import depth_to_normal
import cv2 
import torch.nn.functional as F 
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer.gaussian_raytracer import GaussianRaytracer
from scene.cameras import Camera

# import nerfacc
import numpy as np 

import os 

def render_pass(camera: Camera, gaussians: GaussianModel, raytracer: GaussianRaytracer, pipe_params: PipelineParams, bg_color: torch.Tensor,  is_diffuse_pass=False, diffuse_package=None, iteration=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if not is_diffuse_pass:
        assert diffuse_package is not None

    if gaussians.model_params.use_masks and not is_diffuse_pass:
        mask_map = camera.glossy_image.sum(0) > 0
    else:
        mask_map = torch.ones(1, camera.image_height, camera.image_width, device="cuda").bool()
        # mask_map = torch.tensor(0.0, device="cuda")

    # roughness_map = camera.sample_roughness_image().cuda().mean(dim=0, keepdim=True)
    roughness_map = torch.ones(1, camera.image_height, camera.image_width, device="cuda").bool()

    # todo simplify this code which duplicates what's already happening in cuda (maybe move it all to cuda)
    # *** moved this here for a sanity check, dont need to recompute when we have gt
    if is_diffuse_pass:
        # refl_ray_o = camera.sample_position_image() #not used anyways
        # refl_ray_d = camera.sample_reflection_ray_image() #not used anyways
        # refl_ray_o = refl_ray_o + gaussians.model_params.ray_offset * refl_ray_d
        refl_ray_o = torch.tensor(0.0, device="cuda")
        refl_ray_d = torch.tensor(0.0, device="cuda")
        # from torchvision.utils import save_image
        # save_image(torch.stack([refl_ray_o, refl_ray_d / 2 + 0.5], dim=0), "refl_ray.png")
    else:
        dim_x = camera.image_width
        dim_y = camera.image_height 
        vertical_fov_radians = camera.FoVy
        idx_y, idx_x = torch.meshgrid([torch.arange(dim_y, device="cuda"), torch.arange(dim_x, device="cuda")])
        aspect_ratio = float(dim_x) / float(dim_y)
        y = math.tan(vertical_fov_radians / 2) * (1.0 - 2.0 * ((idx_y + 1 / 2) / float(dim_y)))
        x = aspect_ratio * math.tan(vertical_fov_radians / 2) * (2.0 * ((idx_x + 1 / 2) / (float(dim_x))) - 1.0)
        R_colmap_init = torch.from_numpy(camera.R).clone().cuda().float() if isinstance(camera.R, np.ndarray) else camera.R.clone().cuda().float()
        _R_blender = -R_colmap_init
        _R_blender[:, 0] = -_R_blender[:, 0]
        R_blender = _R_blender # R_blender is c2w
        w2c_R_blender = R_blender.mT
        incident = F.normalize(w2c_R_blender[0] * x.unsqueeze(-1) + w2c_R_blender[1] * y.unsqueeze(-1) - w2c_R_blender[2], dim=-1)
        incident = incident / incident.norm(dim=-1, keepdim=True)
        used_normals = F.normalize(diffuse_package.normal, dim=0) if gaussians.model_params.use_attached_brdf and not gaussians.model_params.detach_normals else camera.sample_normal_image()
        reflection_ray_image = (incident - 2 * (incident * used_normals.moveaxis(0, -1)).sum(dim=-1).unsqueeze(-1) * used_normals.moveaxis(0, -1)).moveaxis(-1, 0)

        used_position = diffuse_package.position if gaussians.model_params.use_attached_brdf and not gaussians.model_params.detach_position else camera.sample_position_image()

        if not gaussians.model_params.precomp_ray:
            refl_ray_d = reflection_ray_image
            refl_ray_o = used_position + gaussians.model_params.ray_offset * refl_ray_d
    
    if is_diffuse_pass or gaussians.model_params.brdf_mode == "disabled":
        input_brdf_map = torch.ones(3, camera.image_height, camera.image_width, device="cuda")
    elif gaussians.model_params.brdf_mode == "gt":
        input_brdf_map = camera.sample_brdf_image()
    else:
        assert "lut" in gaussians.model_params.brdf_mode
        used_roughness = diffuse_package.roughness if gaussians.model_params.use_attached_brdf and not gaussians.model_params.detach_roughness else camera.sample_roughness_image().mean(dim=0, keepdim=True)
        used_F0 = diffuse_package.F0 if gaussians.model_params.use_attached_brdf and not gaussians.model_params.detach_F0 else camera.sample_F0_image()
        
        n_dot_v = (-incident.moveaxis(-1, 0) * used_normals).sum(dim=0).clamp(0) #*** was missing clamp 0
        uv = torch.stack([2 * n_dot_v - 1, 2 * torch.zeros_like(used_roughness[0]) - 1], -1)
        lut_values = F.grid_sample(gaussians.get_brdf_lut[None], uv[None], align_corners=True)
        input_brdf_map = lut_values[:, 0] * used_F0 + lut_values[:, 1]

    if is_diffuse_pass and torch.is_grad_enabled():
        target = camera.diffuse_image
        target_position = camera.sample_position_image()
        target_normal = camera.sample_normal_image()
        target_F0 = camera.sample_F0_image()
        target_roughness = camera.sample_roughness_image().mean(dim=0, keepdim=True) # todo do this averaging during image load
        target_brdf_params = torch.cat([target_F0, target_roughness], dim=0)
    else:
        if torch.is_grad_enabled():
            target = camera.glossy_image
        else:
            target = None
        target_position = None
        target_normal = None
        target_F0 = None
        target_roughness = None
        target_brdf_params = None   

    if iteration is not None and not is_diffuse_pass:
        do_backprop = torch.is_grad_enabled() and iteration > gaussians.model_params.warmup
    else:
        do_backprop = torch.is_grad_enabled()

    with torch.set_grad_enabled(do_backprop):
        raytracing_pkg = raytracer(refl_ray_o, refl_ray_d, mask_map, roughness_map, input_brdf_map, camera, gaussians, pipe_params, bg_color, target=target, rays_from_camera=is_diffuse_pass, target_position=target_position, target_normal=target_normal, target_brdf_params=target_brdf_params, is_diffuse_pass=is_diffuse_pass)

    if not is_diffuse_pass and gaussians.model_params.brdf_mode == "finetuned_lut" and torch.is_grad_enabled(): 
        gaussians._brdf_lut_residual.grad = torch.autograd.grad(input_brdf_map, gaussians._brdf_lut_residual, input_brdf_map.grad)[0]

    class package:
        "All of these results are reshaped to (C, H, W)"
        render = raytracing_pkg["render"].moveaxis(-1, 0)
        # todo don't sample again here
        roughness = raytracer.output_brdf_params[..., 3:4].clone().detach().reshape(*raytracing_pkg["render"].shape[:2], 1).moveaxis(-1, 0).repeat(3, 1, 1)
        F0 = raytracer.output_brdf_params[..., :3].clone().detach().reshape(*raytracing_pkg["render"].shape[:2], 3).moveaxis(-1, 0)
        position = raytracer.output_position_buffer.clone().detach().reshape(*raytracing_pkg["render"].shape[:2], 3).moveaxis(-1, 0)
        normal = raytracer.output_normal_buffer.clone().detach().reshape(*raytracing_pkg["render"].shape[:2], 3).moveaxis(-1, 0)
        mask = mask_map
        brdf = input_brdf_map

        "These results are reshaped to (num_gaussians, _)"
        visibility_filter = raytracing_pkg["visibility_filter"]

    return package 


def render(camera: Camera, gaussians: GaussianModel, pipe_params: PipelineParams, bg_color: torch.Tensor, raytracer: GaussianRaytracer, iteration=None):
    class package:
        diffuse = render_pass(camera, gaussians, raytracer, pipe_params, bg_color, is_diffuse_pass=True, iteration=iteration)
        glossy = render_pass(camera, gaussians, raytracer, pipe_params, bg_color, diffuse_package=diffuse, iteration=iteration)

    return package