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


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, render_depth = False, secondary_view = None, iteration = None, raytracing_renderer=None, target=None, rays_from_camera=False, primal_pc=None, target_position=None, target_normal=None, target_roughness=None, target_F0=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    cam = secondary_view or viewpoint_camera
    
    # Set up rasterization configuration
    tanfovx = math.tan(cam.FoVx * 0.5)
    tanfovy = math.tan(cam.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height),
        image_width=int(cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=cam.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    def positional_encoding(positions, freqs=2):
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    mask = None
    
    if override_color is None:
        if pc.model_params.diffuse_only and not pc.model_params.dual and not pc.model_params.fused_scene:
            colors_precomp = pc._features_dc[:, 0]
        else:
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            view_pe = positional_encoding(dir_pp_normalized)

            if raytracing_renderer is not None:
                if pc.model_params.use_masks:
                    mask = (viewpoint_camera.glossy_image.sum(0) > 0).cuda()
                else:
                    mask = torch.ones_like(viewpoint_camera.glossy_image.sum(0))

                # roughness_map = viewpoint_camera.roughness_image.moveaxis(0, -1).flatten(0, 1).cuda()[mask.flatten(0, 1)]
                # metalness_map = viewpoint_camera.metalness_image.moveaxis(0, -1).flatten(0, 1).cuda()[mask.flatten(0, 1)]
                # albedo_map = viewpoint_camera.albedo_image.moveaxis(0, -1).flatten(0, 1).cuda()[mask.flatten(0, 1)]
                roughness_map = viewpoint_camera.roughness_image.moveaxis(0, -1).flatten(0, 1).cuda().mean(dim=-1)
                metalness_map = viewpoint_camera.metalness_image.moveaxis(0, -1).flatten(0, 1).cuda().mean(dim=-1)
                albedo_map = viewpoint_camera.albedo_image.moveaxis(0, -1).flatten(0, 1).cuda()

                refl_ray_o = viewpoint_camera.position_image.moveaxis(0, -1).flatten(0, 1).cuda()
                normal = viewpoint_camera.normal_image.moveaxis(0, -1).flatten(0, 1).cuda()
                if pc.model_params.optimize_normals:
                    normal_noise = torch.nn.functional.grid_sample(pc.mlp.normals.moveaxis(-1,0)[None], refl_ray_o[None, None, None])[0, :, 0, 0].mT * 10
                    normal = normal + normal_noise
                incident = refl_ray_o - viewpoint_camera.camera_center
                incident = incident / incident.norm(dim=-1, keepdim=True)
                
                relf_ray_d = incident - 2 * (incident * normal).sum(dim=-1).unsqueeze(-1) * normal

                if pc.model_params.brdf:
                    if pc.model_params.brdf_optimize:
                        if pc.model_params.brdf_f0_grid:
                            input_F0_image = torch.nn.functional.grid_sample(pc.mlp.F0_map.moveaxis(-1,0)[None], refl_ray_o[None, None, None])[0, :, 0, 0].reshape(viewpoint_camera.metalness_image.shape) 
                            #!!!* 10  # todo *10 is a hack to increase the lr 
                        n_dot_v = (-incident * normal).sum(dim=-1)
                        target_roughness_map = torch.zeros_like(n_dot_v)
                        uv = torch.stack([2 * n_dot_v - 1, 2 * target_roughness_map - 1], -1)
                        lut_values = F.grid_sample(brdf_lut[None, ...], uv[None, :, None, ...], align_corners=True)[0,...,0].reshape(input_F0_image.shape)
                        input_brdf_map = lut_values[0] * input_F0_image + lut_values[1] 
                    else:
                        
                        input_F0_image = torch.zeros_like(input_brdf_map)

                    raytracing_pkg = raytracing_renderer(refl_ray_o, relf_ray_d, mask, roughness_map, input_brdf_map, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, target=target, rays_from_camera=rays_from_camera, target_position=target_position, target_normal=target_normal, target_brdf_params=torch.cat([target_F0, target_roughness], dim=0) if target_F0 is not None else None)
                    rendered_image = raytracing_pkg["render"].moveaxis(-1, 0)
                    visibility_filter = raytracing_pkg["visibility_filter"]
                    
                    if pc.model_params.brdf_optimize:
                        if torch.is_grad_enabled():
                            if pc.model_params.brdf_f0_grid:
                                pc.mlp.F0_map.grad = torch.autograd.grad(input_brdf_map, pc.mlp.F0_map, grad_outputs=input_brdf_map.grad)[0]
                            else:
                                primal_pc._features_rest.grad = torch.autograd.grad(input_F0_image, primal_pc._features_rest, grad_outputs=input_brdf_map.grad)[0] # *** note the hard assignment
                else:
                    input_brdf_map = torch.ones_like(roughness_map)
                    raytracing_pkg = raytracing_renderer(refl_ray_o, relf_ray_d, mask, roughness_map, input_brdf_map, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, target=target, rays_from_camera=rays_from_camera, target_position=target_position, target_normal=target_normal)
                    rendered_image = raytracing_pkg["render"].moveaxis(-1, 0)
                    visibility_filter = raytracing_pkg["visibility_filter"]

                F0_image = raytracing_renderer.output_brdf_params[..., :3].clone().detach().reshape(*viewpoint_camera.position_image.shape[1:3], 3).moveaxis(-1, 0)
                roughness_image = raytracing_renderer.output_brdf_params[..., 3:4].clone().detach().reshape(*viewpoint_camera.position_image.shape[1:3], 1).moveaxis(-1, 0).repeat(3, 1, 1)
                position_image = raytracing_renderer.output_position_buffer.clone().detach().reshape(*viewpoint_camera.position_image.shape[1:3], 3).moveaxis(-1, 0)
                normal_image = raytracing_renderer.output_normal_buffer.clone().detach().reshape(*viewpoint_camera.normal_image.shape[1:3], 3).moveaxis(-1, 0)

                

                result = {
                    "render": rendered_image,
                    "roughness": roughness_image,
                    "F0": F0_image,
                    "position": position_image,
                    "normal": normal_image,
                    # todo brdf params
                    "brdf": input_brdf_map,
                    "refl_ray_o": refl_ray_o.moveaxis(-1, 0).reshape(*viewpoint_camera.normal_image.shape),
                    "refl_ray_d": relf_ray_d.moveaxis(-1, 0).reshape(*viewpoint_camera.normal_image.shape),
                    # "n_dot_v": n_dot_v.reshape(*viewpoint_camera.normal_image.shape[1:3]),
                    "viewspace_points": screenspace_points,
                    'mask': mask,
                    "visibility_filter": visibility_filter,
                }
                return result 
            
            assert not pc.model_params.fused_scene
  
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii, allmap
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }
    
    # if render_depth:
    #     with torch.no_grad():
    #         rendered_xyz = rasterizer(
    #             means3D = means3D,
    #             means2D = means2D,
    #             shs = None,
    #             colors_precomp = pc.get_xyz,
    #             opacities = opacity,
    #             scales = scales,
    #             rotations = rotations,
    #             cov3D_precomp = cov3D_precomp)[0]
    #         result["depth"] = (rendered_xyz - torch.from_numpy(viewpoint_camera.T)[None, :, None, None].cuda()).norm(dim=1, keepdim=True) / 20
    
    if False:
        # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        # transform normal from view space to world space
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
        
        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        # get depth distortion map
        render_dist = allmap[6:7]

        # psedo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1; 
        # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
        surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
        
        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        surf_normal = surf_normal * (render_alpha).detach()

        rets.update({
                'rend_alpha': render_alpha,
                'rend_normal': render_normal,
                'rend_dist': render_dist,
                'surf_depth': surf_depth,
                'surf_normal': surf_normal,
        })

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return rets
