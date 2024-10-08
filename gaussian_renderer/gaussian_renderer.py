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
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer #as SurfelRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from bvh import RayTracer
import contextlib
import io 
from utils.point_utils import depth_to_normal

import nerfacc
from gauss_render import * 


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, render_depth = False, secondary_view = None, iteration = None, nomask = False):
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
    
    if override_color is None:
        if pc.model_params.diffuse_only and not pc.model_params.dual:
            colors_precomp = pc.get_features.squeeze(1)
        else:
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            view_pe = positional_encoding(dir_pp_normalized)

            if pc.model_params.dual:
                mask = (viewpoint_camera.glossy_image.sum(0) > 0).flatten(0, 1).cuda()
                if nomask:
                    mask = torch.ones_like(mask)
                raytracer = RayTracer(pc.get_xyz, pc.get_scaling, pc.get_rotation)
                ray_o = viewpoint_camera.position_image.moveaxis(0, -1).flatten(0, 1).cuda()[mask]
                normal = viewpoint_camera.normal_image.moveaxis(0, -1).flatten(0, 1).cuda()[mask]
                if pc.model_params.optimize_normals:
                    normal_noise = torch.nn.functional.grid_sample(pc.mlp.normals.moveaxis(-1,0)[None], ray_o[None, None, None])[0, :, 0, 0].mT * 10
                    normal = normal + normal_noise
                incident = ray_o - viewpoint_camera.camera_center
                incident = incident / incident.norm(dim=-1, keepdim=True)
                ray_d = incident - 2 * (incident * normal).sum(dim=-1).unsqueeze(-1) * normal
                hits = raytracer.trace(ray_o, ray_d, pc.get_xyz, pc.get_inverse_covariance(), pc.get_opacity)

                roughness_map = viewpoint_camera.roughness_image.moveaxis(0, -1).flatten(0, 1).cuda()[mask]
                metalness_map = viewpoint_camera.metalness_image.moveaxis(0, -1).flatten(0, 1).cuda()[mask]
                albedo_map = viewpoint_camera.albedo_image.moveaxis(0, -1).flatten(0, 1).cuda()[mask]

                cov3d = build_covariance_3d(pc.get_scaling, pc.get_rotation)
                ho = ray_o[hits.ray_ids]
                hd = ray_d[hits.ray_ids]

                # compute the point of gaussian evaluation for each intersection, which is the peak of the gaussian along the ray
                tmp = (cov3d[hits.gaussian_ids] @ hd.unsqueeze(-1)).squeeze(-1)
                t_max = ((pc.get_xyz[hits.gaussian_ids] - ho) * tmp).sum(dim=-1) / (hd * tmp).sum(dim=-1)
                eval_pos = ho + hd * t_max.unsqueeze(-1)
                
                # eval the guassians at these points
                dx = (eval_pos - pc.get_xyz[hits.gaussian_ids])  
                tmp2 = (cov3d.inverse()[hits.gaussian_ids] @ dx.unsqueeze(-1)).squeeze(-1)
                gauss_weight = torch.exp(-0.5*(dx * tmp2).sum(dim=-1))
                
                opacity = pc.get_opacity[hits.gaussian_ids].squeeze(-1)
                if pc.model_params.optimize_roughness:
                    roughness = torch.nn.functional.grid_sample(pc.mlp.roughnesses.moveaxis(-1,0)[None], ray_o[None, None, None])[0, :, 0, 0].mT.mean(dim=-1) * 10  # *10 is a hack to increase the lr
                else:
                    roughness = roughness_map[hits.ray_ids].mean(dim=-1)
                metalness = metalness_map[hits.ray_ids].mean(dim=-1)
                slope = pc._features_rest[:, 0, -1][hits.gaussian_ids]
                mult = 1.0 - (slope * roughness[hits.ray_ids]).clamp(0, 1)
                alphas = gauss_weight * opacity * mult
                
                colors = pc.get_features[:, 0][hits.gaussian_ids] 

                weights, transmittances = nerfacc.render_weight_from_alpha(alphas=alphas, ray_indices=hits.ray_ids.long())
                pixel_colors = nerfacc.accumulate_along_rays(weights=weights, values=colors, ray_indices=hits.ray_ids.long(), n_rays=ray_o.shape[0])
                
                reflectivity = torch.zeros_like(pixel_colors)
                if pc.model_params.optimize_reflectivity:
                    reflectivity = torch.nn.functional.grid_sample(pc.mlp.reflectivies.moveaxis(-1,0)[None], ray_o[None, None, None])[0, :, 0, 0].mT * 10  # *10 is a hack to increase the lr 
                    eye = -incident
                    # halfway = (eye + ray_d) / (eye + ray_d).norm(dim=-1, keepdim=True)
                    r0 = reflectivity
                    fresnel = r0 + (1.0 - r0) * (1.0 - (normal * eye).sum(dim=-1, keepdim=True).clamp(0))**5
                    #??? what about the lambert term 
                    pixel_colors = pixel_colors * fresnel #!!! eye or halfway vector?
                    # https://google.github.io/filament/Filament.html#materialsystem/specularbrdf uses halfway
                    # https://learnopengl.com/PBR/Theory uses halfway
                elif pc.model_params.mlp_brdf:
                    mlp_in = torch.cat([roughness_map.mean(dim=-1, keepdim=True), metalness_map.mean(dim=-1, keepdim=True), albedo_map, normal, incident, ray_d], dim=-1) 
                    brdf = pc.mlp(positional_encoding(mlp_in))
                else:
                    brdf = 1.0
                
                rendered_image = torch.zeros_like(viewpoint_camera.position_image.moveaxis(0, -1).flatten(0, 1)).cuda()
                rendered_image[mask] = pixel_colors 
                rendered_image = rendered_image.reshape(*viewpoint_camera.normal_image.shape[1:3], 3).moveaxis(-1, 0)
                
                if pc.model_params.optimize_roughness:
                    roughness_image = torch.zeros_like(viewpoint_camera.position_image.moveaxis(0, -1).flatten(0, 1)).cuda()
                    roughness_image[mask] = roughness.unsqueeze(-1).repeat(1, 3) 
                    roughness_image = roughness_image.reshape(*viewpoint_camera.normal_image.shape[1:3], 3).moveaxis(-1, 0)
                else:
                    roughness_image = viewpoint_camera.roughness_image

                if pc.model_params.optimize_reflectivity:
                    reflectivity_image = torch.zeros_like(viewpoint_camera.position_image.moveaxis(0, -1).flatten(0, 1)).cuda()
                    reflectivity_image[mask] = reflectivity
                    reflectivity_image = reflectivity_image.reshape(*viewpoint_camera.normal_image.shape[1:3], 3).moveaxis(-1, 0)
                else:
                    reflectivity_image = viewpoint_camera.albedo_image * viewpoint_camera.metalness_image

                if pc.model_params.optimize_normals:
                    normal_image = torch.zeros_like(viewpoint_camera.position_image.moveaxis(0, -1).flatten(0, 1)).cuda()
                    normal_image[mask] = normal
                    normal_image = normal_image.reshape(*viewpoint_camera.normal_image.shape[1:3], 3).moveaxis(-1, 0)
                else:
                    normal_image = viewpoint_camera.normal_image

                result = {
                    "render": rendered_image,
                    "roughness": roughness_image,
                    "reflectivity": reflectivity_image,
                    "normal": normal_image,
                    "viewspace_points": screenspace_points,
                }
                return result 
  
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
    rendered_image, radii, allmap = rasterizer(
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
