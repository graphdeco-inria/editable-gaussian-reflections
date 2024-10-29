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

import os 
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from torchvision.utils import save_image

torch.classes.load_library("/home/ypoirier/optix/gausstracer/build/libgausstracer.so")

DEBUG_SCENE = False

class RaytracingRenderer:
    def __init__(self, pc: GaussianModel, example_camera):
        self.image_sizes_buffer = torch.tensor([example_camera.image_width, example_camera.image_height], device="cuda")
        self.vertical_fov_radians_buffer = torch.tensor(example_camera.FoVy, dtype=torch.float32, device="cuda")

        self.ray_origins_buffer = torch.zeros(example_camera.image_height * example_camera.image_width, 3, dtype=torch.float32, device="cuda")
        self.ray_directions_buffer = torch.zeros(example_camera.image_height * example_camera.image_width, 3, dtype=torch.float32, device="cuda")

        self.mask_buffer = torch.ones(example_camera.image_height * example_camera.image_width, dtype=torch.int32, device="cuda")
        self.roughness_buffer = torch.ones(example_camera.image_height * example_camera.image_width, dtype=torch.float32, device="cuda")
        self.brdf_buffer = torch.ones(example_camera.image_height * example_camera.image_width, 3, dtype=torch.float32, device="cuda")
        self.brdf_buffer_grad = torch.ones(example_camera.image_height * example_camera.image_width, 3, dtype=torch.float32, device="cuda")
        
        self.camera_c2w_rot_buffer = torch.zeros(3, 3, dtype=torch.float32, device="cuda")
        self.camera_w2c_rot_buffer = torch.zeros(3, 3, dtype=torch.float32, device="cuda")
        self.camera_position_buffer = torch.zeros(3, dtype=torch.float32, device="cuda")

        self.gaussian_scales_buffer = pc.get_scaling.detach().requires_grad_()
        self.gaussian_scales_buffer_grad = torch.zeros_like(self.gaussian_scales_buffer)

        self.gaussian_extra_features_buffer = pc.get_scaling.detach().requires_grad_()
        self.gaussian_extra_features_buffer_grad = torch.zeros_like(self.gaussian_extra_features_buffer)

        self.gaussian_rotations_buffer = pc.get_rotation.detach().requires_grad_()
        self.gaussian_rotations_buffer_grad = torch.zeros_like(self.gaussian_rotations_buffer)

        self.gaussian_xyz_buffer = pc.get_xyz.detach().requires_grad_()
        self.gaussian_xyz_buffer_grad = torch.zeros_like(self.gaussian_xyz_buffer)

        self.gaussian_opacity_buffer = pc.get_opacity.detach().requires_grad_()
        self.gaussian_opacity_buffer_grad = torch.zeros_like(self.gaussian_opacity_buffer)

        self.gaussian_rgb_buffer = torch.zeros_like(pc.get_xyz).detach().requires_grad_()
        self.gaussian_rgb_buffer_grad = torch.zeros_like(self.gaussian_rgb_buffer)

        self.gaussian_normal_buffer = torch.zeros_like(pc.get_xyz).detach().requires_grad_()
        self.gaussian_normal_buffer_grad = torch.zeros_like(self.gaussian_normal_buffer)

        self.gaussian_F0_buffer = torch.zeros_like(pc.get_xyz).detach().requires_grad_()
        self.gaussian_F0_buffer_grad = torch.zeros_like(self.gaussian_F0_buffer)

        self.gaussian_roughness_buffer = torch.zeros_like(pc.get_xyz).detach().requires_grad_()
        self.gaussian_roughness_buffer_grad = torch.zeros_like(self.gaussian_roughness_buffer)

        self.output_visibility_buffer = torch.zeros(example_camera.image_height * example_camera.image_width, dtype=torch.int32, device="cuda")

        self.loss_tensor = torch.tensor([0.0], device="cuda")

        self.output_rgbt_buffer = torch.zeros(example_camera.image_height, example_camera.image_width, 4, dtype=torch.float32, device="cuda")
        self.output_normal_buffer = torch.zeros(example_camera.image_height, example_camera.image_width, 3, dtype=torch.float32, device="cuda")
        self.output_position_buffer = torch.zeros(example_camera.image_height, example_camera.image_width, 3, dtype=torch.float32, device="cuda")
        self.output_F0_buffer = torch.zeros(example_camera.image_height, example_camera.image_width, 3, dtype=torch.float32, device="cuda")
        self.output_roughness_buffer = torch.zeros(example_camera.image_height, example_camera.image_width, 1, dtype=torch.float32, device="cuda")
        self.output_brdf_buffer = torch.zeros(example_camera.image_height, example_camera.image_width, 3, dtype=torch.float32, device="cuda")

        self.target_rgb_buffer = torch.zeros(example_camera.image_height, example_camera.image_width, 3, dtype=torch.float32, device="cuda")
        self.target_normal_buffer = torch.zeros(example_camera.image_height, example_camera.image_width, 3, dtype=torch.float32, device="cuda")
        self.target_F0_buffer = torch.zeros(example_camera.image_height, example_camera.image_width, 3, dtype=torch.float32, device="cuda")
        self.target_roughness_buffer = torch.zeros(example_camera.image_height, example_camera.image_width, 1, dtype=torch.float32, device="cuda")
    
        # todo rid of this
        import sys 
        sys.path.append("/home/ypoirier/optix/gausstracer")
        import opt_shared_code as ref

        #     #
        #     gaussian_normal,
        #     gaussian_normal.grad,
        #     gaussian_F0,
        #     gaussian_F0.grad,
        #     gaussian_roughness,
        #     gaussian_roughness.grad,
        #     #

        self.raytracer = torch.classes.gausstracer.Raytracer(
            self.image_sizes_buffer,
            self.vertical_fov_radians_buffer,
            #
            self.camera_c2w_rot_buffer, 
            self.camera_w2c_rot_buffer, 
            self.camera_position_buffer,
            #
            self.ray_origins_buffer,
            self.ray_directions_buffer,
            self.mask_buffer,
            #
            self.roughness_buffer,
            self.brdf_buffer,
            self.brdf_buffer_grad,
            #
            self.gaussian_rgb_buffer,
            self.gaussian_rgb_buffer_grad,
            self.gaussian_opacity_buffer,
            self.gaussian_opacity_buffer_grad,
            #
            self.gaussian_normal_buffer,
            self.gaussian_normal_buffer_grad,
            self.gaussian_F0_buffer,
            self.gaussian_F0_buffer_grad,
            self.gaussian_roughness_buffer,
            self.gaussian_roughness_buffer_grad,
            #
            self.gaussian_extra_features_buffer,
            self.gaussian_extra_features_buffer_grad,
            #
            self.output_rgbt_buffer,
            self.output_normal_buffer,
            self.output_position_buffer,
            self.output_F0_buffer,
            self.output_roughness_buffer,
            self.output_brdf_buffer,
            #
            self.target_rgb_buffer,
            self.target_normal_buffer,
            self.target_F0_buffer,
            self.target_roughness_buffer,
            #
            self.gaussian_scales_buffer,
            self.gaussian_scales_buffer_grad,
            self.gaussian_rotations_buffer,
            self.gaussian_rotations_buffer_grad,
            self.gaussian_xyz_buffer,
            self.gaussian_xyz_buffer_grad,
            #
            self.loss_tensor,
            #
            self.output_visibility_buffer,
            ref.output_max_radii2d,
            ref.background_rgb,
            ref.full_T,
            ref.gaussian_mask,
            ref.random_numbers
        )

        if False:
            self.optim = torch.optim.Adam([
                dict(params=[pc._opacity], lr=0.025),
                dict(params=[pc._features_dc, pc._features_rest], lr=0.0025),
                dict(params=[pc._xyz], lr=0.00016), 
                dict(params=[pc._rotation], lr=0.001),
                dict(params=[pc._scaling], lr=0.0005) # *** 0.005 in 3dgs
            ],  betas=[0.9, 0.999]) 

        self.pc = pc

    @torch.no_grad()
    def rebuild_bvh(self):
        new_size = self.pc.get_scaling.shape[0]

        self.gaussian_scales_buffer.requires_grad_(False).requires_grad_(False).resize_(new_size, 3).requires_grad_()
        self.gaussian_scales_buffer_grad.resize_(new_size, 3)
        self.gaussian_rotations_buffer.requires_grad_(False).resize_(new_size, 4).requires_grad_()
        self.gaussian_rotations_buffer_grad.resize_(new_size, 4)
        self.gaussian_xyz_buffer.requires_grad_(False).resize_(new_size, 3).requires_grad_()
        self.gaussian_xyz_buffer_grad.resize_(new_size, 3)
        self.gaussian_opacity_buffer.requires_grad_(False).resize_(new_size, 1).requires_grad_()
        self.gaussian_opacity_buffer_grad.resize_(new_size, 1)
        self.gaussian_rgb_buffer.requires_grad_(False).resize_(new_size, 3).requires_grad_()
        self.gaussian_rgb_buffer_grad.resize_(new_size, 3)
        self.gaussian_extra_features_buffer.requires_grad_(False).resize_(new_size, self.gaussian_extra_features_buffer.shape[-1]).requires_grad_()
        self.gaussian_extra_features_buffer_grad.resize_(new_size, self.gaussian_extra_features_buffer.shape[-1])

        self.gaussian_scales_buffer.copy_(self.pc.get_scaling) 
        self.gaussian_rotations_buffer.copy_(self.pc.get_rotation)
        self.gaussian_xyz_buffer.copy_(self.pc.get_xyz)
        self.gaussian_opacity_buffer.copy_(self.pc.get_opacity)
        self.gaussian_rgb_buffer.copy_(self.pc._features_dc[:, 0])
        self.gaussian_extra_features_buffer.copy_(self.pc._features_rest[:, 0]) 
        
        self.output_visibility_buffer.resize_(new_size)
        self.output_visibility_buffer.zero_()

        self.raytracer.rebuild_bvh()   

    def zero_grad(self):
        self.gaussian_scales_buffer_grad.zero_()
        self.gaussian_rotations_buffer_grad.zero_()
        self.gaussian_xyz_buffer_grad.zero_()
        self.gaussian_opacity_buffer_grad.zero_()
        self.gaussian_rgb_buffer_grad.zero_()
        self.gaussian_extra_features_buffer_grad.zero_()
        self.brdf_buffer_grad.zero_()

    def __call__(self, ray_o, ray_d, mask, roughness, brdf, viewpoint_camera, pc : GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier = 1.0, override_color = None, target = None, rays_from_camera=False):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        # colors_precomp = None
        # if override_color is None:
        #     colors_precomp = run_sh(pc, viewpoint_camera)
        # else:
        #     colors_precomp = override_color

        colors_precomp = pc._features_dc[:, 0].sigmoid() #!!

        # Temporary solution, use splatting to measure the radii
        # with torch.no_grad():
        #     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        #     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        #     try:
        #         screenspace_points.retain_grad()
        #     except:
        #         pass
            
        #     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        #     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        #     raster_settings = GaussianRasterizationSettings(
        #         image_height=int(viewpoint_camera.image_height),
        #         image_width=int(viewpoint_camera.image_width),
        #         tanfovx=tanfovx,
        #         tanfovy=tanfovy,
        #         bg=torch.tensor([0.0, 0.0, 0.0]).cuda(),
        #         scale_modifier=scaling_modifier,
        #         viewmatrix=viewpoint_camera.world_view_transform,
        #         projmatrix=viewpoint_camera.full_proj_transform,
        #         sh_degree=pc.active_sh_degree,
        #         campos=viewpoint_camera.camera_center,
        #         prefiltered=False,
        #         debug=pipe.debug
        #     )
        #     means3D = pc.get_xyz
        #     means2D = screenspace_points

        #     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        #     # scaling / rotation by the rasterizer.
        #     scales = None
        #     rotations = None
        #     cov3D_precomp = None
        #     if pipe.compute_cov3D_python:
        #         cov3D_precomp = pc.get_covariance(scaling_modifier)
        #     else:
        #         scales = pc.get_scaling
        #         rotations = pc.get_rotation
            
        #     rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
        #     _, radii = rasterizer(
        #         means3D = means3D.data,
        #         means2D = means2D,
        #         shs = pc.get_features,
        #         colors_precomp = None,
        #         opacities = pc.get_opacity,
        #         scales = scales,
        #         rotations = rotations,
        #         cov3D_precomp = cov3D_precomp
        #     )

        scaling = pc.get_scaling
        rotation = pc.get_rotation
        xyz = pc.get_xyz
        opacity = pc.get_opacity

        with torch.no_grad():
            R = torch.from_numpy(viewpoint_camera.R).cuda().float()
            R_c2w_blender = -R 
            R_c2w_blender[:, 0] = -R_c2w_blender[:, 0] #*** these do match blender json correctly

            self.camera_c2w_rot_buffer.copy_(R_c2w_blender.contiguous()) 
            self.camera_w2c_rot_buffer.copy_(R_c2w_blender.mT.contiguous()) 
            self.camera_position_buffer.copy_(viewpoint_camera.camera_center.contiguous()) 

            if False:
                self.gaussian_opacity_buffer.data.clamp_(0.0 + 1e-4, 1.0 - 1e-4)  #!!
                self.gaussian_rgb_buffer.data.clamp_(0.0 + 1e-4, 1.0 - 1e-4)
                self.gaussian_scales_buffer.data.clamp_(1e-4, 1000.0) 
                self.gaussian_rotations_buffer.copy_(torch.nn.functional.normalize(self.gaussian_rotations_buffer, dim=-1))

            self.gaussian_scales_buffer.copy_(scaling) 
            self.gaussian_rotations_buffer.copy_(rotation)
            self.gaussian_xyz_buffer.copy_(xyz)
            self.gaussian_opacity_buffer.copy_(opacity)
            self.gaussian_rgb_buffer.copy_(colors_precomp)
            self.gaussian_extra_features_buffer.copy_(pc._features_rest.squeeze(1))
            self.mask_buffer.copy_(mask.flatten())
            self.roughness_buffer.copy_(roughness.flatten())
            self.output_visibility_buffer.zero_()
            if pc.model_params.brdf:
                self.brdf_buffer.copy_(brdf.moveaxis(0, 2).flatten(0, 1))
            self.vertical_fov_radians_buffer.copy_(torch.tensor(viewpoint_camera.FoVy, dtype=torch.float32, device="cuda"))
            if rays_from_camera:
                self.ray_origins_buffer.zero_()
                self.ray_directions_buffer.zero_()
            else:
                self.ray_origins_buffer.copy_(ray_o)
                self.ray_directions_buffer.copy_(ray_d)
                
        if target is not None:
            self.target_rgb_buffer.copy_(target.moveaxis(0, -1).clone()) 

        if "CHECK_NAN" in os.environ:
            if self.camera_c2w_rot_buffer.isnan().any() or self.camera_position_buffer.isnan().any() or self.vertical_fov_radians_buffer.isnan().any() or self.gaussian_scales_buffer.isnan().any() or self.gaussian_rotations_buffer.isnan().any() or self.gaussian_xyz_buffer.isnan().any() or self.gaussian_opacity_buffer.isnan().any() or self.gaussian_rgb_buffer.isnan().any():
                raise Exception("NaNs in input buffers!")
            if self.output_rgbt_buffer.isnan().any():
                raise Exception("NaNs in output buffers!")

        torch.cuda.synchronize()
        self.raytracer.update_bvh()
        self.raytracer.raytrace()
        torch.cuda.synchronize()

        if "CHECK_NAN" in os.environ:
            if self.gaussian_scales_buffer_grad.isnan().any():
                raise Exception("NaNs in scale gradients!")
            if self.gaussian_rotations_buffer_grad.isnan().any():
                raise Exception("NaNs in rotation gradients!")
            if self.gaussian_xyz_buffer_grad.isnan().any():
                raise Exception("NaNs in xyz gradients!")
            if self.gaussian_opacity_buffer_grad.isnan().any():
                raise Exception("NaNs in opacity gradients!")
            if self.gaussian_rgb_buffer_grad.isnan().any():
                raise Exception("NaNs in color gradients!")

        if torch.is_grad_enabled():
            with torch.no_grad():
                pc._scaling.grad.add_(torch.autograd.grad(scaling, [pc._scaling], grad_outputs=self.gaussian_scales_buffer_grad)[0])
                pc._rotation.grad.add_(torch.autograd.grad(rotation, [pc._rotation], grad_outputs=self.gaussian_rotations_buffer_grad)[0])
                pc._xyz.grad.add_(torch.autograd.grad(xyz, [pc._xyz], grad_outputs=self.gaussian_xyz_buffer_grad)[0])
                pc._opacity.grad.add_(torch.autograd.grad(opacity, [pc._opacity], grad_outputs=self.gaussian_opacity_buffer_grad)[0])
                pc._features_dc.grad.add_(self.gaussian_rgb_buffer_grad.unsqueeze(1))
                pc._features_rest.grad.add_(torch.autograd.grad(pc._features_rest, [pc._features_rest], grad_outputs=self.gaussian_extra_features_buffer_grad.unsqueeze(1))[0])
                if pc.model_params.brdf:
                    assert not pc.model_params.fused_scene
                    brdf_grad = self.brdf_buffer_grad.moveaxis(0, 1).reshape(brdf.shape)
                    if brdf.grad is None:
                        brdf.grad = brdf_grad
                    else:
                        brdf.grad.add_(brdf_grad)

        if torch.is_grad_enabled():
            if False:
                self.optim.step()
        else:
            assert self.gaussian_scales_buffer_grad.sum().item() == 0.0
            assert self.gaussian_rotations_buffer_grad.sum().item() == 0.0
            assert self.gaussian_xyz_buffer_grad.sum().item() == 0.0
            assert self.gaussian_opacity_buffer_grad.sum().item() == 0.0
            assert self.gaussian_rgb_buffer_grad.sum().item() == 0.0

        # screenspace_points.grad = (viewpoint_camera.full_proj_transform.T[:3, :3] @ self.gaussian_xyz_buffer_grad.T).T

        self.gaussian_rgb_buffer_grad.zero_()
        self.gaussian_scales_buffer_grad.zero_()
        self.gaussian_rotations_buffer_grad.zero_()
        self.gaussian_xyz_buffer_grad.zero_()
        self.gaussian_opacity_buffer_grad.zero_()

        return {
            "render": self.output_rgbt_buffer[:, :, :3].clone(),
            "visibility_filter" : self.output_visibility_buffer.clone(),
            # "viewspace_points": screenspace_points,
            # "radii": radii
        }
        

def run_sh(pc, viewpoint_camera):
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    return torch.clamp_min(sh2rgb + 0.5, 0.0)
