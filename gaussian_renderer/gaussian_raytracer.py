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
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np

LOADED = False

class GaussianRaytracer:

    def __init__(self, pc: GaussianModel, example_camera):
        global LOADED
        if not LOADED:
            torch.classes.load_library(f"raytracer_builds/{pc.model_params.raytracer_version}/libgausstracer.so")
            LOADED = True

        self.cuda_raytracer = torch.classes.gausstracer.Raytracer(
            example_camera.image_width,
            example_camera.image_height,
            pc.get_scaling.shape[0]
        )

        self.pc = pc

        self._export_param_values()
        torch.cuda.synchronize() #!!! remove
        self.cuda_raytracer.rebuild_bvh()
        torch.cuda.synchronize() #!!! remove

        import sys
        sys.path.append(f"raytracer_builds/{pc.model_params.raytracer_version}")
        import raytracer_config
        self.config = raytracer_config

        self.cuda_raytracer.set_blur_kernel_bandwidth(pc.model_params.blur_kernel_bandwidth)
        torch.cuda.synchronize()

    @torch.no_grad()
    def rebuild_bvh(self):
        new_size = self.pc._xyz.shape[0]

        torch.cuda.synchronize() #!!! remove
        self.cuda_raytracer.resize(new_size)
        torch.cuda.synchronize() #!!! remove
        self._export_param_values()
        torch.cuda.synchronize() #!!! remove
        self.cuda_raytracer.rebuild_bvh()  
        torch.cuda.synchronize() #!!! remove

    @torch.no_grad()
    def _export_param_values(self):
        self.cuda_raytracer.gaussian_scales.copy_(self.pc._scaling)
        self.cuda_raytracer.gaussian_rotations.copy_(self.pc._rotation)
        self.cuda_raytracer.gaussian_means.copy_(self.pc._xyz)
        self.cuda_raytracer.gaussian_opacity.copy_(self.pc._opacity)
        if self.cuda_raytracer.gaussian_assigned_blur_level is not None:
            self.cuda_raytracer.gaussian_assigned_blur_level.copy_(self.pc._assigned_blur_level)
        self.cuda_raytracer.gaussian_rgb.copy_(self.pc._diffuse)
        if self.cuda_raytracer.gaussian_position is not None:
            self.cuda_raytracer.gaussian_position.copy_(self.pc._position)
        if self.cuda_raytracer.gaussian_normal is not None:
            self.cuda_raytracer.gaussian_normal.copy_(self.pc._normal)
        if self.cuda_raytracer.gaussian_roughness is not None:
            self.cuda_raytracer.gaussian_roughness.copy_(self.pc._roughness)
        if self.cuda_raytracer.gaussian_f0 is not None:
            self.cuda_raytracer.gaussian_f0.copy_(self.pc._f0)

    @torch.no_grad()
    def _import_param_gradients(self):
        self.pc._xyz.grad.add_(self.cuda_raytracer.gaussian_means.grad)
        self.pc._opacity.grad.add_(self.cuda_raytracer.gaussian_opacity.grad)
        if self.cuda_raytracer.gaussian_assigned_blur_level is not None:
            self.pc._assigned_blur_level.grad.add_(self.cuda_raytracer.gaussian_assigned_blur_level.grad)
        self.pc._scaling.grad.add_(self.cuda_raytracer.gaussian_scales.grad)
        self.pc._rotation.grad.add_(self.cuda_raytracer.gaussian_rotations.grad)
        self.pc._diffuse.grad.add_(self.cuda_raytracer.gaussian_rgb.grad)
        if self.cuda_raytracer.gaussian_position is not None:
            self.pc._position.grad.add_(self.cuda_raytracer.gaussian_position.grad)
        if self.cuda_raytracer.gaussian_normal is not None:
            self.pc._normal.grad.add_(self.cuda_raytracer.gaussian_normal.grad)
        if self.cuda_raytracer.gaussian_roughness is not None:
            self.pc._roughness.grad.add_(self.cuda_raytracer.gaussian_roughness.grad)
        if self.cuda_raytracer.gaussian_f0 is not None:
            self.pc._f0.grad.add_(self.cuda_raytracer.gaussian_f0.grad)

    def zero_grad(self):
        self.cuda_raytracer.gaussian_rgb.grad.zero_()
        self.cuda_raytracer.gaussian_opacity.grad.zero_()
        self.cuda_raytracer.gaussian_scales.grad.zero_()
        self.cuda_raytracer.gaussian_rotations.grad.zero_()
        self.cuda_raytracer.gaussian_means.grad.zero_()
        # 
        if self.cuda_raytracer.gaussian_assigned_blur_level is not None:
            self.cuda_raytracer.gaussian_assigned_blur_level.grad.zero_()
        if self.cuda_raytracer.gaussian_position is not None:
            self.cuda_raytracer.gaussian_position.grad.zero_()
        if self.cuda_raytracer.gaussian_normal is not None:
            self.cuda_raytracer.gaussian_normal.grad.zero_()
        if self.cuda_raytracer.gaussian_roughness is not None:
            self.cuda_raytracer.gaussian_roughness.grad.zero_()
        if self.cuda_raytracer.gaussian_f0 is not None:
            self.cuda_raytracer.gaussian_f0.grad.zero_()
        if self.cuda_raytracer.gaussian_specular is not None:
            self.cuda_raytracer.gaussian_specular.grad.zero_()
        if self.cuda_raytracer.gaussian_metalness is not None:
            self.cuda_raytracer.gaussian_metalness.grad.zero_()

    def __call__(self, viewpoint_camera,  pipe_params: PipelineParams, bg_color: torch.Tensor, target = None, target_diffuse = None, target_glossy = None, target_position=None, target_normal=None, target_roughness=None, target_f0=None, target_brdf=None):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """

        # *** time lost due to copies: 30s for 30000k iterations (~260k gaussians)

        with torch.no_grad():
            R = torch.from_numpy(viewpoint_camera.R).cuda().float() if isinstance(viewpoint_camera.R, np.ndarray) else viewpoint_camera.R.cuda()
            R_c2w_blender = -R 
            R_c2w_blender[:, 0] = -R_c2w_blender[:, 0] 

            self.cuda_raytracer.set_camera(R_c2w_blender.contiguous(), viewpoint_camera.camera_center.contiguous(), viewpoint_camera.FoVy)

            self._export_param_values()

            if target is not None:
                self.cuda_raytracer.target_rgb.copy_(target.moveaxis(0, -1))
            else:
                self.cuda_raytracer.target_rgb.zero_()

            if self.cuda_raytracer.target_diffuse is not None:
                if target_diffuse is not None:
                    self.cuda_raytracer.target_diffuse.copy_(target_diffuse.moveaxis(0, -1))
                else:
                    self.cuda_raytracer.target_diffuse.zero_()
            
            if self.cuda_raytracer.target_glossy is not None:
                if target_glossy is not None:
                    self.cuda_raytracer.target_glossy.copy_(target_glossy.moveaxis(0, -1))
                else:
                    self.cuda_raytracer.target_glossy.zero_()

            if self.cuda_raytracer.target_position is not None:
                if target_position is not None:
                    self.cuda_raytracer.target_position.copy_(target_position.moveaxis(0, -1)) 
                else:
                    self.cuda_raytracer.target_position.zero_()
            
            if self.cuda_raytracer.target_normal is not None:
                if target_normal is not None:
                    self.cuda_raytracer.target_normal.copy_(target_normal.moveaxis(0, -1)) 
                else:
                    self.cuda_raytracer.target_normal.zero_()

            if self.cuda_raytracer.target_roughness is not None:
                if target_roughness is not None:
                    self.cuda_raytracer.target_roughness.copy_(target_roughness.moveaxis(0, -1)) 
                else:
                    self.cuda_raytracer.target_roughness.zero_()

            if self.cuda_raytracer.target_f0 is not None:
                if target_f0 is not None:
                    self.cuda_raytracer.target_f0.copy_(target_f0.moveaxis(0, -1)) 
                else:
                    self.cuda_raytracer.target_f0.zero_()

            if self.cuda_raytracer.target_brdf is not None:
                if target_brdf is not None:
                    self.cuda_raytracer.target_brdf.copy_(target_brdf.moveaxis(0, -1)) 
                else:
                    self.cuda_raytracer.target_brdf.zero_()

        if "CHECK_NAN" in os.environ:
            if self.camera_c2w_rot_buffer.isnan().any() or self.camera_position_buffer.isnan().any() or self.vertical_fov_radians_buffer.isnan().any() or self.gaussian_scales_buffer.isnan().any() or self.gaussian_rotations_buffer.isnan().any() or self.gaussian_xyz_buffer.isnan().any() or self.gaussian_opacity_buffer.isnan().any() or self.gaussian_rgb_buffer.isnan().any():
                raise Exception("NaNs in input buffers!")
            if self.output_rgbt_buffer.isnan().any():
                raise Exception("NaNs in output buffers!")

        if torch.is_grad_enabled():
            self.cuda_raytracer.update_bvh()
        self.cuda_raytracer.raytrace()

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
            self._import_param_gradients()

        return {
            "render": self.cuda_raytracer.output_rgb.clone(),
        }
        
