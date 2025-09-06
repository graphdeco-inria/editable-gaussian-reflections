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

import numpy as np
import torch

from gaussian_tracing import make_raytracer
from gaussian_tracing.scene.gaussian_model import GaussianModel

LOADED = False


class GaussianRaytracer:
    def __init__(self, pc: GaussianModel, image_width: int, image_height: int):
        self.image_width: int = image_width
        self.image_height: int = image_height
        self.cuda_module = make_raytracer(
            image_width, image_height, pc.get_scaling.shape[0]
        )

        os.environ["DIFFUSE_LOSS_WEIGHT"] = str(pc.model_params.loss_weight_diffuse)
        os.environ["GLOSSY_LOSS_WEIGHT"] = str(pc.model_params.loss_weight_glossy)
        os.environ["NORMAL_LOSS_WEIGHT"] = str(pc.model_params.loss_weight_normal)
        os.environ["POSITION_LOSS_WEIGHT"] = str(pc.model_params.loss_weight_depth)
        os.environ["F0_LOSS_WEIGHT"] = str(pc.model_params.loss_weight_f0)
        os.environ["ROUGHNESS_LOSS_WEIGHT"] = str(pc.model_params.loss_weight_roughness)

        self.cuda_module.set_losses(True)

        self.pc = pc

        self._export_param_values()
        torch.cuda.synchronize()  #!!! remove
        self.cuda_module.rebuild_bvh()
        torch.cuda.synchronize()  #!!! remove

    @torch.no_grad()
    def rebuild_bvh(self):
        new_size = self.pc._xyz.shape[0]

        torch.cuda.synchronize()  #!!! remove
        self.cuda_module.resize(new_size)
        torch.cuda.synchronize()  #!!! remove
        self._export_param_values()
        torch.cuda.synchronize()  #!!! remove
        self.cuda_module.rebuild_bvh()
        torch.cuda.synchronize()  #!!! remove

    @torch.no_grad()
    def _export_param_values(self):
        self.cuda_module.gaussian_scales.copy_(self.pc._get_scaling)
        self.cuda_module.gaussian_rotations.copy_(self.pc._get_rotation)
        self.cuda_module.gaussian_means.copy_(self.pc.get_xyz)
        self.cuda_module.gaussian_opacity.copy_(self.pc._opacity)
        self.cuda_module.gaussian_rgb.copy_(self.pc.get_diffuse)
        self.cuda_module.gaussian_normal.copy_(self.pc.get_normal)
        self.cuda_module.gaussian_roughness.copy_(self.pc.get_roughness)
        self.cuda_module.gaussian_f0.copy_(self.pc.get_f0)

    @torch.no_grad()
    def _import_param_gradients(self):
        self.pc._xyz.grad.add_(self.cuda_module.gaussian_means.grad)
        self.pc._opacity.grad.add_(self.cuda_module.gaussian_opacity.grad)
        self.pc._scaling.grad.add_(self.cuda_module.gaussian_scales.grad)
        self.pc._rotation.grad.add_(self.cuda_module.gaussian_rotations.grad)
        self.pc._diffuse.grad.add_(self.cuda_module.gaussian_rgb.grad)
        self.pc._normal.grad.add_(self.cuda_module.gaussian_normal.grad)
        self.pc._roughness.grad.add_(self.cuda_module.gaussian_roughness.grad)
        self.pc._f0.grad.add_(self.cuda_module.gaussian_f0.grad)

    def zero_grad(self):
        self.cuda_module.gaussian_rgb.grad.zero_()
        self.cuda_module.gaussian_opacity.grad.zero_()
        self.cuda_module.gaussian_scales.grad.zero_()
        self.cuda_module.gaussian_rotations.grad.zero_()
        self.cuda_module.gaussian_means.grad.zero_()
        self.cuda_module.gaussian_normal.grad.zero_()
        self.cuda_module.gaussian_roughness.grad.zero_()
        self.cuda_module.gaussian_f0.grad.zero_()

        self.cuda_module.densification_gradient_diffuse.zero_()
        self.cuda_module.densification_gradient_glossy.zero_()

    def __call__(
        self,
        viewpoint_camera,
        target=None,
        target_diffuse=None,
        target_glossy=None,
        target_depth=None,
        target_normal=None,
        target_roughness=None,
        target_f0=None,
        force_update_bvh=False,
        denoise=False,
    ):
        """
        Render the scene.
        """

        # *** time lost due to copies: 30s for 30000k iterations (~260k gaussians)
        if denoise:
            self.cuda_module.denoise.copy_(denoise)

        with torch.no_grad():
            R = (
                torch.from_numpy(viewpoint_camera.R).cuda().float()
                if isinstance(viewpoint_camera.R, np.ndarray)
                else viewpoint_camera.R.cuda()
            )
            R_c2w_blender = -R
            R_c2w_blender[:, 0] = -R_c2w_blender[:, 0]

            self.cuda_module.set_camera(
                R_c2w_blender.contiguous(),
                viewpoint_camera.camera_center.contiguous(),
                viewpoint_camera.FoVy,
                float(os.getenv("ZNEAR", 0.01)),
                float(os.getenv("ZFAR", 999.9)),
                self.pc.model_params.lod_max_world_size_blur,
            )

            self._export_param_values()

            if target is not None:
                self.cuda_module.target_rgb.copy_(target.moveaxis(0, -1))
            else:
                self.cuda_module.target_rgb.zero_()

            if target_diffuse is not None:
                self.cuda_module.target_diffuse.copy_(target_diffuse.moveaxis(0, -1))
            else:
                self.cuda_module.target_diffuse.zero_()

            if target_glossy is not None:
                self.cuda_module.target_glossy.copy_(target_glossy.moveaxis(0, -1))
            else:
                self.cuda_module.target_glossy.zero_()

            if target_depth is not None:
                self.cuda_module.target_depth.copy_(target_depth.unsqueeze(-1))
            else:
                self.cuda_module.target_depth.zero_()

            if target_normal is not None:
                self.cuda_module.target_normal.copy_(target_normal.moveaxis(0, -1))
            else:
                self.cuda_module.target_normal.zero_()

            if target_roughness is not None:
                self.cuda_module.target_roughness.copy_(
                    target_roughness.moveaxis(0, -1)
                )
            else:
                self.cuda_module.target_roughness.zero_()

            if target_f0 is not None:
                self.cuda_module.target_f0.copy_(target_f0.moveaxis(0, -1))
            else:
                self.cuda_module.target_f0.zero_()

        if torch.is_grad_enabled() or force_update_bvh:
            self.cuda_module.update_bvh()
        self.cuda_module.raytrace()

        if torch.is_grad_enabled():
            self._import_param_gradients()

        return {
            "render": self.cuda_module.output_rgb.clone(),
        }
