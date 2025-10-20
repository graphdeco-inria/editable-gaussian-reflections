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
        self.cuda_module = make_raytracer(image_width, image_height, pc.get_scaling.shape[0])

        config = self.cuda_module.get_config()
        config.loss_weight_diffuse.fill_(pc.cfg.loss_weight_diffuse)
        config.loss_weight_glossy.fill_(pc.cfg.loss_weight_glossy)
        config.loss_weight_normal.fill_(pc.cfg.loss_weight_normal)
        config.loss_weight_depth.fill_(pc.cfg.loss_weight_depth)
        config.loss_weight_f0.fill_(pc.cfg.loss_weight_f0)
        config.loss_weight_roughness.fill_(pc.cfg.loss_weight_roughness)
        config.transmittance_threshold.fill_(pc.cfg.transmittance_threshold)
        config.alpha_threshold.fill_(pc.cfg.alpha_threshold)
        config.exp_power.fill_(pc.cfg.exp_power)

        self.pc = pc

        self._export_param_values()
        torch.cuda.synchronize()
        self.cuda_module.rebuild_bvh()
        torch.cuda.synchronize()

    @torch.no_grad()
    def rebuild_bvh(self):
        new_size = self.pc._xyz.shape[0]

        torch.cuda.synchronize()
        self.cuda_module.resize(new_size)
        torch.cuda.synchronize()
        self._export_param_values()
        torch.cuda.synchronize()
        self.cuda_module.rebuild_bvh()
        torch.cuda.synchronize()

    @torch.no_grad()
    def _export_param_values(self):
        gaussians = self.cuda_module.get_gaussians()
        gaussians.scale.copy_(self.pc._get_scaling)
        gaussians.rotation.copy_(self.pc._get_rotation)
        gaussians.mean.copy_(self.pc.get_xyz)
        gaussians.opacity.copy_(self.pc._opacity)
        gaussians.rgb.copy_(self.pc.get_diffuse)
        gaussians.normal.copy_(self.pc.get_normal)
        gaussians.roughness.copy_(self.pc.get_roughness)
        gaussians.f0.copy_(self.pc.get_f0)

    @torch.no_grad()
    def _import_param_gradients(self):
        gaussians = self.cuda_module.get_gaussians()
        self.pc._xyz.grad.add_(gaussians.mean.grad)
        self.pc._opacity.grad.add_(gaussians.opacity.grad)
        self.pc._scaling.grad.add_(gaussians.scale.grad)
        self.pc._rotation.grad.add_(gaussians.rotation.grad)
        self.pc._diffuse.grad.add_(gaussians.rgb.grad)
        self.pc._normal.grad.add_(gaussians.normal.grad)
        self.pc._roughness.grad.add_(gaussians.roughness.grad)
        self.pc._f0.grad.add_(gaussians.f0.grad)

    def zero_grad(self):
        gaussians = self.cuda_module.get_gaussians()
        gaussians.rgb.grad.zero_()
        gaussians.opacity.grad.zero_()
        gaussians.scale.grad.zero_()
        gaussians.rotation.grad.zero_()
        gaussians.mean.grad.zero_()
        gaussians.normal.grad.zero_()
        gaussians.roughness.grad.zero_()
        gaussians.f0.grad.zero_()

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
        znear=0.01,
        zfar=999.9,
    ):
        """
        Render the scene.
        """
        # *** time lost due to copies: 30s for 30000k iterations (~260k gaussians)
        with torch.no_grad():
            R = torch.from_numpy(viewpoint_camera.R).cuda().float() if isinstance(viewpoint_camera.R, np.ndarray) else viewpoint_camera.R.cuda()
            R_c2w_blender = -R
            R_c2w_blender[:, 0] = -R_c2w_blender[:, 0]

            camera = self.cuda_module.get_camera()
            camera.znear.fill_(float(os.getenv("ZNEAR", znear)))
            camera.zfar.fill_(float(os.getenv("ZFAR", zfar)))
            camera.vertical_fov_radians.fill_(torch.tensor(viewpoint_camera.FoVy, device="cuda"))
            camera.set_pose(viewpoint_camera.camera_center.contiguous(), R_c2w_blender.contiguous())

            self._export_param_values()

            framebuffer = self.cuda_module.get_framebuffer()

            if target_diffuse is not None:
                framebuffer.target_diffuse.copy_(target_diffuse.moveaxis(0, -1))
            else:
                framebuffer.target_diffuse.zero_()

            if target_glossy is not None:
                framebuffer.target_glossy.copy_(target_glossy.moveaxis(0, -1))
            else:
                framebuffer.target_glossy.zero_()

            if target_depth is not None:
                framebuffer.target_depth.copy_(target_depth.moveaxis(0, -1))
            else:
                framebuffer.target_depth.zero_()

            if target_normal is not None:
                framebuffer.target_normal.copy_(target_normal.moveaxis(0, -1))
            else:
                framebuffer.target_normal.zero_()

            if target_roughness is not None:
                framebuffer.target_roughness.copy_(target_roughness.moveaxis(0, -1))
            else:
                framebuffer.target_roughness.zero_()

            if target_f0 is not None:
                framebuffer.target_f0.copy_(target_f0.moveaxis(0, -1))
            else:
                framebuffer.target_f0.zero_()

        if torch.is_grad_enabled() or force_update_bvh:
            self.cuda_module.update_bvh()
        self.cuda_module.raytrace()

        if denoise:
            self.cuda_module.denoise()

        if torch.is_grad_enabled():
            self._import_param_gradients()

        return {
            "render": framebuffer.output_rgb.clone(),
        }
