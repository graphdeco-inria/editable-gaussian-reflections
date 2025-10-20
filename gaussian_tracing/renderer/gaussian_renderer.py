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

from types import SimpleNamespace

import torch

from gaussian_tracing.renderer.gaussian_raytracer import GaussianRaytracer
from gaussian_tracing.scene.cameras import Camera


# todo remove this
def render(
    camera: Camera,
    raytracer: GaussianRaytracer,
    targets_available=True,
    force_update_bvh=False,
    denoise=False,
    znear=0.01,
    zfar=999.9,
):
    do_backprop = torch.is_grad_enabled()

    if targets_available:
        target = camera.original_image
        target_diffuse = camera.diffuse_image
        target_glossy = camera.glossy_image
        target_normal = camera.normal_image
        target_f0 = camera.f0_image
        target_roughness = camera.roughness_image
        target_depth = camera.depth_image
    else:
        target = None
        target_diffuse = None
        target_glossy = None
        target_normal = None
        target_roughness = None
        target_f0 = None
        target_depth = None

    _target = target
    _target_diffuse = target_diffuse
    _target_glossy = target_glossy
    _target_normal = target_normal
    _target_roughness = target_roughness
    _target_f0 = target_f0
    _target_depth = target_depth

    with torch.set_grad_enabled(do_backprop):
        raytracer(
            camera,
            target=_target,
            target_diffuse=_target_diffuse,
            target_glossy=_target_glossy,
            target_depth=_target_depth,
            target_normal=_target_normal,
            target_roughness=_target_roughness,
            target_f0=_target_f0,
            force_update_bvh=force_update_bvh,
            denoise=denoise,
            znear=znear,
            zfar=zfar,
        )

    # All of these results are reshaped to (C, H, W)
    framebuffer = raytracer.cuda_module.get_framebuffer()
    rgb = framebuffer.output_rgb.clone().detach().moveaxis(-1, 1)
    # todo clean this up
    return SimpleNamespace(
        rgb=rgb,
        final=framebuffer.output_denoised.clone().detach().moveaxis(-1, 1) if denoise else framebuffer.output_final.clone().detach().moveaxis(-1, 1),
        depth=framebuffer.output_depth.clone().detach().moveaxis(-1, 1)
        if hasattr(framebuffer, "output_depth") and framebuffer.output_depth is not None
        else torch.zeros_like(rgb).mean(dim=1, keepdim=True),
        normal=framebuffer.output_normal.clone().detach().moveaxis(-1, 1) if framebuffer.output_normal is not None else torch.zeros_like(rgb),
        roughness=framebuffer.output_roughness.clone().detach().moveaxis(-1, 1) if framebuffer.output_roughness is not None else torch.zeros_like(rgb),
        f0=framebuffer.output_f0.clone().detach().moveaxis(-1, 1) if framebuffer.output_f0 is not None else torch.zeros_like(rgb),
        target=_target,
        target_diffuse=_target_diffuse,
        target_glossy=_target_glossy,
        target_depth=_target_depth,
        target_normal=_target_normal,
        target_roughness=_target_roughness,
        target_f0=_target_f0,
    )
