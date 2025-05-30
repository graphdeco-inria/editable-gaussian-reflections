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

import kornia
import torch

from arguments import PipelineParams
from gaussian_tracing.renderer.gaussian_raytracer import GaussianRaytracer
from gaussian_tracing.scene.cameras import Camera

# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )
from gaussian_tracing.utils.tonemapping import tonemap, untonemap


def render(
    camera: Camera,
    raytracer: GaussianRaytracer,
    pipe_params: PipelineParams,
    bg_color: torch.Tensor,
    iteration=None,
    blur_sigma=None,
    edits=None,
    targets_available=True,
    force_update_bvh=False,
):
    do_backprop = torch.is_grad_enabled()

    if targets_available:
        target = camera.original_image
        target_diffuse = camera.diffuse_image
        target_glossy = camera.glossy_image
        target_position = camera.position_image
        target_normal = camera.normal_image
        target_f0 = camera.F0_image
        target_roughness = camera.roughness_image.mean(dim=0, keepdim=True)
        target_depth = camera.depth_image
        target_brdf = camera.brdf_image
    else:
        target = None
        target_diffuse = None
        target_glossy = None
        target_position = None
        target_normal = None
        target_roughness = None
        target_f0 = None
        target_brdf = None
        target_depth = None

    if blur_sigma is not None:
        if not isinstance(blur_sigma, torch.Tensor):
            blur_sigma = torch.tensor(blur_sigma)
        kernel_size = 2 * int(2 * blur_sigma) + 1
        target_stack = torch.cat(
            [
                target,
                target_diffuse,
                target_glossy,
                target_position,
                target_normal,
                target_roughness,
                target_f0,
                target_brdf,
            ]
        )
        if blur_sigma.item() >= 1e-3:
            target_stack = tonemap(
                kornia.filters.gaussian_blur2d(
                    untonemap(target_stack)[None],
                    (kernel_size, kernel_size),
                    (blur_sigma, blur_sigma),
                )[0]
            )
        (
            _target,
            _target_diffuse,
            _target_glossy,
            _target_position,
            _target_normal,
            _target_roughness,
            _target_f0,
            _target_brdf,
        ) = torch.split(target_stack, [3, 3, 3, 3, 3, 1, 3, 3])
        raytracer.cuda_module.init_blur_sigma.fill_(blur_sigma.item())
    else:
        _target = target
        _target_diffuse = target_diffuse
        _target_glossy = target_glossy
        _target_position = target_position
        _target_normal = target_normal
        _target_roughness = target_roughness
        _target_f0 = target_f0
        _target_brdf = target_brdf
        _target_depth = target_depth

    if raytracer.pc.model_params.use_diffuse_target:
        _target = _target_diffuse
    if raytracer.pc.model_params.use_glossy_target:
        _target = _target_diffuse

    with torch.set_grad_enabled(do_backprop):
        raytracer(
            camera,
            pipe_params,
            bg_color,
            blur_sigma,
            target=_target,
            target_diffuse=_target_diffuse,
            target_glossy=_target_glossy,
            target_position=_target_position,
            target_depth=_target_depth,
            target_normal=_target_normal,
            target_roughness=_target_roughness,
            target_f0=_target_f0,
            target_brdf=_target_brdf,
            edits=edits,
            force_update_bvh=force_update_bvh,
        )
        raytracer.cuda_module.init_blur_sigma.fill_(0.0)

    # All of these results are reshaped to (C, H, W)
    rgb = raytracer.cuda_module.output_rgb.clone().detach().moveaxis(-1, 1)
    return SimpleNamespace(
        rgb=rgb,
        position=raytracer.cuda_module.output_position.clone().detach().moveaxis(-1, 1)
        if raytracer.cuda_module.output_position is not None
        else torch.zeros_like(rgb),
        depth=raytracer.cuda_module.output_depth.clone().detach().moveaxis(-1, 1)
        if hasattr(raytracer.cuda_module, "output_depth")
        and raytracer.cuda_module.output_depth is not None
        else torch.zeros_like(rgb).mean(dim=1, keepdim=True),
        normal=raytracer.cuda_module.output_normal.clone().detach().moveaxis(-1, 1)
        if raytracer.cuda_module.output_normal is not None
        else torch.zeros_like(rgb),
        roughness=raytracer.cuda_module.output_roughness.clone()
        .detach()
        .moveaxis(-1, 1)
        .repeat(1, 3, 1, 1)
        if raytracer.cuda_module.output_roughness is not None
        else torch.zeros_like(rgb),
        F0=raytracer.cuda_module.output_f0.clone().detach().moveaxis(-1, 1)
        if raytracer.cuda_module.output_f0 is not None
        else torch.zeros_like(rgb),
        brdf=raytracer.cuda_module.output_brdf.clone().detach().moveaxis(-1, 1)
        if raytracer.cuda_module.output_brdf is not None
        else torch.zeros_like(rgb),
        target=_target,
        target_diffuse=_target_diffuse,
        target_glossy=_target_glossy,
        target_position=_target_position,
        target_depth=_target_depth,
        target_normal=_target_normal,
        target_roughness=_target_roughness,
        target_f0=_target_f0,
        target_brdf=_target_brdf,
    )
