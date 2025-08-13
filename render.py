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

import math
import os
import shutil
from argparse import ArgumentParser
from os import makedirs

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from gaussian_tracing.arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from gaussian_tracing.renderer import GaussianRaytracer, render
from gaussian_tracing.scene import GaussianModel, Scene
from gaussian_tracing.utils.general_utils import safe_state
from gaussian_tracing.utils.image_utils import psnr
from gaussian_tracing.utils.tonemapping import tonemap


@torch.no_grad()
def render_set(
    scene,
    model_params,
    model_path,
    split,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    raytracer,
):
    for mode in args.modes:
        for blur_sigma in args.blur_sigmas:
            render_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "render"
            )
            gts_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "render_gt"
            )
            diffuse_render_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "diffuse"
            )
            diffuse_gts_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "diffuse_gt"
            )
            glossy_render_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "glossy"
            )
            glossy_gts_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "glossy_gt"
            )
            position_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "position"
            )
            position_gts_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "position_gt"
            )
            depth_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "depth"
            )
            depth_gts_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "depth_gt"
            )
            normal_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "normal"
            )
            normal_gts_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "normal_gt"
            )
            roughness_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "roughness"
            )
            roughness_gts_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "roughness_gt"
            )
            F0_path = os.path.join(model_path, split, "ours_{}".format(iteration), "F0")
            F0_gts_path = os.path.join(
                model_path, split, "ours_{}".format(iteration), "F0_gt"
            )

            makedirs(render_path, exist_ok=True)
            makedirs(gts_path, exist_ok=True)
            makedirs(diffuse_render_path, exist_ok=True)
            makedirs(diffuse_gts_path, exist_ok=True)
            makedirs(glossy_render_path, exist_ok=True)
            makedirs(glossy_gts_path, exist_ok=True)
            makedirs(position_path, exist_ok=True)
            makedirs(position_gts_path, exist_ok=True)
            makedirs(depth_path, exist_ok=True)
            makedirs(depth_gts_path, exist_ok=True)
            makedirs(normal_path, exist_ok=True)
            makedirs(normal_gts_path, exist_ok=True)
            makedirs(roughness_path, exist_ok=True)
            makedirs(roughness_gts_path, exist_ok=True)
            makedirs(F0_path, exist_ok=True)
            makedirs(F0_gts_path, exist_ok=True)

            all_renders = []
            all_gts = []

            all_diffuse_renders = []
            all_diffuse_gts = []

            all_glossy_renders = []
            all_glossy_gts = []

            all_position_renders = []
            all_position_gts = []

            all_depth_renders = []
            all_depth_gts = []

            all_normal_renders = []
            all_normal_gts = []

            all_roughness_renders = []
            all_roughness_gts = []

            all_F0_renders = []
            all_F0_gts = []

            #

            l1_test = 0.0
            psnr_test = 0.0

            glossy_l1_test = 0.0
            glossy_psnr_test = 0.0

            diffuse_l1_test = 0.0
            diffuse_psnr_test = 0.0

            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                if mode == "lod":
                    view = views[0]

                if "env" in mode:
                    if idx == 0:
                        view0 = view
                        view0.FoVx = 2.0944 * 2  # ??? wrong value still works?
                        view0.FoVy = -2.0944 * 2  # ?? why negative
                        continue  # * Skip frame 0, rotation is incorrect for some reason I don't understand
                    view = view0

                    R_colmap_init = view.R
                    _R_blender = -R_colmap_init
                    _R_blender[:, 0] = -_R_blender[:, 0]
                    R_blender = _R_blender
                    T_blender = -R_colmap_init @ view.T

                    if "env_rot" in mode:
                        theta = (2 * math.pi * idx) / len(views)
                        rotation = torch.tensor(
                            (
                                (math.cos(theta), -math.sin(theta), 0.0),
                                (math.sin(theta), math.cos(theta), 0.0),
                                (0.0, 0.0, 1.0),
                            )
                        )
                        if idx > 0:
                            R_blender = rotation.to(torch.float64) @ np.array(
                                (
                                    (
                                        -0.9882196187973022,
                                        0.10767492651939392,
                                        -0.10875695198774338,
                                    ),
                                    (
                                        -0.10844696313142776,
                                        0.008747747167944908,
                                        0.9940638542175293,
                                    ),
                                    (
                                        0.10798710584640503,
                                        0.994147777557373,
                                        0.003032323671504855,
                                    ),
                                )
                            )
                    elif "env_move" in mode:
                        theta = 0
                        rotation = torch.tensor(
                            (
                                (math.cos(theta), -math.sin(theta), 0.0),
                                (math.sin(theta), math.cos(theta), 0.0),
                                (0.0, 0.0, 1.0),
                            )
                        )
                        R_blender = rotation.to(torch.float64) @ np.array(
                            (
                                (
                                    -0.9882196187973022,
                                    0.10767492651939392,
                                    -0.10875695198774338,
                                ),
                                (
                                    -0.10844696313142776,
                                    0.008747747167944908,
                                    0.9940638542175293,
                                ),
                                (
                                    0.10798710584640503,
                                    0.994147777557373,
                                    0.003032323671504855,
                                ),
                            )
                        )

                    if mode == "env_rot_1":
                        T_blender = np.array([0.0, -0.2, 0.2])
                    elif mode == "env_rot_2":
                        T_blender = np.array([1.3, -2.0, 0.0])
                    elif mode == "env_move_1":
                        t = idx / (len(views) - 1)
                        T_blender = (1.0 - t) * np.array(
                            [0.0, -0.2, 0.2]
                        ) + t * np.array([1.3, -2.0, 0.0])
                    elif mode == "env_move_2":
                        t = idx / (len(views) - 1)
                        T_blender = (1.0 - t) * np.array(
                            [0.0, -0.2, 0.2]
                        ) + t * np.array([1.3, -0.3, 0.0])

                    R_colmap = -R_blender
                    R_colmap[:, 0] = -R_colmap[:, 0]
                    T_colmap = -R_colmap.T @ T_blender

                    view.R = np.array(R_colmap)
                    view.T = np.array(T_colmap)

                    view.update()

                if mode == "lod":
                    alpha = idx / (len(views) - 1)
                    blur_sigma = alpha * scene.max_pixel_blur_sigma
                else:
                    blur_sigma = blur_sigma

                raytracer.cuda_module.denoise.copy_(not args.skip_denoiser)

                if args.max_bounces > -1:
                    raytracer.cuda_module.num_bounces.copy_(args.max_bounces)

                if args.spp > 1:
                    raytracer.cuda_module.accumulate.copy_(True)
                    raytracer.cuda_module.accumulated_rgb.zero_()
                    raytracer.cuda_module.accumulated_normal.zero_()
                    raytracer.cuda_module.accumulated_depth.zero_()
                    raytracer.cuda_module.accumulated_f0.zero_()
                    raytracer.cuda_module.accumulated_roughness.zero_()
                    raytracer.cuda_module.accumulated_sample_count.zero_()
                    for i in range(args.spp):
                        package = render(
                            view, raytracer, pipeline, background, blur_sigma=blur_sigma
                        )
                else:
                    package = render(
                        view, raytracer, pipeline, background, blur_sigma=blur_sigma
                    )

                if args.supersampling > 1:
                    for key, value in package.__dict__.items():
                        batched = value.ndim == 4
                        resized = torch.nn.functional.interpolate(
                            value[None] if not batched else value,
                            scale_factor=1.0 / args.supersampling,
                            mode="area",
                        )
                        setattr(package, key, resized[0] if not batched else resized)

                diffuse_gt_image = tonemap(view.diffuse_image).clamp(0.0, 1.0)
                glossy_gt_image = tonemap(view.glossy_image).clamp(0.0, 1.0)
                gt_image = tonemap(view.original_image).clamp(0.0, 1.0)
                position_gt_image = view.position_image
                normal_gt_image = view.normal_image
                roughness_gt_image = view.roughness_image
                depth_gt_image = view.depth_image.unsqueeze(0)
                F0_gt_image = view.F0_image

                if args.supersampling > 1:
                    diffuse_gt_image = torch.nn.functional.interpolate(
                        diffuse_gt_image[None],
                        scale_factor=1.0 / args.supersampling,
                        mode="area",
                    )[0]
                    glossy_gt_image = torch.nn.functional.interpolate(
                        glossy_gt_image[None],
                        scale_factor=1.0 / args.supersampling,
                        mode="area",
                    )[0]
                    gt_image = torch.nn.functional.interpolate(
                        gt_image[None],
                        scale_factor=1.0 / args.supersampling,
                        mode="area",
                    )[0]
                    position_gt_image = torch.nn.functional.interpolate(
                        position_gt_image[None],
                        scale_factor=1.0 / args.supersampling,
                        mode="area",
                    )[0]
                    depth_gt_image = torch.nn.functional.interpolate(
                        depth_gt_image[None],
                        scale_factor=1.0 / args.supersampling,
                        mode="area",
                    )[0]
                    normal_gt_image = torch.nn.functional.interpolate(
                        normal_gt_image[None],
                        scale_factor=1.0 / args.supersampling,
                        mode="area",
                    )[0]
                    roughness_gt_image = torch.nn.functional.interpolate(
                        roughness_gt_image[None],
                        scale_factor=1.0 / args.supersampling,
                        mode="area",
                    )[0]
                    F0_gt_image = torch.nn.functional.interpolate(
                        F0_gt_image[None],
                        scale_factor=1.0 / args.supersampling,
                        mode="area",
                    )[0]

                diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
                glossy_image = tonemap(package.rgb[1:-1].sum(dim=0)).clamp(0, 1)
                pred_image = tonemap(package.rgb[-1]).clamp(0, 1)

                psnr_test += psnr(pred_image, gt_image).mean() / len(views)
                l1_test += F.l1_loss(pred_image, gt_image) / len(views)

                glossy_psnr_test += psnr(glossy_image, glossy_gt_image).mean() / len(
                    views
                )
                glossy_l1_test += F.l1_loss(glossy_image, glossy_gt_image) / len(views)

                diffuse_psnr_test += psnr(diffuse_image, diffuse_gt_image).mean() / len(
                    views
                )
                diffuse_l1_test += F.l1_loss(diffuse_image, diffuse_gt_image) / len(
                    views
                )

                if not args.skip_save_frames and mode == "regular":
                    torchvision.utils.save_image(
                        glossy_image,
                        os.path.join(
                            glossy_render_path, "{0:05d}".format(idx) + "_glossy.png"
                        ),
                    )
                    torchvision.utils.save_image(
                        glossy_gt_image,
                        os.path.join(
                            glossy_gts_path, "{0:05d}".format(idx) + "_glossy.png"
                        ),
                    )

                    torchvision.utils.save_image(
                        diffuse_image,
                        os.path.join(
                            diffuse_render_path, "{0:05d}".format(idx) + "_diffuse.png"
                        ),
                    )
                    torchvision.utils.save_image(
                        diffuse_gt_image,
                        os.path.join(
                            diffuse_gts_path, "{0:05d}".format(idx) + "_diffuse.png"
                        ),
                    )

                    torchvision.utils.save_image(
                        package.position[0],
                        os.path.join(
                            position_path, "{0:05d}".format(idx) + "_position.png"
                        ),
                    )
                    torchvision.utils.save_image(
                        position_gt_image,
                        os.path.join(
                            position_gts_path, "{0:05d}".format(idx) + "_position.png"
                        ),
                    )

                    torchvision.utils.save_image(
                        package.depth[0].unsqueeze(0) / package.target_depth.amax(),
                        os.path.join(depth_path, "{0:05d}".format(idx) + "_depth.png"),
                    )
                    torchvision.utils.save_image(
                        depth_gt_image / package.target_depth.amax(),
                        os.path.join(
                            depth_gts_path, "{0:05d}".format(idx) + "_depth.png"
                        ),
                    )

                    torchvision.utils.save_image(
                        package.normal[0] / 2 + 0.5,
                        os.path.join(
                            normal_path, "{0:05d}".format(idx) + "_normal.png"
                        ),
                    )
                    torchvision.utils.save_image(
                        normal_gt_image / 2 + 0.5,
                        os.path.join(
                            normal_gts_path, "{0:05d}".format(idx) + "_normal.png"
                        ),
                    )

                    torchvision.utils.save_image(
                        package.roughness[0],
                        os.path.join(
                            roughness_path, "{0:05d}".format(idx) + "_roughness.png"
                        ),
                    )
                    torchvision.utils.save_image(
                        roughness_gt_image,
                        os.path.join(
                            roughness_gts_path, "{0:05d}".format(idx) + "_roughness.png"
                        ),
                    )

                    torchvision.utils.save_image(
                        package.F0[0],
                        os.path.join(F0_path, "{0:05d}".format(idx) + "_F0.png"),
                    )
                    torchvision.utils.save_image(
                        F0_gt_image,
                        os.path.join(F0_gts_path, "{0:05d}".format(idx) + "_F0.png"),
                    )

                    torchvision.utils.save_image(
                        gt_image,
                        os.path.join(gts_path, "{0:05d}".format(idx) + "_render.png"),
                    )
                    torchvision.utils.save_image(
                        pred_image,
                        os.path.join(
                            render_path, "{0:05d}".format(idx) + "_render.png"
                        ),
                    )

                def format_image(image):
                    image = F.interpolate(
                        image[None],
                        (image.shape[-2] // 2 * 2, image.shape[-1] // 2 * 2),
                        mode="bilinear",
                    )[0]
                    return (
                        (image.clamp(0, 1) * 255).to(torch.uint8).moveaxis(0, -1).cpu()
                    )

                all_renders.append(format_image(pred_image))
                all_gts.append(format_image(tonemap(package.target)))

                all_diffuse_renders.append(format_image(diffuse_image))
                all_diffuse_gts.append(format_image(tonemap(package.target_diffuse)))

                all_glossy_renders.append(format_image(glossy_image))
                all_glossy_gts.append(format_image(tonemap(package.target_glossy)))

                all_position_renders.append(format_image(package.position[0]))
                all_position_gts.append(format_image(package.target_position))

                max_depth = package.target_depth.amax()
                all_depth_renders.append(
                    format_image(package.depth[0] / max_depth).repeat(1, 1, 3)
                )
                all_depth_gts.append(
                    format_image(package.target_depth.unsqueeze(0) / max_depth).repeat(
                        1, 1, 3
                    )
                )

                all_normal_renders.append(format_image(package.normal[0] / 2 + 0.5))
                all_normal_gts.append(format_image(package.target_normal / 2 + 0.5))

                all_roughness_renders.append(format_image(package.roughness[0]))
                all_roughness_gts.append(
                    format_image(package.target_roughness.repeat(3, 1, 1))
                )

                all_F0_renders.append(format_image(package.F0[0]))
                all_F0_gts.append(format_image(package.target_f0))

            blur_suffix = f"_blur_{blur_sigma}" if blur_sigma is not None else ""
            video_dir = f"videos_{mode}{blur_suffix}/".replace("_normal", "")
            os.makedirs(os.path.join(model_params.model_path, video_dir), exist_ok=True)

            if not args.skip_video:
                print("Writing videos...")
                path = os.path.join(
                    model_params.model_path, f"{{dir}}{split}_{{name}}.mp4"
                )

                for label, quality in [("hq", "18"), ("lq", "30")]:
                    kwargs = dict(fps=30, options={"crf": quality})

                    torchvision.io.write_video(
                        path.format(name=f"renders_{label}", dir=video_dir),
                        torch.stack(all_renders),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"gts_{label}", dir=video_dir),
                        torch.stack(all_gts),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"comparison_{label}", dir=video_dir),
                        torch.cat(
                            [torch.stack(all_renders), torch.stack(all_gts)], dim=2
                        ),
                        **kwargs,
                    )

                    torchvision.io.write_video(
                        path.format(name=f"diffuse_renders_{label}", dir=video_dir),
                        torch.stack(all_diffuse_renders),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"diffuse_gts_{label}", dir=video_dir),
                        torch.stack(all_diffuse_gts),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"diffuse_comparison_{label}", dir=video_dir),
                        torch.cat(
                            [
                                torch.stack(all_diffuse_renders),
                                torch.stack(all_diffuse_gts),
                            ],
                            dim=2,
                        ),
                        **kwargs,
                    )

                    torchvision.io.write_video(
                        path.format(name=f"glossy_renders_{label}", dir=video_dir),
                        torch.stack(all_glossy_renders),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"glossy_gts_{label}", dir=video_dir),
                        torch.stack(all_glossy_gts),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"glossy_comparison_{label}", dir=video_dir),
                        torch.cat(
                            [
                                torch.stack(all_glossy_renders),
                                torch.stack(all_glossy_gts),
                            ],
                            dim=2,
                        ),
                        **kwargs,
                    )

                    torchvision.io.write_video(
                        path.format(name=f"position_renders_{label}", dir=video_dir),
                        torch.stack(all_position_renders),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"position_gts_{label}", dir=video_dir),
                        torch.stack(all_position_gts),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"position_comparison_{label}", dir=video_dir),
                        torch.cat(
                            [
                                torch.stack(all_position_renders),
                                torch.stack(all_position_gts),
                            ],
                            dim=2,
                        ),
                        **kwargs,
                    )

                    torchvision.io.write_video(
                        path.format(name=f"depth_renders_{label}", dir=video_dir),
                        torch.stack(all_depth_renders),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"depth_gts_{label}", dir=video_dir),
                        torch.stack(all_depth_gts),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"depth_comparison_{label}", dir=video_dir),
                        torch.cat(
                            [
                                torch.stack(all_depth_renders),
                                torch.stack(all_depth_gts),
                            ],
                            dim=2,
                        ),
                        **kwargs,
                    )

                    torchvision.io.write_video(
                        path.format(name=f"normal_renders_{label}", dir=video_dir),
                        torch.stack(all_normal_renders),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"normal_gts_{label}", dir=video_dir),
                        torch.stack(all_normal_gts),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"normal_comparison_{label}", dir=video_dir),
                        torch.cat(
                            [
                                torch.stack(all_normal_renders),
                                torch.stack(all_normal_gts),
                            ],
                            dim=2,
                        ),
                        **kwargs,
                    )

                    torchvision.io.write_video(
                        path.format(name=f"roughness_renders_{label}", dir=video_dir),
                        torch.stack(all_roughness_renders),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"roughness_gts_{label}", dir=video_dir),
                        torch.stack(all_roughness_gts),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(
                            name=f"roughness_comparison_{label}", dir=video_dir
                        ),
                        torch.cat(
                            [
                                torch.stack(all_roughness_renders),
                                torch.stack(all_roughness_gts),
                            ],
                            dim=2,
                        ),
                        **kwargs,
                    )

                    torchvision.io.write_video(
                        path.format(name=f"F0_renders_{label}", dir=video_dir),
                        torch.stack(all_F0_renders),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"F0_gts_{label}", dir=video_dir),
                        torch.stack(all_F0_gts),
                        **kwargs,
                    )
                    torchvision.io.write_video(
                        path.format(name=f"F0_comparison_{label}", dir=video_dir),
                        torch.cat(
                            [torch.stack(all_F0_renders), torch.stack(all_F0_gts)],
                            dim=2,
                        ),
                        **kwargs,
                    )

                if split == "test":
                    shutil.copy(
                        path.format(name="comparison_lq", dir=video_dir),
                        path.format(name=f"comparison_lq_{mode}{blur_suffix}", dir=""),
                    )
                    shutil.copy(
                        path.format(name="comparison_hq", dir=video_dir),
                        path.format(name=f"comparison_hq_{mode}{blur_suffix}", dir=""),
                    )

        if mode == "regular":
            print(f"PSNR: {psnr_test}")
            with open(os.path.join(model_path, split, "psnr.txt"), "w") as f:
                f.write(f"{psnr_test}\n")


@torch.no_grad()
def render_sets(model_params: ModelParams, iteration: int, pipeline: PipelineParams):
    gaussians = GaussianModel(model_params)

    scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)
    if "DBG_FLOATERS" in os.environ:
        mask = scene.select_points_to_prune_near_cameras(
            gaussians.get_xyz, gaussians.get_scaling
        )
        gaussians._opacity.data[mask] = -100000000.0

    if args.red_region:
        bbox_min = [0.22, -0.5, -0.22]
        bbox_max = [0.46, -0.13, -0.05]

        mask = (
            (gaussians.get_xyz < torch.tensor(bbox_max, device="cuda"))
            .all(dim=-1)
            .logical_and(
                (gaussians.get_xyz > torch.tensor(bbox_min, device="cuda")).all(dim=-1)
            )
        )
        gaussians._features_dc[mask] = torch.tensor([1.0, 0.0, 0.0], device="cuda")

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    raytracer = GaussianRaytracer(
        gaussians, viewpoint_stack[0].image_width, viewpoint_stack[0].image_height
    )

    if "MAKE_MIRROR" in os.environ:
        gaussians._roughness.zero_()

    if args.spp > 1:
        raytracer.cuda_module.denoise.fill_(False)

    if args.train_views:
        render_set(
            scene,
            model_params,
            model_params.model_path,
            "train",
            scene.loaded_iter,
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
            raytracer,
        )
    else:
        render_set(
            scene,
            model_params,
            model_params.model_path,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipeline,
            background,
            raytracer,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    _ = OptimizationParams(parser)
    pipeline = PipelineParams(parser)

    # Dummy repeat training args
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--flip_camera", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[
            1,
            100,
            500,
            1_000,
            2_500,
            5_000,
            10_000,
            20_000,
            30_000,
            60_000,
            90_000,
        ],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[1, 1_000, 2_500, 7_000, 15_000, 30_000, 60_000, 90_000],
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    # Rendering args
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--max_bounces", default=-1, type=int)
    parser.add_argument("--spp", default=128, type=int)
    parser.add_argument("--supersampling", default=1, type=int)
    parser.add_argument("--train_views", action="store_true")
    parser.add_argument("--skip_denoiser", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--modes",
        type=str,
        choices=[
            "regular",
            "lod",
            "env_rot_1",
            "env_rot_2",
            "env_move_1",
            "env_move_2",
        ],
        default=["regular", "env_rot_1", "env_move_1", "env_move_2"],
        nargs="+",
    )  # env_rot_1 is at the scene's origin, env_rot_2 it somewhere in the far-field, env_move_1 dollys forward, env_move_2 trucks sideways # ["regular", "lod", "env_rot_1", "env_move_1", "env_move_2"]
    parser.add_argument(
        "--blur_sigmas", type=float, default=[None], nargs="+"
    )  # [None, 4.0, 16.0]
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--red_region", action="store_true")
    parser.add_argument("--skip_save_frames", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # if not args.train_views:
    #     args.max_images = min(100, args.max_images)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    model_params = model.extract(args)
    model_params.resolution *= args.supersampling
    render_sets(model_params, args.iteration, pipeline.extract(args))
