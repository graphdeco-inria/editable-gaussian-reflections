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
from os import makedirs

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tyro
from tqdm import tqdm

from gaussian_tracing.arguments import (
    TyroConfig,
)
from gaussian_tracing.renderer import GaussianRaytracer, render
from gaussian_tracing.scene import GaussianModel, Scene
from gaussian_tracing.utils.general_utils import safe_state
from gaussian_tracing.utils.image_utils import psnr
from gaussian_tracing.utils.tonemapping import tonemap


@torch.no_grad()
def render_set(
    cfg,
    scene,
    split,
    iteration,
    views,
    raytracer,
):
    model_path = cfg.model_path

    for mode in cfg.modes:
        for blur_sigma in cfg.blur_sigmas:
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

                config = raytracer.cuda_module.get_config()
                if cfg.max_bounces > -1:
                    config.num_bounces.copy_(cfg.max_bounces)

                if cfg.spp > 1:
                    config.accumulate_samples.copy_(True)
                    raytracer.cuda_module.reset_accumulators()
                    for i in range(cfg.spp):
                        package = render(
                            view,
                            raytracer,
                            denoise=cfg.denoise,
                        )
                else:
                    package = render(
                        view,
                        raytracer,
                        denoise=cfg.denoise,
                    )

                diffuse_gt_image = tonemap(view.diffuse_image).clamp(0.0, 1.0)
                glossy_gt_image = tonemap(view.glossy_image).clamp(0.0, 1.0)
                gt_image = tonemap(view.original_image).clamp(0.0, 1.0)
                normal_gt_image = view.normal_image
                roughness_gt_image = view.roughness_image
                depth_gt_image = view.depth_image.unsqueeze(0)
                F0_gt_image = view.F0_image

                diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
                glossy_image = tonemap(package.rgb[1:].sum(dim=0)).clamp(0, 1)
                pred_image = tonemap(package.final.squeeze(0)).clamp(0, 1)

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

                if not cfg.skip_save_frames and mode == "regular":
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
            os.makedirs(os.path.join(cfg.model_path, video_dir), exist_ok=True)

            if not cfg.skip_video:
                print("Writing videos...")
                path = os.path.join(cfg.model_path, f"{{dir}}{split}_{{name}}.mp4")

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


def main(cfg: TyroConfig):
    # Initialize system state (RNG)
    safe_state(cfg.quiet)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

    gaussians = GaussianModel(cfg)
    scene = Scene(cfg, gaussians, load_iteration=cfg.iteration, shuffle=False)

    if cfg.red_region:
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

    viewpoint_stack = scene.getTrainCameras().copy()
    raytracer = GaussianRaytracer(
        gaussians, viewpoint_stack[0].image_width, viewpoint_stack[0].image_height
    )

    if cfg.train_views:
        render_set(
            cfg,
            scene,
            "train",
            scene.loaded_iter,
            scene.getTrainCameras(),
            raytracer,
        )
    else:
        render_set(
            cfg,
            scene,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            raytracer,
        )


if __name__ == "__main__":
    cfg = tyro.cli(TyroConfig)
    main(cfg)
