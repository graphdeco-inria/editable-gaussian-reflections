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
import copy
import math
import os
import random
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace
from datetime import datetime
from random import randint
from threading import Thread

import pandas as pd
import plotly.express as px
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import GaussianRaytracer, render
from scene import GaussianModel, Scene
from scene.gaussian_model import build_scaling_rotation
from scene.tonemapping import *
from utils.general_utils import (
    build_rotation,
    colormap,
    get_expon_lr_func,
    inverse_sigmoid,
    safe_state,
)
from utils.graphics_utils import BasicPointCloud
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args: ModelParams, opt_params):
    if not args.model_path:
        args.model_path = os.path.join(
            "./output/", datetime.now().isoformat(timespec="seconds")
        )

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "model_params"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    with open(os.path.join(args.model_path, "opt_params"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt_params))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration):
    if iteration in args.test_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    sorted(scene.getTrainCameras(), key=lambda x: x.image_name)[
                        idx % len(scene.getTrainCameras())
                    ]
                    for idx in args.val_views
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                glossy_l1_test = 0.0
                glossy_psnr_test = 0.0

                diffuse_l1_test = 0.0
                diffuse_psnr_test = 0.0

                for idx, viewpoint in enumerate(config["cameras"]):
                    package = render(viewpoint, raytracer, pipe_params, bg)

                    os.makedirs(
                        tb_writer.log_dir + "/" + f"{config['name']}_view",
                        exist_ok=True,
                    )

                    if "SKIP_TONEMAPPING_OUTPUT" in os.environ:
                        diffuse_image = package.rgb[0].clamp(0, 1)
                        glossy_image = package.rgb[1:-1].sum(dim=0)
                        pred_image = package.rgb[-1].clamp(0, 1)
                        pred_image_without_denoising = package.rgb[:-1].sum(dim=0)
                        diffuse_gt_image = torch.clamp(
                            viewpoint.diffuse_image, 0.0, 1.0
                        )
                        glossy_gt_image = torch.clamp(viewpoint.glossy_image, 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    elif raytracer.config.TONEMAP:
                        diffuse_image = package.rgb[0].clamp(0, 1)
                        glossy_image = tonemap(
                            untonemap(package.rgb[1:-1]).sum(dim=0)
                        ).clamp(0, 1)
                        pred_image = package.rgb[-1].clamp(0, 1)
                        pred_image_without_denoising = tonemap(
                            untonemap(package.rgb[:-1]).sum(dim=0)
                        )
                        diffuse_gt_image = torch.clamp(
                            viewpoint.diffuse_image, 0.0, 1.0
                        )
                        glossy_gt_image = torch.clamp(viewpoint.glossy_image, 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    else:
                        diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
                        glossy_image = tonemap(package.rgb[1:-1].sum(dim=0)).clamp(0, 1)
                        pred_image = tonemap(package.rgb[-1]).clamp(0, 1)
                        pred_image_without_denoising = tonemap(
                            package.rgb[:-1].sum(dim=0)
                        )
                        diffuse_gt_image = tonemap(viewpoint.diffuse_image).clamp(0, 1)
                        glossy_gt_image = tonemap(viewpoint.glossy_image).clamp(0, 1)
                        gt_image = tonemap(viewpoint.original_image).clamp(0, 1)

                    if tb_writer and (idx < len(args.val_views)):
                        error_diffuse = diffuse_image - diffuse_gt_image
                        error_glossy = glossy_image - glossy_gt_image
                        error_final = pred_image - gt_image
                        save_image(
                            torch.stack(
                                [
                                    diffuse_image,
                                    diffuse_gt_image,
                                    glossy_image,
                                    glossy_gt_image,
                                    pred_image,
                                    gt_image,
                                ]
                            ).clamp(0, 1),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}.png",
                            nrow=2,
                            padding=0,
                        )
                        save_image(
                            torch.stack(
                                [
                                    diffuse_image,
                                    diffuse_gt_image,
                                    error_diffuse.abs() / error_diffuse.std() / 3,
                                    glossy_image,
                                    glossy_gt_image,
                                    error_glossy.abs() / error_glossy.std() / 3,
                                    gt_image,
                                    error_final.abs() / error_final.std() / 3,
                                ]
                            ).clamp(0, 1),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_error_maps.png",
                            nrow=3,
                            padding=0,
                        )

                    normal_gt_image = torch.clamp(
                        viewpoint.normal_image / 2 + 0.5, 0.0, 1.0
                    )
                    roughness_image = torch.clamp(package.roughness[0], 0.0, 1.0)
                    normal_image = torch.clamp(package.normal[0] / 2 + 0.5, 0.0, 1.0)
                    position_image = torch.clamp(package.position[0], 0.0, 1.0)
                    F0_image = torch.clamp(package.F0[0], 0.0, 1.0)
                    if model_params.brdf_mode != "disabled":
                        brdf_image = torch.clamp(package.brdf[0], 0.0, 1.0)

                    normal_gt_image = torch.clamp(
                        viewpoint.normal_image / 2 + 0.5, 0.0, 1.0
                    )
                    position_gt_image = torch.clamp(viewpoint.position_image, 0.0, 1.0)
                    F0_gt_image = torch.clamp(viewpoint.F0_image, 0.0, 1.0)
                    roughness_gt_image = torch.clamp(
                        viewpoint.roughness_image, 0.0, 1.0
                    )
                    if model_params.brdf_mode != "disabled":
                        brdf_gt_image = torch.clamp(viewpoint.brdf_image, 0.0, 1.0)

                    diffuse_l1_test += (
                        l1_loss(diffuse_image, diffuse_gt_image).mean().double()
                    )
                    diffuse_psnr_test += (
                        psnr(diffuse_image, diffuse_gt_image).mean().double()
                    )
                    glossy_l1_test += (
                        l1_loss(glossy_image, glossy_gt_image).mean().double()
                    )
                    glossy_psnr_test += (
                        psnr(glossy_image, glossy_gt_image).mean().double()
                    )
                    l1_test += l1_loss(pred_image, gt_image).mean().double()
                    psnr_test += psnr(pred_image, gt_image).mean().double()

                    if tb_writer and (idx < len(args.val_views)):
                        if package.rgb.shape[0] > 2:
                            save_image(
                                package.rgb[:-1].clamp(0, 1),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_rgb_all_rays.png",
                                padding=0,
                            )
                            save_image(
                                torch.clamp(package.normal / 2 + 0.5, 0.0, 1.0),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_normal_all_rays.png",
                                padding=0,
                            )
                            save_image(
                                torch.clamp(package.F0, 0.0, 1.0),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_F0_all_rays.png",
                                padding=0,
                            )
                            save_image(
                                torch.clamp(package.position, 0.0, 1.0),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_pos_all_rays.png",
                                padding=0,
                            )
                            save_image(
                                torch.clamp(package.brdf, 0.0, 1.0),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_brdf_all_rays.png",
                                padding=0,
                            )
                            if (
                                raytracer.cuda_module.output_incident_radiance
                                is not None
                            ):
                                save_image(
                                    raytracer.cuda_module.output_incident_radiance.moveaxis(
                                        -1, 1
                                    ).clamp(0, 1),
                                    tb_writer.log_dir
                                    + "/"
                                    + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_incident_radiance_all_rays.png",
                                    padding=0,
                                )
                        if raytracer.config.SAVE_RAY_IMAGES:
                            save_image(
                                raytracer.cuda_module.output_ray_origin[0].moveaxis(
                                    -1, 0
                                ),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_ray_origin.png",
                                padding=0,
                            )
                            save_image(
                                raytracer.cuda_module.output_ray_direction[0].moveaxis(
                                    -1, 0
                                ),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_ray_direction.png",
                                padding=0,
                            )

                        if raytracer.config.SAVE_LOD_IMAGES:
                            save_image(
                                raytracer.cuda_module.output_lod_mean,
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_lod_mean_all_rays.png",
                                padding=0,
                            )
                            save_image(
                                raytracer.cuda_module.output_lod_scale,
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_lod_scale_all_rays.png",
                                padding=0,
                            )
                            save_image(
                                raytracer.cuda_module.output_ray_lod,
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_ray_lod_all_rays.png",
                                padding=0,
                            )

                        if raytracer.config.SAVE_HIT_STATS:
                            torch.save(
                                raytracer.cuda_module.num_hits_per_pixel,
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_num_hits_per_pixel.pt",
                            )
                            torch.save(
                                raytracer.cuda_module.num_traversed_per_pixel,
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_num_traversed_per_pixel.pt",
                            )
                            # also save them as normalized png
                            save_image(
                                (
                                    raytracer.cuda_module.num_hits_per_pixel.float()
                                    / raytracer.cuda_module.num_hits_per_pixel.max()
                                ),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_num_hits_per_pixel.png",
                                padding=0,
                            )
                            save_image(
                                (
                                    raytracer.cuda_module.num_traversed_per_pixel.float()
                                    / raytracer.cuda_module.num_traversed_per_pixel.max()
                                ),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_num_traversed_per_pixel.png",
                                padding=0,
                            )

                        save_image(
                            torch.stack(
                                [roughness_image.cuda(), roughness_gt_image]
                            ).clamp(0, 1),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_roughness_vs_target.png",
                            nrow=2,
                            padding=0,
                        )
                        save_image(
                            torch.stack([F0_image.cuda(), F0_gt_image]).clamp(0, 1),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_F0_vs_target.png",
                            nrow=2,
                            padding=0,
                        )
                        save_image(
                            torch.stack([pred_image, gt_image]).clamp(0, 1),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_final_denoised_vs_target.png",
                            nrow=2,
                            padding=0,
                        )
                        save_image(
                            torch.stack([pred_image_without_denoising, gt_image]).clamp(
                                0, 1
                            ),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_final_without_denoising_vs_target.png",
                            nrow=2,
                            padding=0,
                        )
                        save_image(
                            torch.stack([diffuse_image, diffuse_gt_image]).clamp(0, 1),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_diffuse_vs_target.png",
                            nrow=2,
                            padding=0,
                        )
                        save_image(
                            torch.stack([glossy_image, glossy_gt_image]).clamp(0, 1),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_glossy_vs_target.png",
                            nrow=2,
                            padding=0,
                        )
                        save_image(
                            torch.stack(
                                [position_image.cuda(), position_gt_image]
                            ).clamp(0, 1),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_position_vs_target.png",
                            nrow=2,
                            padding=0,
                        )
                        save_image(
                            torch.stack([normal_image.cuda(), normal_gt_image]).clamp(
                                0, 1
                            ),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_normal_vs_target.png",
                            nrow=2,
                            padding=0,
                        )
                        if model_params.brdf_mode != "disabled":
                            save_image(
                                torch.stack([brdf_image, brdf_gt_image.cuda()]).clamp(
                                    0, 1
                                ),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_brdf_vs_target.png",
                                nrow=2,
                                padding=0,
                            )

                        if raytracer.cuda_module.output_incident_radiance is not None:
                            save_image(
                                raytracer.cuda_module.output_incident_radiance[1]
                                .clamp(0, 1)
                                .moveaxis(-1, 0),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_incident_radiance_vs_target.png",
                                padding=0,
                            )

                        if raytracer.config.USE_LEVEL_OF_DETAIL:
                            for k, alpha in enumerate(torch.linspace(0.0, 1.0, 4)):
                                package = render(
                                    viewpoint,
                                    raytracer,
                                    pipe_params,
                                    bg,
                                    blur_sigma=alpha * scene.max_pixel_blur_sigma
                                    if not model_params.lod_force_blur_sigma >= 0.0
                                    else torch.tensor(
                                        model_params.lod_force_blur_sigma, device="cuda"
                                    ),
                                )

                                diffuse_gt_image = package.target_diffuse
                                glossy_gt_image = package.target_glossy
                                gt_image = package.target

                                if raytracer.config.TONEMAP:
                                    diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
                                    glossy_image = tonemap(
                                        package.rgb[1:-1].sum(dim=0)
                                    ).clamp(0, 1)
                                    pred_image = tonemap(package.rgb[-1]).clamp(0, 1)
                                else:
                                    diffuse_image = package.rgb[0].clamp(0, 1)
                                    glossy_image = (
                                        package.rgb[1:-1].sum(dim=0).clamp(0, 1)
                                    )
                                    pred_image = package.rgb[-1].clamp(0, 1)

                                save_image(
                                    torch.stack(
                                        [
                                            diffuse_pred,
                                            diffuse_gt_image,
                                            glossy_pred,
                                            glossy_gt_image,
                                            pred,
                                            gt_image,
                                        ]
                                    ).clamp(0, 1),
                                    tb_writer.log_dir
                                    + "/"
                                    + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_blurred_{k}÷3.png",
                                    nrow=2,
                                    padding=0,
                                )

                if model_params.brdf_mode == "static_lut":
                    save_image(
                        torch.stack([gaussians.get_brdf_lut]).abs(),
                        os.path.join(
                            tb_writer.log_dir,
                            f"{config['name']}_view/lut_iter_{iteration:09}.png",
                        ),
                        nrow=1,
                        padding=0,
                    )
                elif model_params.brdf_mode == "finetuned_lut":
                    save_image(
                        torch.stack([gaussians._brdf_lut, gaussians.get_brdf_lut]),
                        os.path.join(
                            tb_writer.log_dir,
                            f"{config['name']}_view/lut_iter_{iteration:09}.png",
                        ),
                        nrow=1,
                        padding=0,
                    )
                    save_image(
                        torch.stack(
                            [
                                gaussians._brdf_lut_residual,
                                gaussians._brdf_lut_residual * 5,
                                gaussians._brdf_lut_residual * 10,
                                gaussians._brdf_lut_residual * 20,
                                gaussians._brdf_lut_residual * 50,
                                gaussians._brdf_lut_residual * 200,
                                gaussians._brdf_lut_residual * 10000,
                            ]
                        ).abs(),
                        os.path.join(
                            tb_writer.log_dir,
                            f"{config['name']}_view/lut_residual_amplified_iter_{iteration:09}.png",
                        ),
                        nrow=1,
                        padding=0,
                    )

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])

                diffuse_psnr_test /= len(config["cameras"])
                diffuse_l1_test /= len(config["cameras"])

                glossy_psnr_test /= len(config["cameras"])
                glossy_l1_test /= len(config["cameras"])

                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - glossy_l1_loss",
                        glossy_l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - glossy_psnr",
                        glossy_psnr_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - diffuse_l1_loss",
                        diffuse_l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - diffuse_psnr",
                        diffuse_psnr_test,
                        iteration,
                    )

                with open(
                    os.path.join(tb_writer.log_dir, f"losses_{config['name']}.csv"), "a"
                ) as f:
                    f.write(
                        f"{iteration:05d}, {diffuse_psnr_test:02.2f}, {glossy_psnr_test:02.2f}, {psnr_test:02.2f}\n"
                    )

        # if tb_writer:
        #     tb_writer.add_histogram(
        #         "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
        #     )
        #     tb_writer.add_scalar(
        #         "total_points", scene.gaussians.get_xyz.shape[0], iteration
        #     )

        torch.cuda.empty_cache()


# Set up command line argument parser
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--viewer", action="store_true")
parser.add_argument("--viewer_mode", type=str, default="local")
parser.add_argument("--detect_anomaly", action="store_true", default=False)
parser.add_argument("--flip_camera", action="store_true", default=False)
parser.add_argument(
    "--val_views", nargs="+", type=int, default=[75]
)  # 135 for teaser book sceen
parser.add_argument(
    "--test_iterations",
    nargs="+",
    type=int,
    default=[4, 6_000, 12_000, 18_000, 24_000],
)
parser.add_argument(
    "--save_iterations", nargs="+", type=int, default=[4, 6_000, 12_000, 18_000, 24_000]
)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default=None)
args = parser.parse_args(sys.argv[1:])
args.save_iterations.append(args.iterations)

if "_with_book" in args.source_path:
    args.max_images = 199  ##! crashes if last image is included

torch.autograd.set_detect_anomaly(args.detect_anomaly)
model_params = lp.extract(args)
opt_params = op.extract(args)
pipe_params = pp.extract(args)

if args.viewer:
    args.test_iterations.clear()

if opt_params.timestretch != 1:
    model_params.no_bounces_until_iter = int(
        model_params.no_bounces_until_iter * opt_params.timestretch
    )
    model_params.max_one_bounce_until_iter = int(
        model_params.max_one_bounce_until_iter * opt_params.timestretch
    )
    model_params.rebalance_losses_at_iter = int(
        model_params.rebalance_losses_at_iter * opt_params.timestretch
    )
    args.test_iterations = [
        int(x * opt_params.timestretch) for x in args.test_iterations
    ]
    args.save_iterations = [
        int(x * opt_params.timestretch) for x in args.save_iterations
    ]
    opt_params.iterations = int(opt_params.timestretch * opt_params.iterations)
    opt_params.densification_interval = int(
        opt_params.timestretch * opt_params.densification_interval
    )
    opt_params.position_lr_max_steps = int(
        opt_params.timestretch * opt_params.position_lr_max_steps
    )
    opt_params.densify_from_iter = int(
        opt_params.timestretch * opt_params.densify_from_iter
    )
    opt_params.densify_until_iter = int(
        opt_params.timestretch * opt_params.densify_until_iter
    )

print("Optimizing " + args.model_path)

# Initialize system state (RNG)
safe_state(args.quiet)

tb_writer = prepare_output_and_logger(model_params, opt_params)

gaussians = GaussianModel(model_params)

scene = Scene(model_params, gaussians)

gaussians.training_setup(opt_params)

first_iter = 0
if args.start_checkpoint:
    (capture_data, first_iter) = torch.load(
        args.start_checkpoint, weights_only=False
    )  #!!! Cant release like this, security risk
    gaussians.restore(capture_data, opt_params)
first_iter += 1

bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

iter_start = torch.cuda.Event(enable_timing=True)
iter_end = torch.cuda.Event(enable_timing=True)

viewpoint_stack = scene.getTrainCameras().copy()
raytracer = GaussianRaytracer(
    gaussians, viewpoint_stack[0].image_width, viewpoint_stack[0].image_height
)
# raytracer.cuda_module.num_samples.fill_(model_params.num_samples)


if args.viewer:
    from gaussianviewer import GaussianViewer
    from viewer.types import ViewerMode

    mode = ViewerMode.LOCAL if args.viewer_mode == "local" else ViewerMode.SERVER
    SPARSE_ADAM_AVAILABLE = False
    viewer = GaussianViewer.from_gaussians(
        raytracer, model_params, opt_params, gaussians, SPARSE_ADAM_AVAILABLE, mode
    )
    viewer.accumulate_samples = False
    if args.viewer_mode != "none":
        viewer_thd = Thread(target=viewer.run, daemon=True)
        viewer_thd.start()

ema_loss_for_log = 0.0
start = time.time()

if model_params.no_bounces_until_iter > 0:
    raytracer.cuda_module.num_bounces.copy_(0)
elif model_params.max_one_bounce_until_iter > 0:
    raytracer.cuda_module.num_bounces.copy_(min(raytracer.config.MAX_BOUNCES, 1))

if model_params.warmup_until_iter > 0:
    os.environ["DIFFUSE_LOSS_WEIGHT"] = str(model_params.warmup_diffuse_loss_weight)
    raytracer.cuda_module.set_losses(True)

for iteration in tqdm(
    range(first_iter, opt_params.iterations + 1),
    desc="Training progress",
    total=opt_params.iterations,
    initial=first_iter,
):
    iter_start.record()

    if args.viewer:
        viewer.gaussian_lock.acquire()

    xyz_lr = gaussians.update_learning_rate(iteration)
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

    bg = torch.rand((3), device="cuda") if opt_params.random_background else background

    # *** run fused forward + backprop
    if (
        raytracer.config.USE_LEVEL_OF_DETAIL
        and random.random() < model_params.lod_prob_blur_targets
    ):
        if model_params.lod_force_blur_sigma >= 0.0:
            blur_sigma = torch.tensor(model_params.lod_force_blur_sigma, device="cuda")
        else:
            blur_sigma = (
                torch.rand(1, device="cuda") ** model_params.lod_schedule_power
                * scene.max_pixel_blur_sigma
            )
    else:
        blur_sigma = None

    torch.cuda.synchronize()  # todo may be needed or not, idk, occasional crash. double check after deadline
    package = render(viewpoint_cam, raytracer, pipe_params, bg, blur_sigma=blur_sigma)

    if opt_params.opacity_reg > 0:
        gaussians._opacity.grad += torch.autograd.grad(
            args.opacity_reg * torch.abs(gaussians.get_opacity).mean(),
            gaussians._opacity,
        )[0]
    if opt_params.scale_reg > 0:
        gaussians._scaling.grad += torch.autograd.grad(
            args.scale_reg * torch.abs(gaussians.get_scaling).mean(), gaussians._scaling
        )[0]

    with torch.no_grad():
        if opt_params.opacity_decay < 1.0:
            gaussians._opacity.copy_(
                inverse_sigmoid(gaussians.get_opacity * opt_params.opacity_decay)
            )
        if opt_params.scale_decay < 1.0:
            gaussians._scaling.copy_(
                torch.log(gaussians.get_scaling * opt_params.scale_decay)
            )
        if opt_params.lod_mean_decay < 1.0:
            gaussians._lod_mean.copy_(
                torch.log(gaussians.get_lod_mean * opt_params.lod_mean_decay)
            )
        if opt_params.lod_scale_decay < 1.0:
            gaussians._lod_scale.copy_(
                torch.log(gaussians.get_lod_scale * opt_params.lod_scale_decay)
            )

    # todo clamp the min opacities so they don't go under ALPHA_THRESHOLD
    iter_end.record()

    with torch.no_grad():
        # Log and save
        training_report(tb_writer, iteration)
        torch.cuda.synchronize()  # not sure if needed

        if iteration % 1000 == 0 or iteration == 1:
            os.makedirs(os.path.join(args.model_path, "plots"), exist_ok=True)

            if False:
                # Save a histogram of gaussian opacities
                opacities = gaussians.get_opacity.cpu().numpy()
                df = pd.DataFrame(opacities, columns=["opacity"])
                fig = px.histogram(
                    df, x="opacity", nbins=50, title="Histogram of Gaussian Opacities"
                )
                fig.write_image(
                    os.path.join(
                        args.model_path,
                        f"plots/opacity_histogram_{iteration:05d}.png",
                        padding=0,
                    )
                )

                # Save a histogram of gaussian _lod_mean
                lod_mean = gaussians.get_lod_mean.cpu().numpy()
                df = pd.DataFrame(lod_mean, columns=["lod_mean"])
                fig = px.histogram(
                    df, x="lod_mean", nbins=50, title="Histogram of Gaussian LOD Mean"
                )
                fig.write_image(
                    os.path.join(
                        args.model_path,
                        f"plots/lod_mean_histogram_{iteration:05d}.png",
                        padding=0,
                    )
                )

                # Save a histogram of gaussian _lod_scale
                lod_scale = gaussians.get_lod_scale.cpu().numpy()
                df = pd.DataFrame(lod_scale, columns=["lod_scale"])
                fig = px.histogram(
                    df, x="lod_scale", nbins=50, title="Histogram of Gaussian LOD Scale"
                )
                fig.write_image(
                    os.path.join(
                        args.model_path,
                        f"plots/lod_scale_histogram_{iteration:05d}.png",
                        padding=0,
                    )
                )

            if False:
                # Save a scatter plot of gaussian round counter vs lod_mean
                sample_indices = random.sample(
                    range(gaussians._round_counter.shape[0]),
                    int(0.20 * gaussians._round_counter.shape[0]),
                )
                round_counter = gaussians._round_counter[sample_indices].cpu().numpy()
                lod_mean = gaussians.get_lod_mean[sample_indices].cpu().numpy()
                df = pd.DataFrame(
                    {"round_counter": round_counter[:, 0], "lod_mean": lod_mean[:, 0]}
                )
                fig = px.scatter(
                    df,
                    x="lod_mean",
                    y="round_counter",
                    title="Scatter Plot of Gaussian Densification Round Counter vs LOD Mean",
                    opacity=0.01,
                )
                fig.write_image(
                    os.path.join(
                        args.model_path,
                        f"plots/round_counter_vs_lod_mean_{iteration:05d}.png",
                        padding=0,
                    )
                )

            # Save the elapsed time
            delta = time.time() - start
            with open(os.path.join(args.model_path, "time.txt"), "a") as f:
                minutes, seconds = divmod(int(delta), 60)
                timestamp = f"{minutes:02}:{seconds:02}"
                print("Elapsed time: ", timestamp)
                f.write(f"{iteration:5}: {timestamp}\n")

            # Save the average and std opacity
            with open(os.path.join(args.model_path, "opacity.txt"), "a") as f:
                f.write(
                    f"{iteration:5}: {gaussians.get_opacity.mean().item():.3f} +- {gaussians.get_opacity.std().item():.3f}\n"
                )

            # Save the average and std size
            with open(os.path.join(args.model_path, "size.txt"), "a") as f:
                f.write(
                    f"{iteration:5}: {gaussians.get_scaling.mean().item():.3f} +- {gaussians.get_scaling.std().item():.3f}\n"
                )

            # Save the average and std size of the largest axis per gaussian
            with open(os.path.join(args.model_path, "size_axis_max.txt"), "a") as f:
                f.write(
                    f"{iteration:5}: {gaussians.get_scaling.amax(dim=1).mean().item():.5f} +- {gaussians.get_scaling.amax(dim=1).std().item():.5f}\n"
                )

            # same but for the median axis
            with open(os.path.join(args.model_path, "size_axis_median.txt"), "a") as f:
                f.write(
                    f"{iteration:5}: {gaussians.get_scaling.median(dim=1).values.mean().item():.5f} +- {gaussians.get_scaling.median(dim=1).values.std().item():.5f}\n"
                )

            # Same but for the smallest axis
            with open(os.path.join(args.model_path, "size_axis_min.txt"), "a") as f:
                f.write(
                    f"{iteration:5}: {gaussians.get_scaling.amin(dim=1).mean().item():.5f} +- {gaussians.get_scaling.amin(dim=1).std().item():.5f}\n"
                )

            # From raytracer.num_hits, print the mean, max, and std
            num_hits = raytracer.cuda_module.num_hits_per_pixel.float()
            with open(os.path.join(args.model_path, "num_hits.txt"), "a") as f:
                f.write(
                    f"{iteration:5}: {num_hits.mean().item():.3f} +- {num_hits.std().item():.3f}\n"
                )

            num_traversed = raytracer.cuda_module.num_traversed_per_pixel.float()
            with open(os.path.join(args.model_path, "num_traversed.txt"), "a") as f:
                f.write(
                    f"{iteration:5}: {num_traversed.mean().item():.3f} +- {num_traversed.std().item():.3f}\n"
                )

            with open(os.path.join(args.model_path, "num_gaussians.txt"), "a") as f:
                f.write(f"{iteration:5}: {gaussians.get_xyz.shape[0]}\n")
                print("Number of gaussians: ", gaussians.get_xyz.shape[0])

        if iteration in args.save_iterations:
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        if opt_params.densif_use_top_k and iteration <= opt_params.densify_until_iter:
            gaussians.add_densification_stats_3d(
                raytracer.cuda_module.densification_gradient_diffuse,
                raytracer.cuda_module.densification_gradient_glossy,
            )

        if "DENSIFY" in os.environ:
            # max_ws_size = (
            #     scene.cameras_extent
            #     * model_params.glossy_bbox_size_mult
            #     * model_params.scene_extent_multiplier
            # )
            max_ws_size = 9999999.99
            densif_args = (scene, opt_params, model_params.min_opacity, max_ws_size)
            if (
                iteration % opt_params.densification_interval == 0
                and iteration > opt_params.densify_from_iter
            ):
                if iteration < opt_params.densify_until_iter:
                    trace = gaussians.densify_and_prune_top_k(*densif_args)
                    trace = f"Iteration {iteration}; " + trace
                    with open(
                        os.path.join(scene.model_path + "/densification_trace.txt"), "a"
                    ) as f:
                        f.write(trace)
                else:
                    gaussians.prune_znear_only(scene)
                raytracer.rebuild_bvh()
            # elif iteration % opt_params.densification_interval == 0:
            #     if opt_params.prune_even_without_densification:
            #         gaussians.prune(*densif_args)
            #     if "NO_REBUILD" not in os.environ:
            #         raytracer.rebuild_bvh()
        else:
            if iteration % opt_params.densification_interval == 0:
                if model_params.min_weight > 0:
                    gaussians.prune_points(
                        (
                            raytracer.cuda_module.gaussian_total_weight
                            < model_params.min_weight
                        ).squeeze(1)
                    )
                if model_params.znear_densif_pruning:
                    gaussians.prune_znear_only(scene)
                raytracer.cuda_module.gaussian_total_weight.zero_()
                torch.cuda.synchronize()
                raytracer.rebuild_bvh()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=False)
        raytracer.zero_grad()

        with torch.no_grad():
            gaussians._diffuse.data.clamp_(min=0.0)
            gaussians._roughness.data.clamp_(min=0.0, max=1.0)
            gaussians._f0.data.clamp_(min=0.0, max=1.0)

        if raytracer.config.USE_LEVEL_OF_DETAIL:
            with torch.no_grad():
                gaussians._lod_mean.data.clamp_(min=0)

            if model_params.lod_clamp_minsize:
                with torch.no_grad():
                    gaussians._scaling.data.clamp_(
                        min=torch.log(
                            gaussians._lod_mean.clamp(
                                min=float(os.getenv("LOD_CLAMP_EPS", 0.0))  # was 1e-8
                            )
                        )
                    )
                if (
                    torch.isnan(gaussians._lod_mean).any()
                    or torch.isnan(gaussians._scaling).any()
                ):
                    print("NANs in lod_mean or _scaling")
                    quit()

        # Might help for roughness or for tint spehres, doesn't help in the regular case
        if "SKIP_CLAMP_RELMINSIZE" not in os.environ:
            with torch.no_grad():
                farfield_mask = ~gaussians.comes_from_colmap_pc.bool()
                max_scaling = gaussians.get_scaling.amax(dim=-1)
                clamped_size_farfield = gaussians._scaling.data.clamp(
                    min=torch.log(
                        max_scaling * float(os.getenv("RELMINSIZE_FARFIELD", 0.20))
                    ).unsqueeze(-1)
                )
                clamped_size_nearfield = gaussians._scaling.data.clamp(
                    min=torch.log(
                        max_scaling * float(os.getenv("RELMINSIZE_NEARFIELD", 0.05))
                    ).unsqueeze(-1)
                )
                gaussians._scaling.data.copy_(
                    torch.where(
                        farfield_mask.repeat(1, 3),
                        clamped_size_farfield,
                        clamped_size_nearfield,
                    )
                )
                # gaussians._scaling.data.clamp_(
                #     min=torch.log(
                #         gaussians.get_scaling.amax(dim=-1) * float(os.getenv("RELMINSIZE", 0.2))
                #     ).unsqueeze(-1)
                # )

        # Might help for rough chromespheres, but leads to slight blurry and no gain in other scenes (0.05)
        if "SKIP_CLAMP_SIZEDIST" not in os.environ:
            with torch.no_grad():
                farfield_mask = ~gaussians.comes_from_colmap_pc.bool()
                distance_to_zero = gaussians.get_xyz.norm(dim=-1)
                clamped_size = gaussians._scaling.clamp(
                    min=torch.log(
                        distance_to_zero * float(os.getenv("SIZEDIST", 0.0025)) + 1e-8
                    ).unsqueeze(-1)
                )
                # using indexing failed to overwrite in place, going with this ugly solution
                gaussians._scaling.data.copy_(
                    torch.where(
                        farfield_mask.repeat(1, 3), clamped_size, gaussians._scaling
                    )
                )

        if "PRUNE_USELESS_GAUSSIANS" in os.environ:
            if i % 1000 == 0:
                breakpoint()

        if iteration in args.checkpoint_iterations:
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save(
                (gaussians.capture(), iteration),
                scene.model_path + "/chkpnt" + str(iteration) + ".pth",
            )

    if iteration == model_params.warmup_until_iter:
        os.environ["DIFFUSE_LOSS_WEIGHT"] = str(model_params.diffuse_loss_weight)
        raytracer.cuda_module.set_losses(True)

    if iteration == model_params.no_bounces_until_iter:
        raytracer.cuda_module.num_bounces.copy_(min(raytracer.config.MAX_BOUNCES, 1))
        if not "SKIP_INIT_FARFIELD" in os.environ:
            torch.cuda.synchronize()
            gaussians.add_farfield_points(scene)
        raytracer.rebuild_bvh()
        torch.cuda.synchronize()

    if (
        iteration == model_params.max_one_bounce_until_iter
        and iteration > model_params.no_bounces_until_iter
    ):
        raytracer.cuda_module.num_bounces.copy_(raytracer.config.MAX_BOUNCES)

    if iteration == model_params.rebalance_losses_at_iter:
        os.environ["GLOSSY_LOSS_WEIGHT"] = str(
            model_params.glossy_loss_weight_after_rebalance
        )
        os.environ["DIFFUSE_LOSS_WEIGHT"] = str(
            model_params.diffuse_loss_weight_after_rebalance
        )
        raytracer.cuda_module.set_losses(True)

    if iteration == model_params.enable_regular_loss_at_iter:
        os.environ["REGULAR_LOSS_WEIGHT"] = "1.0"
        raytracer.cuda_module.set_losses(True)

    if args.viewer:
        viewer.gaussian_lock.release()

# All done
print("\nTraining complete.")
