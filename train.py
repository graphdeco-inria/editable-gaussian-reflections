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
import random
import time
from datetime import datetime
from random import randint
from threading import Thread

import torch
import tyro
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from gaussian_tracing.arguments import (
    TyroConfig,
)
from gaussian_tracing.renderer import GaussianRaytracer, render
from gaussian_tracing.scene import GaussianModel, Scene
from gaussian_tracing.utils.general_utils import (
    inverse_sigmoid,
    safe_state,
)
from gaussian_tracing.utils.image_utils import psnr
from gaussian_tracing.utils.loss_utils import l1_loss
from gaussian_tracing.utils.tonemapping import tonemap

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(cfg: TyroConfig):
    if not cfg.model_path:
        cfg.model_path = os.path.join(
            "./output/", datetime.now().isoformat(timespec="seconds")
        )

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))
    os.makedirs(cfg.model_path, exist_ok=True)
    # Dump cfg.
    with open(f"{cfg.model_path}/cfg.yml", "w") as f:
        yaml.dump(vars(cfg), f)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(
    cfg: TyroConfig,
    scene,
    raytracer,
    tb_writer,
    iteration,
):
    if iteration in cfg.test_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    sorted(scene.getTrainCameras(), key=lambda x: x.image_name)[
                        idx % len(scene.getTrainCameras())
                    ]
                    for idx in cfg.val_views
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
                    package = render(
                        viewpoint,
                        raytracer,
                        denoise=cfg.denoise,
                    )

                    os.makedirs(
                        tb_writer.log_dir + "/" + f"{config['name']}_view",
                        exist_ok=True,
                    )

                    if "SKIP_TONEMAPPING_OUTPUT" in os.environ:
                        diffuse_image = package.rgb[0].clamp(0, 1)
                        glossy_image = package.rgb[1:].sum(dim=0)
                        pred_image = package.final[0].clamp(0, 1)
                        pred_image_without_denoising = package.rgb.sum(dim=0)
                        diffuse_gt_image = torch.clamp(
                            viewpoint.diffuse_image, 0.0, 1.0
                        )
                        glossy_gt_image = torch.clamp(viewpoint.glossy_image, 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    else:
                        diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
                        glossy_image = tonemap(package.rgb[1:].sum(dim=0)).clamp(0, 1)
                        pred_image = tonemap(package.final[0]).clamp(0, 1)
                        pred_image_without_denoising = tonemap(package.rgb.sum(dim=0))
                        diffuse_gt_image = tonemap(viewpoint.diffuse_image).clamp(0, 1)
                        glossy_gt_image = tonemap(viewpoint.glossy_image).clamp(0, 1)
                        gt_image = tonemap(viewpoint.original_image).clamp(0, 1)

                    if tb_writer and (idx < len(cfg.val_views)):
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
                    depth_image = package.depth[0]

                    F0_image = torch.clamp(package.F0[0], 0.0, 1.0)
                    normal_gt_image = torch.clamp(
                        viewpoint.normal_image / 2 + 0.5, 0.0, 1.0
                    )
                    depth_gt_image = viewpoint.depth_image
                    F0_gt_image = torch.clamp(viewpoint.F0_image, 0.0, 1.0)
                    roughness_gt_image = torch.clamp(
                        viewpoint.roughness_image, 0.0, 1.0
                    )

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

                    if tb_writer and (idx < len(cfg.val_views)):
                        if package.rgb.shape[0] > 2:
                            save_image(
                                tonemap(package.rgb).clamp(0, 1),
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
                                package.depth
                                / package.depth.amax(dim=(1, 2, 3), keepdim=True),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_depth_all_rays.png",
                                padding=0,
                            )
                            save_image(
                                torch.clamp(package.brdf, 0.0, 1.0),
                                tb_writer.log_dir
                                + "/"
                                + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_brdf_all_rays.png",
                                padding=0,
                            )

                        fb = raytracer.cuda_module.get_framebuffer()
                        save_image(
                            fb.output_ray_origin[0].moveaxis(-1, 0).abs() / 5,
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_ray_origin.png",
                            padding=0,
                        )
                        save_image(
                            fb.output_ray_direction[0].moveaxis(-1, 0) / 2 + 0.5,
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_ray_direction.png",
                            padding=0,
                        )
                        stats = raytracer.cuda_module.get_stats()
                        torch.save(
                            stats.num_traversed_per_pixel,
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_num_traversed_per_pixel.pt",
                        )
                        torch.save(
                            stats.num_traversed_per_pixel,
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_num_traversed_per_pixel.pt",
                        )
                        # also save them as normalized png
                        save_image(
                            (
                                stats.num_traversed_per_pixel.float()
                                / stats.num_traversed_per_pixel.max()
                            ),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_num_traversed_per_pixel.png",
                            padding=0,
                        )
                        save_image(
                            (
                                stats.num_traversed_per_pixel.float()
                                / stats.num_traversed_per_pixel.max()
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
                            (
                                torch.stack(
                                    [depth_image.cuda(), depth_gt_image.unsqueeze(0)]
                                )
                                - depth_gt_image.amin()
                            )
                            / (depth_gt_image.amax() - depth_gt_image.amin()),
                            tb_writer.log_dir
                            + "/"
                            + f"{config['name']}_view/iter_{iteration:09}_view_{viewpoint.colmap_id}_depth_vs_target.png",
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


def main(cfg: TyroConfig):
    model_params = cfg.model_params
    opt_params = cfg.opt_params

    # Initialize system state (RNG)
    safe_state(cfg.quiet)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

    tb_writer = prepare_output_and_logger(cfg)
    gaussians = GaussianModel(cfg)
    scene = Scene(cfg, gaussians)
    gaussians.training_setup(opt_params)

    first_iter = 0
    if cfg.start_checkpoint:
        (capture_data, first_iter) = torch.load(
            cfg.start_checkpoint, weights_only=False
        )  #!!! Cant release like this, security risk
        gaussians.restore(capture_data, opt_params)
    first_iter += 1

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    raytracer = GaussianRaytracer(
        gaussians, viewpoint_stack[0].image_width, viewpoint_stack[0].image_height
    )

    if "KEYVIEW" in os.environ:
        keyview = [
            x
            for x in scene.getTrainCameras()
            if x.colmap_id == int(os.environ["KEYVIEW"])
        ][0]

    if cfg.viewer:
        from gaussianviewer import GaussianViewer
        from viewer.types import ViewerMode

        mode = ViewerMode.LOCAL if cfg.viewer_mode == "local" else ViewerMode.SERVER
        SPARSE_ADAM_AVAILABLE = False
        viewer = GaussianViewer.from_gaussians(
            raytracer, model_params, opt_params, gaussians, SPARSE_ADAM_AVAILABLE, mode
        )
        viewer.accumulate_samples = False
        if cfg.viewer_mode != "none":
            viewer_thd = Thread(target=viewer.run, daemon=True)
            viewer_thd.start()

    start = time.time()

    config = raytracer.cuda_module.get_config()
    MAX_BOUNCES = config.num_bounces.item()
    config.num_bounces.fill_(0)

    if model_params.no_bounces_until_iter > 0:
        config.num_bounces.copy_(0)
    elif model_params.max_one_bounce_until_iter > 0:
        config.num_bounces.copy_(min(raytracer.config.MAX_BOUNCES, 1))

    for iteration in tqdm(
        range(first_iter, cfg.iterations + 1),
        desc="Training progress",
        total=cfg.iterations,
        initial=first_iter,
    ):
        iter_start.record()

        if cfg.viewer:
            viewer.gaussian_lock.acquire()

        _ = gaussians.update_learning_rate(iteration)
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if "KEYVIEW" in os.environ and random.random() < 0.5:
            viewpoint_cam = keyview

        torch.cuda.synchronize()  # todo may be needed or not, idk, occasional crash. double check after deadline
        _ = render(
            viewpoint_cam,
            raytracer,
            denoise=cfg.denoise,
        )

        if opt_params.opacity_reg > 0:
            gaussians._opacity.grad += torch.autograd.grad(
                cfg.opacity_reg * torch.abs(gaussians.get_opacity).mean(),
                gaussians._opacity,
            )[0]
        if opt_params.scale_reg > 0:
            gaussians._scaling.grad += torch.autograd.grad(
                cfg.scale_reg * torch.abs(gaussians.get_scaling).mean(),
                gaussians._scaling,
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
            NOW = time.time()

            training_report(cfg, scene, raytracer, tb_writer, iteration)
            torch.cuda.synchronize()  # not sure if needed

            if iteration % 1000 == 0 or iteration == 1:
                os.makedirs(os.path.join(cfg.model_path, "plots"), exist_ok=True)

                # Save the elapsed time
                delta = time.time() - start
                with open(os.path.join(cfg.model_path, "time.txt"), "a") as f:
                    minutes, seconds = divmod(int(delta), 60)
                    timestamp = f"{minutes:02}:{seconds:02}"
                    print("Elapsed time: ", timestamp)
                    f.write(
                        "\n[ITER {}] elapsed {}".format(
                            iteration,
                            time.strftime("%H:%M:%S", time.gmtime(NOW - start)),
                        )
                    )

                # Save the average and std opacity
                with open(os.path.join(cfg.model_path, "opacity.txt"), "a") as f:
                    f.write(
                        f"{iteration:5}: {gaussians.get_opacity.mean().item():.3f} +- {gaussians.get_opacity.std().item():.3f}\n"
                    )

                # Save the average and std size
                with open(os.path.join(cfg.model_path, "size.txt"), "a") as f:
                    f.write(
                        f"{iteration:5}: {gaussians.get_scaling.mean().item():.3f} +- {gaussians.get_scaling.std().item():.3f}\n"
                    )

                # Save the average and std size of the largest axis per gaussian
                with open(os.path.join(cfg.model_path, "size_axis_max.txt"), "a") as f:
                    f.write(
                        f"{iteration:5}: {gaussians.get_scaling.amax(dim=1).mean().item():.5f} +- {gaussians.get_scaling.amax(dim=1).std().item():.5f}\n"
                    )

                # same but for the median axis
                with open(
                    os.path.join(cfg.model_path, "size_axis_median.txt"), "a"
                ) as f:
                    f.write(
                        f"{iteration:5}: {gaussians.get_scaling.median(dim=1).values.mean().item():.5f} +- {gaussians.get_scaling.median(dim=1).values.std().item():.5f}\n"
                    )

                # Same but for the smallest axis
                with open(os.path.join(cfg.model_path, "size_axis_min.txt"), "a") as f:
                    f.write(
                        f"{iteration:5}: {gaussians.get_scaling.amin(dim=1).mean().item():.5f} +- {gaussians.get_scaling.amin(dim=1).std().item():.5f}\n"
                    )

                stats = raytracer.cuda_module.get_stats()

                # From raytracer.num_hits, print the mean, max, and std
                num_traversed = stats.num_traversed_per_pixel.float()
                with open(os.path.join(cfg.model_path, "num_traversed.txt"), "a") as f:
                    f.write(
                        f"{iteration:5}: {num_traversed.mean().item():.3f} +- {num_traversed.std().item():.3f}\n"
                    )
                num_accumulated = stats.num_accumulated_per_pixel.float()
                with open(
                    os.path.join(cfg.model_path, "num_accumulated.txt"), "a"
                ) as f:
                    f.write(
                        f"{iteration:5}: {num_accumulated.mean().item():.3f} +- {num_accumulated.std().item():.3f}\n"
                    )

                with open(os.path.join(cfg.model_path, "num_gaussians.txt"), "a") as f:
                    f.write(
                        "\n[ITER {}] # {}".format(
                            iteration, scene.gaussians.get_xyz.shape[0]
                        )
                    )
                    print("Number of gaussians: ", gaussians.get_xyz.shape[0])

            if iteration in cfg.save_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % opt_params.densification_interval == 0:
                if (
                    iteration > model_params.no_bounces_until_iter + 500
                    and model_params.min_weight > 0
                ):
                    gaussians.prune_points(
                        (
                            raytracer.cuda_module.get_gaussians().total_weight
                            / opt_params.densification_interval
                            < model_params.min_weight
                        ).squeeze(1)
                    )
                if model_params.znear_densif_pruning:
                    gaussians.prune_znear_only(scene)
                raytracer.cuda_module.get_gaussians().total_weight.zero_()
                assert "DENSIFY" not in os.environ

                torch.cuda.synchronize()
                raytracer.rebuild_bvh()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=False)
            raytracer.zero_grad()

            with torch.no_grad():
                gaussians._diffuse.data.clamp_(min=0.0)
                gaussians._roughness.data.clamp_(min=0.0, max=1.0)
                gaussians._f0.data.clamp_(min=0.0, max=1.0)

            if iteration in cfg.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    cfg.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

        if iteration == model_params.no_bounces_until_iter:
            if model_params.max_one_bounce_until_iter in [0, -1]:
                config.num_bounces.copy_(MAX_BOUNCES)
            else:
                config.num_bounces.copy_(min(MAX_BOUNCES, 1))

            if "SKIP_INIT_FARFIELD" not in os.environ:
                torch.cuda.synchronize()
                gaussians.add_farfield_points(scene)
            raytracer.rebuild_bvh()
            torch.cuda.synchronize()

        if iteration == 1 and (
            model_params.no_bounces_until_iter in [-1, 0]
            or model_params.no_bounces_until_iter > 900_000
        ):
            gaussians.add_farfield_points(scene)
            raytracer.rebuild_bvh()
            torch.cuda.synchronize()

        if (
            iteration == model_params.max_one_bounce_until_iter
            and iteration > model_params.no_bounces_until_iter
        ):
            config.num_bounces.copy_(config.num_bounces)

        if cfg.viewer:
            viewer.gaussian_lock.release()

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    cfg = tyro.cli(TyroConfig)

    # TODO: Remove this custom config modification.
    if cfg.viewer:
        cfg.test_iterations = []
    if cfg.opt_params.timestretch != 1:
        cfg.model_params.no_bounces_until_iter = int(
            cfg.model_params.no_bounces_until_iter * cfg.opt_params.timestretch
        )
        cfg.model_params.max_one_bounce_until_iter = int(
            cfg.model_params.max_one_bounce_until_iter * cfg.opt_params.timestretch
        )
        cfg.test_iterations = [
            int(x * cfg.opt_params.timestretch) for x in cfg.test_iterations
        ]
        cfg.save_iterations = [
            int(x * cfg.opt_params.timestretch) for x in cfg.save_iterations
        ]
        cfg.iterations = int(cfg.opt_params.timestretch * cfg.iterations)
        cfg.opt_params.densification_interval = int(
            cfg.opt_params.timestretch * cfg.opt_params.densification_interval
        )
        cfg.opt_params.densify_from_iter = int(
            cfg.opt_params.timestretch * cfg.opt_params.densify_from_iter
        )
        cfg.opt_params.densify_until_iter = int(
            cfg.opt_params.timestretch * cfg.opt_params.densify_until_iter
        )

    main(cfg)
