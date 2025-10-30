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
import json
import os
import time
from datetime import datetime
from random import randint
from threading import Thread

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from editable_gauss_refl.config import Config
from editable_gauss_refl.renderer import GaussianRaytracer, render
from editable_gauss_refl.scene import GaussianModel, Scene
from editable_gauss_refl.utils.general_utils import (
    set_seeds,
)
from editable_gauss_refl.utils.image_utils import psnr
from editable_gauss_refl.utils.tonemapping import tonemap


def prepare_output_and_logger(cfg: Config):
    if not cfg.model_path:
        cfg.model_path = os.path.join("output", datetime.now().isoformat(timespec="seconds"))

    # * Set up output folder
    print("Output folder: {}".format(cfg.model_path))
    os.makedirs(cfg.model_path, exist_ok=True)

    # * Copy transforms json files and bounding_boxes if they exist
    try:
        import shutil

        shutil.copyfile(
            os.path.join(cfg.source_path, "transforms_train.json"),
            os.path.join(cfg.model_path, "transforms_train.json"),
        )
        shutil.copyfile(
            os.path.join(cfg.source_path, "transforms_test.json"),
            os.path.join(cfg.model_path, "transforms_test.json"),
        )
    except Exception as e:
        print("Could not copy transforms json files: ", e)

    try:
        import shutil

        shutil.copyfile(
            os.path.join(cfg.source_path, "bounding_boxes.json"),
            os.path.join(cfg.model_path, "bounding_boxes.json"),
        )
    except Exception:
        pass

    # * Dump cfg as JSON.
    with open(os.path.join(cfg.model_path, "cfg.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)

    return SummaryWriter(cfg.model_path)


@torch.no_grad()
def training_report(cfg: Config, scene, raytracer, tb_writer, iteration, start_time):
    # * Save the elapsed time
    delta = time.time() - start_time
    with open(os.path.join(cfg.model_path, "time.txt"), "a") as f:
        minutes, seconds = divmod(int(delta), 60)
        timestamp = f"{minutes:02}:{seconds:02}"
        print("Elapsed time: ", timestamp)
        f.write("\n[ITER {}] elapsed {}".format(iteration, time.strftime("%H:%M:%S", time.gmtime(delta))))

    # * Save the number of gaussians
    with open(os.path.join(cfg.model_path, "num_gaussians.txt"), "a") as f:
        f.write("\n[ITER {}] # {}".format(iteration, scene.gaussians.get_xyz.shape[0]))
        print("Number of gaussians: ", scene.gaussians.get_xyz.shape[0])

    # * Run validation
    validation_configs = []
    validation_configs.append(
        {"name": "train", "cameras": [sorted(scene.getTrainCameras(), key=lambda x: x.image_name)[min(cfg.val_view, (cfg.max_images or 1) - 1)]]},
    )
    if len(scene.getTestCameras()) > 0:
        validation_configs.append(
            {"name": "test", "cameras": scene.getTestCameras()},
        )
    for config in validation_configs:
        psnr_test = 0.0
        specular_psnr_test = 0.0
        diffuse_psnr_test = 0.0

        for idx, viewpoint in enumerate(config["cameras"]):
            package = render(viewpoint, raytracer, denoise=True)

            os.makedirs(os.path.join(tb_writer.log_dir, f"{config['name']}_preview"), exist_ok=True)

            diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
            specular_image = tonemap(package.rgb[1:].sum(dim=0)).clamp(0, 1)
            pred_image = tonemap(package.final[0]).clamp(0, 1)
            pred_image_without_denoising = tonemap(package.rgb.sum(dim=0))
            diffuse_gt_image = tonemap(viewpoint.diffuse_image).clamp(0, 1)
            specular_gt_image = tonemap(viewpoint.specular_image).clamp(0, 1)
            gt_image = tonemap(viewpoint.original_image).clamp(0, 1)

            if tb_writer and idx == 0:
                preview = torch.stack([diffuse_image, diffuse_gt_image, specular_image, specular_gt_image, pred_image, gt_image]).clamp(0, 1)
                save_image(preview, os.path.join(tb_writer.log_dir, f"{config['name']}_preview_iteration_{iteration}.png"), nrow=2, padding=0)

            normal_gt_image = torch.clamp(viewpoint.normal_image / 2 + 0.5, 0.0, 1.0)
            roughness_image = torch.clamp(package.roughness[0], 0.0, 1.0)
            normal_image = torch.clamp(package.normal[0] / 2 + 0.5, 0.0, 1.0)
            depth_image = package.depth[0]

            f0_image = torch.clamp(package.f0[0], 0.0, 1.0)
            normal_gt_image = torch.clamp(viewpoint.normal_image / 2 + 0.5, 0.0, 1.0)
            depth_gt_image = viewpoint.depth_image
            f0_gt_image = torch.clamp(viewpoint.f0_image, 0.0, 1.0)
            roughness_gt_image = torch.clamp(viewpoint.roughness_image, 0.0, 1.0)

            diffuse_psnr_test += psnr(diffuse_image, diffuse_gt_image).mean().double()
            specular_psnr_test += psnr(specular_image, specular_gt_image).mean().double()
            psnr_test += psnr(pred_image, gt_image).mean().double()

            if tb_writer and idx == 0:
                all_rays_dir = os.path.join(tb_writer.log_dir, f"{config['name']}_preview", f"iteration_{iteration}", "all_rays")
                os.makedirs(all_rays_dir, exist_ok=True)
                save_image(tonemap(package.rgb).clamp(0, 1), os.path.join(all_rays_dir, "rgb.png"), padding=0)
                save_image(torch.clamp(package.normal / 2 + 0.5, 0.0, 1.0), os.path.join(all_rays_dir, "normal.png"), padding=0)
                save_image(torch.clamp(package.f0, 0.0, 1.0), os.path.join(all_rays_dir, "f0.png"), padding=0)
                depth_rescaled = (package.depth - package.depth.amin()) / (package.depth.amax() - package.depth.amin())
                save_image(depth_rescaled, os.path.join(all_rays_dir, "depth.png"), padding=0)

                vs_target_dir = os.path.join(tb_writer.log_dir, f"{config['name']}_preview", f"iteration_{iteration}", "vs_target")
                os.makedirs(vs_target_dir, exist_ok=True)
                save_image(torch.stack([roughness_image.cuda(), roughness_gt_image]).clamp(0, 1), os.path.join(vs_target_dir, "roughness.png"), nrow=2, padding=0)
                save_image(torch.stack([f0_image.cuda(), f0_gt_image]).clamp(0, 1), os.path.join(vs_target_dir, "f0.png"), nrow=2, padding=0)
                save_image(torch.stack([pred_image, gt_image]).clamp(0, 1), os.path.join(vs_target_dir, "final_denoised.png"), nrow=2, padding=0)
                save_image(torch.stack([pred_image_without_denoising, gt_image]).clamp(0, 1), os.path.join(vs_target_dir, "final_without_denoising.png"), nrow=2, padding=0)
                save_image(torch.stack([diffuse_image, diffuse_gt_image]).clamp(0, 1), os.path.join(vs_target_dir, "diffuse.png"), nrow=2, padding=0)
                save_image(torch.stack([specular_image, specular_gt_image]).clamp(0, 1), os.path.join(vs_target_dir, "specular.png"), nrow=2, padding=0)
                depth_rescaled = (torch.stack([depth_image.cuda(), depth_gt_image]) - depth_gt_image.amin()) / (depth_gt_image.amax() - depth_gt_image.amin())
                save_image(depth_rescaled, os.path.join(vs_target_dir, "depth.png"), nrow=2, padding=0)
                save_image(torch.stack([normal_image.cuda(), normal_gt_image]).clamp(0, 1), os.path.join(vs_target_dir, "normal.png"), nrow=2, padding=0)

        psnr_test /= len(config["cameras"])
        diffuse_psnr_test /= len(config["cameras"])
        specular_psnr_test /= len(config["cameras"])

        print("\n[ITER {}] Evaluating {}: PSNR {}".format(iteration, config["name"], psnr_test))
        if tb_writer:
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - specular_psnr", specular_psnr_test, iteration)
            tb_writer.add_scalar(config["name"] + "/loss_viewpoint - diffuse_psnr", diffuse_psnr_test, iteration)

        with open(os.path.join(tb_writer.log_dir, f"{config['name']}_validation_scores.csv"), "a") as f:
            f.write(f"{iteration}, {diffuse_psnr_test:02.2f}, {specular_psnr_test:02.2f}, {psnr_test:02.2f}\n")

    torch.cuda.empty_cache()


def main(cfg: Config):
    set_seeds()

    tb_writer = prepare_output_and_logger(cfg)
    gaussians = GaussianModel(cfg)
    scene = Scene(cfg, gaussians)
    gaussians.training_setup(cfg)

    first_iter = 0
    first_iter += 1

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    raytracer = GaussianRaytracer(gaussians, viewpoint_stack[0].image_width, viewpoint_stack[0].image_height)

    if cfg.viewer:
        from gaussian_viewer import GaussianViewer
        from viewer.types import ViewerMode

        mode = ViewerMode.LOCAL if cfg.viewer_mode == "local" else ViewerMode.SERVER
        viewer = GaussianViewer.from_gaussians(raytracer, cfg, gaussians, False, mode)
        viewer.accumulate_samples = False
        if cfg.viewer_mode != "none":
            viewer_thd = Thread(target=viewer.run, daemon=True)
            viewer_thd.start()

    start_time = time.time()

    config = raytracer.cuda_module.get_config()
    MAX_BOUNCES = config.num_bounces.item()
    config.num_bounces.fill_(0)

    if cfg.no_bounces_until_iter > 0:
        config.num_bounces.copy_(0)
    elif cfg.max_one_bounce_until_iter > 0:
        config.num_bounces.copy_(min(raytracer.config.MAX_BOUNCES, 1))

    for iteration in tqdm(range(first_iter, cfg.iterations + 1), desc="Training progress", total=cfg.iterations, initial=first_iter):
        iter_start.record()

        if cfg.viewer:
            viewer.gaussian_lock.acquire()

        gaussians.update_learning_rate(iteration)
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render(viewpoint_cam, raytracer, denoise=False)

        with torch.no_grad():
            if cfg.scale_decay < 1.0:
                gaussians._scaling.copy_(torch.log(gaussians.get_scaling * cfg.scale_decay))

        iter_end.record()

        with torch.no_grad():
            if iteration in cfg.test_iterations:
                training_report(cfg, scene, raytracer, tb_writer, iteration, start_time)

            if iteration in cfg.save_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % cfg.pruning_interval == 0:
                if iteration > cfg.pruning_start_iter and cfg.min_weight > 0:
                    gaussians.prune_points((raytracer.cuda_module.get_gaussians().total_weight / cfg.pruning_interval < cfg.min_weight).squeeze(1))
                if not cfg.disable_znear_densif_pruning:
                    gaussians.prune_znear_only(scene)
                raytracer.cuda_module.get_gaussians().total_weight.zero_()

                raytracer.rebuild_bvh()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=False)
            raytracer.zero_grad()

            with torch.no_grad():
                gaussians._diffuse.data.clamp_(min=0.0)
                gaussians._roughness.data.clamp_(min=0.0, max=1.0)
                gaussians._f0.data.clamp_(min=0.0, max=1.0)

        if iteration == cfg.no_bounces_until_iter:
            config.num_bounces.copy_(MAX_BOUNCES)

            gaussians.add_farfield_points(scene)
            raytracer.rebuild_bvh()

        if iteration == 1 and cfg.no_bounces_until_iter in [-1, 0]:
            gaussians.add_farfield_points(scene)
            raytracer.rebuild_bvh()

        if cfg.viewer:
            viewer.gaussian_lock.release()

    print("\nTraining complete.")


if __name__ == "__main__":
    cfg = tyro.cli(Config)

    if cfg.viewer:
        cfg.test_iterations = []

    main(cfg)
