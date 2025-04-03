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
from utils.general_utils import colormap
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, GaussianRaytracer
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
import copy 
from utils.graphics_utils import BasicPointCloud
from datetime import datetime
import time
from scene.gaussian_model import build_scaling_rotation
import math 
from scene.tonemapping import *
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
import pandas as pd
import plotly.express as px
import random

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def prepare_output_and_logger(args: ModelParams, opt_params):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", datetime.now().isoformat(timespec="seconds"))

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "model_params"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    with open(os.path.join(args.model_path, "opt_params"), 'w') as cfg_log_f:
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
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                glossy_l1_test = 0.0
                glossy_psnr_test = 0.0
                
                diffuse_l1_test = 0.0
                diffuse_psnr_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    package = render(viewpoint, raytracer, pipe_params, bg)

                    os.makedirs(tb_writer.log_dir + "/" + f"{config['name']}_view", exist_ok=True)

                    if "TONEMAP_IMAGES_AT_INPUT" in os.environ:
                        diffuse_image = package.rgb[0].clamp(0, 1)
                        glossy_image = package.rgb[1:-1].sum(dim=0)
                        pred_image = package.rgb[-1].clamp(0, 1)
                        diffuse_gt_image = torch.clamp(viewpoint.diffuse_image, 0.0, 1.0)
                        glossy_gt_image = torch.clamp(viewpoint.glossy_image, 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    elif raytracer.config.TONEMAP:
                        diffuse_image = package.rgb[0].clamp(0, 1)
                        glossy_image = tonemap(untonemap(package.rgb[1:-1]).sum(dim=0)).clamp(0, 1)
                        pred_image = package.rgb[-1].clamp(0, 1)
                        diffuse_gt_image = torch.clamp(viewpoint.diffuse_image, 0.0, 1.0)
                        glossy_gt_image = torch.clamp(viewpoint.glossy_image, 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    else:
                        diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
                        glossy_image = tonemap(package.rgb[1:-1].sum(dim=0)).clamp(0, 1)
                        pred_image = tonemap(package.rgb[-1]).clamp(0, 1)
                        diffuse_gt_image = tonemap(viewpoint.diffuse_image).clamp(0, 1)
                        glossy_gt_image = tonemap(viewpoint.glossy_image).clamp(0, 1)
                        gt_image = tonemap(viewpoint.original_image).clamp(0, 1)

                    if tb_writer and (idx < 5): 
                        error_diffuse = diffuse_image - diffuse_gt_image
                        error_glossy = glossy_image - glossy_gt_image
                        error_final = pred_image - gt_image
                        save_image(torch.stack([
                            diffuse_image, diffuse_gt_image, error_diffuse.abs() / error_diffuse.std() / 3,
                            glossy_image, glossy_gt_image, error_glossy.abs() / error_glossy.std() / 3,
                            pred_image, gt_image, error_final.abs() / error_final.std() / 3,
                        ]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}.png", nrow=3)
                    
                    normal_gt_image = torch.clamp(viewpoint.normal_image / 2 + 0.5, 0.0, 1.0)
                    roughness_image = torch.clamp(package.roughness[0], 0.0, 1.0)
                    normal_image = torch.clamp(package.normal[0] / 2 + 0.5, 0.0, 1.0) 
                    position_image = torch.clamp(package.position[0], 0.0, 1.0)
                    F0_image = torch.clamp(package.F0[0], 0.0, 1.0)
                    if model_params.brdf_mode != "disabled":
                        brdf_image = torch.clamp(package.brdf[0], 0.0, 1.0)

                    normal_gt_image = torch.clamp(viewpoint.normal_image / 2 + 0.5, 0.0, 1.0)
                    position_gt_image = torch.clamp(viewpoint.position_image, 0.0, 1.0)
                    F0_gt_image = torch.clamp(viewpoint.F0_image, 0.0, 1.0)
                    roughness_gt_image = torch.clamp(viewpoint.roughness_image, 0.0, 1.0)
                    if model_params.brdf_mode != "disabled":
                        brdf_gt_image = torch.clamp(viewpoint.brdf_image, 0.0, 1.0)

                    diffuse_l1_test += l1_loss(diffuse_image, diffuse_gt_image).mean().double()
                    diffuse_psnr_test += psnr(diffuse_image, diffuse_gt_image).mean().double()
                    glossy_l1_test += l1_loss(glossy_image, glossy_gt_image).mean().double()
                    glossy_psnr_test += psnr(glossy_image, glossy_gt_image).mean().double()
                    l1_test += l1_loss(pred_image, gt_image).mean().double()
                    psnr_test += psnr(pred_image, gt_image).mean().double()

                    if tb_writer and (idx < 5): 
                        if package.rgb.shape[0] > 2:
                            for k, (_rgb_img, _normal_img, _pos_image, _F0_image, _brdf_image) in enumerate(zip(package.rgb, package.normal, package.position, package.F0, package.brdf)):
                                if not raytracer.config.TONEMAP:
                                    _rgb_img = tonemap(_rgb_img)
                                save_image(_rgb_img.clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_rgb_bounce_{k}.png", padding=0)
                                save_image(torch.clamp(_normal_img / 2 + 0.5, 0.0, 1.0), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_normal_bounce_{k}.png", padding=0)
                                save_image(torch.clamp(_F0_image, 0.0, 1.0), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_F0_bounce_{k}.png", padding=0)
                                save_image(torch.clamp(_pos_image, 0.0, 1.0), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_pos_bounce_{k}.png", padding=0)
                                save_image(torch.clamp(_brdf_image, 0.0, 1.0), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_brdf_bounce_{k}.png", padding=0)

                                if raytracer.cuda_module.output_incident_radiance is not None:
                                    save_image(raytracer.cuda_module.output_incident_radiance[k].clamp(0, 1).moveaxis(-1, 0), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_incident_radiance_{k}.png", padding=0)
                        
                        if raytracer.config.SAVE_LOD_IMAGES:
                            for k, (lod_mean, lod_scale, ray_lod) in enumerate(zip(raytracer.cuda_module.output_lod_mean, raytracer.cuda_module.output_lod_scale, raytracer.cuda_module.output_ray_lod)):
                                save_image(lod_mean.moveaxis(-1, 0), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_lod_mean_{k}.png", padding=0)
                                save_image(lod_scale.moveaxis(-1, 0), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_lod_scale_{k}.png", padding=0)
                                save_image(lod_mean.moveaxis(-1, 0), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_ray_lod_{k}.png", padding=0)

                        if raytracer.config.SAVE_HIT_STATS:
                            torch.save(raytracer.cuda_module.num_hits_per_pixel, tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_num_hits_per_pixel.pt")
                            torch.save(raytracer.cuda_module.num_traversed_per_pixel, tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_num_traversed_per_pixel.pt")
                            # also save them as normalized png 
                            save_image((raytracer.cuda_module.num_hits_per_pixel.float() / raytracer.cuda_module.num_hits_per_pixel.max()), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_num_hits_per_pixel.png", padding=0)
                            save_image((raytracer.cuda_module.num_traversed_per_pixel.float() / raytracer.cuda_module.num_traversed_per_pixel.max()), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_num_traversed_per_pixel.png", padding=0)

                        save_image(torch.stack([roughness_image.cuda(), roughness_gt_image]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_roughness.png", nrow=2, padding=0)
                        save_image(torch.stack([F0_image.cuda(), F0_gt_image]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_F0.png", nrow=2, padding=0)
                        save_image(torch.stack([pred_image, gt_image]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_final.png", nrow=2, padding=0)
                        save_image(torch.stack([diffuse_image, diffuse_gt_image]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_diffuse.png", nrow=2, padding=0)
                        save_image(torch.stack([glossy_image, glossy_gt_image]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_glossy.png", nrow=2, padding=0)
                        save_image(torch.stack([position_image.cuda(), position_gt_image]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_position.png", nrow=2, padding=0)
                        save_image(torch.stack([normal_image.cuda(), normal_gt_image]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_normal.png", nrow=2, padding=0)
                        save_image(torch.stack([normal_image.cuda(), normal_gt_image]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_normal.png", nrow=2, padding=0)
                        if model_params.brdf_mode != "disabled":
                            save_image(torch.stack([brdf_image, brdf_gt_image.cuda()]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_brdf.png", nrow=2, padding=0)

                        if raytracer.cuda_module.output_incident_radiance is not None:
                            save_image(raytracer.cuda_module.output_incident_radiance[1].clamp(0, 1).moveaxis(-1, 0), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_incident_radiance.png", padding=0)
                            save_image(raytracer.cuda_module.output_incident_radiance[1].clamp(0, 1).moveaxis(-1, 0) * package.brdf[0], tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_sanity_check.png", padding=0)
                            torch.save(raytracer.cuda_module.output_incident_radiance[1].clamp(0, 1).moveaxis(-1, 0), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_incident_radiance.pt")

                        if raytracer.config.USE_LEVEL_OF_DETAIL:
                            for k, alpha in enumerate(torch.linspace(0.0, 1.0, 4)):
                                package = render(viewpoint, raytracer, pipe_params, bg, blur_sigma=alpha * scene.max_pixel_blur_sigma if not model_params.lod_force_blur_sigma >= 0.0 else torch.tensor(model_params.lod_force_blur_sigma, device="cuda"))

                                diffuse_gt_image = package.target_diffuse
                                glossy_gt_image = package.target_glossy
                                gt_image = package.target
                            
                                if raytracer.config.TONEMAP:
                                    diffuse_image = tonemap(package.rgb[0]).clamp(0, 1)
                                    glossy_image = tonemap(package.rgb[1:-1].sum(dim=0)).clamp(0, 1)
                                    pred_image = tonemap(package.rgb[-1]).clamp(0, 1)
                                else:
                                    diffuse_image = package.rgb[0].clamp(0, 1)
                                    glossy_image = package.rgb[1:-1].sum(dim=0).clamp(0, 1)
                                    pred_image = package.rgb[-1].clamp(0, 1)
                                    
                                save_image(torch.stack([diffuse_pred, diffuse_gt_image, glossy_pred, glossy_gt_image, pred, gt_image]).clamp(0, 1), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_blurred_{k}÷3.png", nrow=2, padding=0)        

                if model_params.brdf_mode == "static_lut":
                    save_image(torch.stack([ gaussians.get_brdf_lut ]).abs(), os.path.join(tb_writer.log_dir, f"{config['name']}_view/lut_iter_{iteration:09}.png"), nrow=1, padding=0)
                elif model_params.brdf_mode == "finetuned_lut":
                    
                    save_image(torch.stack([ gaussians._brdf_lut, gaussians.get_brdf_lut ]), os.path.join(tb_writer.log_dir, f"{config['name']}_view/lut_iter_{iteration:09}.png"), nrow=1, padding=0)
                    save_image(torch.stack([ gaussians._brdf_lut_residual, gaussians._brdf_lut_residual * 5, gaussians._brdf_lut_residual * 10, gaussians._brdf_lut_residual * 20, gaussians._brdf_lut_residual * 50, gaussians._brdf_lut_residual * 200, gaussians._brdf_lut_residual * 10000 ]).abs(), os.path.join(tb_writer.log_dir, f"{config['name']}_view/lut_residual_amplified_iter_{iteration:09}.png"), nrow=1, padding=0)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                diffuse_psnr_test /= len(config['cameras'])
                diffuse_l1_test /= len(config['cameras'])

                glossy_psnr_test /= len(config['cameras'])
                glossy_l1_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - glossy_l1_loss', glossy_l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - glossy_psnr', glossy_psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - diffuse_l1_loss', diffuse_l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - diffuse_psnr', diffuse_psnr_test, iteration)
                
                with open(os.path.join(tb_writer.log_dir, f"losses_{config['name']}.csv"), "a") as f:
                    f.write(f"{iteration:05d}, {diffuse_psnr_test:02.2f}, {glossy_psnr_test:02.2f}, {psnr_test:02.2f}\n")

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()

# Set up command line argument parser
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument('--flip_camera', action='store_true', default=False)
# parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 30_000, 60_000, 90_000])
# parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 2_000, 3_000, 5_000, 10_000, 20_000, 30_000, 60_000, 90_000])
# parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 3_000, 7_000, 15_000, 22_500, 30_000, 60_000, 90_000])
# parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 3_000, 7_000, 15_000, 22_500, 30_000, 60_000, 90_000])
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
# parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
# parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--viewer", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default = None)
args = parser.parse_args(sys.argv[1:])
args.save_iterations.append(args.iterations)
torch.autograd.set_detect_anomaly(args.detect_anomaly)
model_params = lp.extract(args)
opt_params = op.extract(args)
pipe_params = pp.extract(args)

if opt_params.slowdown != 1:
    opt_params.iterations = int(opt_params.slowdown * opt_params.iterations)
    opt_params.densification_interval = int(opt_params.slowdown * opt_params.densification_interval)
    opt_params.position_lr_max_steps = int(opt_params.slowdown * opt_params.position_lr_max_steps)
    opt_params.densify_from_iter = int(opt_params.slowdown * opt_params.densify_from_iter)
    opt_params.densify_until_iter = int(opt_params.slowdown * opt_params.densify_until_iter)

print("Optimizing " + args.model_path)

# Initialize system state (RNG)
safe_state(args.quiet)

# Start GUI server, configure and run training
if args.viewer:
    network_gui.init(args.ip, args.port)

first_iter = 0
tb_writer = prepare_output_and_logger(model_params, opt_params) 

gaussians = GaussianModel(model_params)

scene = Scene(model_params, gaussians)

gaussians.training_setup(opt_params)

if args.start_checkpoint:
    (model_params, first_iter) = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt_params)

bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

iter_start = torch.cuda.Event(enable_timing=True)
iter_end = torch.cuda.Event(enable_timing=True)

viewpoint_stack = scene.getTrainCameras().copy()
raytracer = GaussianRaytracer(gaussians, viewpoint_stack[0])
raytracer.cuda_module.num_samples.fill_(model_params.num_samples)

ema_loss_for_log = 0.0
first_iter += 1

start = time.time()

if model_params.no_bounces_until_iter > 0:
    raytracer.cuda_module.num_bounces.copy_(0)
elif model_params.max_one_bounce_until_iter > 0:
    raytracer.cuda_module.num_bounces.copy_(min(raytracer.config.MAX_BOUNCES, 1))

for iteration in tqdm(range(first_iter, opt_params.iterations + 1), desc="Training progress"):     
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            net_image_bytes = None
            custom_cam, do_training, pipe_params.convert_SHs_python, pipe_params.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                package = render(viewpoint_cam, raytracer, pipe_params, bg)
                render = package.rgb[-1]
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, model_params.source_path)
            if do_training and ((iteration < int(opt_params.iterations)) or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None

    iter_start.record()
    xyz_lr = gaussians.update_learning_rate(iteration)
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

    bg = torch.rand((3), device="cuda") if opt_params.random_background else background

    # *** run fused forward + backprop
    if raytracer.config.USE_LEVEL_OF_DETAIL and random.random() < model_params.lod_prob_blur_targets:
        if model_params.lod_force_blur_sigma >= 0.0:
            blur_sigma = torch.tensor(model_params.lod_force_blur_sigma, device="cuda")
        else:
            blur_sigma = torch.rand(1, device="cuda")**model_params.lod_schedule_power * scene.max_pixel_blur_sigma
    else:
        blur_sigma = None

    package = render(viewpoint_cam, raytracer, pipe_params, bg, blur_sigma=blur_sigma)

    if opt_params.opacity_reg > 0:
        gaussians._opacity.grad += torch.autograd.grad(args.opacity_reg * torch.abs(gaussians.get_opacity).mean(), gaussians._opacity)[0]
    if opt_params.scale_reg > 0:
        gaussians._scaling.grad += torch.autograd.grad(args.scale_reg * torch.abs(gaussians.get_scaling).mean(), gaussians._scaling)[0]

    with torch.no_grad():
        if opt_params.opacity_decay < 1.0:
            gaussians._opacity.copy_(inverse_sigmoid(gaussians.get_opacity * opt_params.opacity_decay)) 
        if opt_params.scale_decay < 1.0:
            gaussians._scaling.copy_(torch.log(gaussians.get_scaling * opt_params.scale_decay))
        if opt_params.lod_mean_decay < 1.0:
            gaussians._lod_mean.copy_(torch.log(gaussians.get_lod_mean * opt_params.lod_mean_decay))
        if opt_params.lod_scale_decay < 1.0:
            gaussians._lod_scale.copy_(torch.log(gaussians.get_lod_scale * opt_params.lod_scale_decay))

    # todo clamp the min opacities so they don't go under ALPHA_THRESHOLD
    iter_end.record()

    with torch.no_grad():
        # Log and save
        training_report(tb_writer, iteration) 

        if iteration % 1000 == 0 or iteration == 1:
            os.makedirs(os.path.join(args.model_path, "plots"), exist_ok=True)

            if False:
                # Save a histogram of gaussian opacities
                opacities = gaussians.get_opacity.cpu().numpy()
                df = pd.DataFrame(opacities, columns=["opacity"])
                fig = px.histogram(df, x="opacity", nbins=50, title="Histogram of Gaussian Opacities")
                fig.write_image(os.path.join(args.model_path, f"plots/opacity_histogram_{iteration:05d}.png", padding=0))

                # Save a histogram of gaussian _lod_mean
                lod_mean = gaussians.get_lod_mean.cpu().numpy()
                df = pd.DataFrame(lod_mean, columns=["lod_mean"])
                fig = px.histogram(df, x="lod_mean", nbins=50, title="Histogram of Gaussian LOD Mean")
                fig.write_image(os.path.join(args.model_path, f"plots/lod_mean_histogram_{iteration:05d}.png", padding=0))

                # Save a histogram of gaussian _lod_scale
                lod_scale = gaussians.get_lod_scale.cpu().numpy()
                df = pd.DataFrame(lod_scale, columns=["lod_scale"])
                fig = px.histogram(df, x="lod_scale", nbins=50, title="Histogram of Gaussian LOD Scale")
                fig.write_image(os.path.join(args.model_path, f"plots/lod_scale_histogram_{iteration:05d}.png", padding=0))

            if False:
                # Save a scatter plot of gaussian round counter vs lod_mean
                sample_indices = random.sample(range(gaussians._round_counter.shape[0]), int(0.20 * gaussians._round_counter.shape[0]))
                round_counter = gaussians._round_counter[sample_indices].cpu().numpy()
                lod_mean = gaussians.get_lod_mean[sample_indices].cpu().numpy()
                df = pd.DataFrame({ 'round_counter': round_counter[:, 0], 'lod_mean': lod_mean[:, 0] })
                fig = px.scatter(df, x="lod_mean", y="round_counter", title="Scatter Plot of Gaussian Densification Round Counter vs LOD Mean", opacity=0.01)
                fig.write_image(os.path.join(args.model_path, f"plots/round_counter_vs_lod_mean_{iteration:05d}.png", padding=0))

            # Save the elapsed time
            delta = time.time() - start
            with open(os.path.join(args.model_path, "time.txt"), "a") as f:
                minutes, seconds = divmod(int(delta), 60)
                timestamp = f"{minutes:02}:{seconds:02}"
                print("Elapsed time: ", timestamp)
                f.write(f"{iteration:5}: {timestamp}\n")

            # Save the average and std opacity
            with open(os.path.join(args.model_path, "opacity.txt"), "a") as f:
                f.write(f"{iteration:5}: {gaussians.get_opacity.mean().item():.3f} +- {gaussians.get_opacity.std().item():.3f}\n")
            
            # Save the average and std size
            with open(os.path.join(args.model_path, "size.txt"), "a") as f:
                f.write(f"{iteration:5}: {gaussians.get_scaling.mean().item():.3f} +- {gaussians.get_scaling.std().item():.3f}\n")

            # Save the average and std size of the largest axis per gaussian
            with open(os.path.join(args.model_path, "size_axis_max.txt"), "a") as f:
                f.write(f"{iteration:5}: {gaussians.get_scaling.amax(dim=1).mean().item():.5f} +- {gaussians.get_scaling.amax(dim=1).std().item():.5f}\n")

            # same but for the median axis
            with open(os.path.join(args.model_path, "size_axis_median.txt"), "a") as f:
                f.write(f"{iteration:5}: {gaussians.get_scaling.median(dim=1).values.mean().item():.5f} +- {gaussians.get_scaling.median(dim=1).values.std().item():.5f}\n")

            # Same but for the smallest axis
            with open(os.path.join(args.model_path, "size_axis_min.txt"), "a") as f:
                f.write(f"{iteration:5}: {gaussians.get_scaling.amin(dim=1).mean().item():.5f} +- {gaussians.get_scaling.amin(dim=1).std().item():.5f}\n")

            # From raytracer.num_hits, print the mean, max, and std 
            num_hits = raytracer.cuda_module.num_hits_per_pixel.float()
            with open(os.path.join(args.model_path, "num_hits.txt"), "a") as f:
                f.write(f"{iteration:5}: {num_hits.mean().item():.3f} +- {num_hits.std().item():.3f}\n")

            num_traversed = raytracer.cuda_module.num_traversed_per_pixel.float()
            with open(os.path.join(args.model_path, "num_traversed.txt"), "a") as f:
                f.write(f"{iteration:5}: {num_traversed.mean().item():.3f} +- {num_traversed.std().item():.3f}\n")

            with open(os.path.join(args.model_path, "num_gaussians.txt"), "a") as f:
                f.write(f"{iteration:5}: {gaussians.get_xyz.shape[0]}\n")
                print("Number of gaussians: ", gaussians.get_xyz.shape[0])

        if iteration in args.save_iterations:
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        # todo check these again & review position, in this codecase it used to be after the densification step, but i don't recall where it was originally in 3dgs
        if model_params.use_opacity_resets and iteration < opt_params.densify_until_iter: 
            if iteration % opt_params.opacity_reset_interval == 0:
                gaussians.reset_opacity()

        if opt_params.densif_use_top_k and iteration <= opt_params.densify_until_iter:
            gaussians.add_densification_stats_3d(raytracer.cuda_module.densification_gradient_diffuse, raytracer.cuda_module.densification_gradient_glossy)

        if opt_params.densif_use_top_k and (iteration % opt_params.densification_interval == 0 or iteration == 1):
            max_ws_size = scene.cameras_extent * model_params.glossy_bbox_size_mult * model_params.scene_extent_multiplier
            densif_args = (scene, opt_params, model_params.min_opacity, max_ws_size)
            if iteration > opt_params.densify_from_iter:
                if iteration < opt_params.densify_until_iter:
                    #!!!!!!!! review why I'm starting with so many fewer gaussians than the # of sfm points.
                    trace = gaussians.densify_and_prune_top_k(*densif_args)
                    trace = f"Iteration {iteration}; " + trace
                    with open(os.path.join(scene.model_path + "/densification_trace.txt"), "a") as f:
                        f.write(trace)
                else:
                    gaussians.prune_znear_only(scene)
            else:
                gaussians.prune(*densif_args)
            raytracer.rebuild_bvh()
        elif iteration % opt_params.densification_interval == 0:
            if "NO_REBUILD" not in os.environ:
                raytracer.rebuild_bvh()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=False)
        raytracer.zero_grad()

        if raytracer.config.USE_LEVEL_OF_DETAIL:
            with torch.no_grad():
                gaussians._lod_mean.data.clamp_(min=0)
            
            if model_params.lod_clamp_minsize:
                with torch.no_grad(): 
                    gaussians._scaling.data.clamp_(min=torch.log(gaussians._lod_mean.clamp(min=float(os.getenv("LOD_CLAMP_EPS", 0.0))))) #!!!!!!!!!!!!!!! was 1e-8
                if torch.isnan(gaussians._lod_mean).any() or torch.isnan(gaussians._scaling).any():
                    print("NANs in lod_mean or _scaling")
                    quit()

        # Clamp the gaussian scales to min 1% of the longest axis for numerical stability
        # if "SKIP_CLAMP_MINSIZE" not in os.environ:
        #     with torch.no_grad():
        #         scaling = gaussians.get_scaling
        #         max_scaling = scaling.amax(dim=1)
        #         gaussians._scaling.data.clamp_(
        #             min=torch.log(max_scaling * float(os.getenv("MIN_RELATIVE_SCALING", 0.01))).unsqueeze(-1)
        #         )

        if model_params.add_mcmc_noise:
            L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
            actual_covariance = L @ L.transpose(1, 2)

            def op_sigmoid(x, k=100, x0=0.995):
                return 1 / (1 + torch.exp(-k * (x - x0)))
            
            print(opt_params.noise_lr * xyz_lr)
            noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity)) *  xyz_lr
            noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
            gaussians._xyz.add_(noise)

        if False:
            if iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if iteration == model_params.no_bounces_until_iter:
        raytracer.cuda_module.num_bounces.copy_(min(raytracer.config.MAX_BOUNCES, 1))

    if iteration == model_params.max_one_bounce_until_iter and iteration > model_params.no_bounces_until_iter:
        raytracer.cuda_module.num_bounces.copy_(raytracer.config.MAX_BOUNCES)

    if iteration == model_params.rebalance_losses_at_iter:
        os.environ["GLOSSY_LOSS_WEIGHT"] = str(model_params.glossy_loss_weight_after_rebalance)
        os.environ["DIFFUSE_LOSS_WEIGHT"] = str(model_params.diffuse_loss_weight_after_rebalance)
        raytracer.cuda_module.set_losses(True)

    if iteration == model_params.enable_regular_loss_at_iter:
        os.environ["REGULAR_LOSS_WEIGHT"] = "1.0"
        raytracer.cuda_module.set_losses(True)

# All done
print("\nTraining complete.")
