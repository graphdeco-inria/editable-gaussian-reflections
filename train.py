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
from scene import Scene, GaussianModel, SurfelModel
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

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def prepare_output_and_logger(args: ModelParams):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", datetime.isoformat(timespec="seconds"))

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, elpased):
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
                    package = render(viewpoint, gaussians, pipe_params, bg, raytracer=raytracer)
                    print(idx, package.glossy.render.amax())
                    os.makedirs(tb_writer.log_dir + "/" + f"{config['name']}_view", exist_ok=True)

                    diffuse_image = torch.clamp(package.diffuse.render, 0.0, 1.0)
                    glossy_image = torch.clamp(package.glossy.render, 0.0, 1.0)
                    diffuse_gt_image = torch.clamp(viewpoint.diffuse_image.to("cuda"), 0.0, 1.0)
                    normal_gt_image = torch.clamp(viewpoint.normal_image.to("cuda") / 2 + 0.5, 0.0, 1.0)
                    glossy_gt_image = torch.clamp(viewpoint.glossy_image.to("cuda"), 0.0, 1.0)
                    pred_image = package.diffuse.render + package.glossy.render

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5): 
                        save_image(torch.stack([diffuse_image, diffuse_gt_image, glossy_image, glossy_gt_image, pred_image, gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}.png", nrow=2)
                    diffuse_l1_test += l1_loss(diffuse_image, diffuse_gt_image).mean().double()
                    diffuse_psnr_test += psnr(diffuse_image, diffuse_gt_image).mean().double()

                    roughness_image = torch.clamp(package.diffuse.roughness, 0.0, 1.0)
                    normal_image = torch.clamp(package.diffuse.normal / 2 + 0.5, 0.0, 1.0) 
                    position_image = torch.clamp(package.diffuse.position, 0.0, 1.0)
                    F0_image = torch.clamp(package.diffuse.F0, 0.0, 1.0)
                    mask_image = torch.clamp(package.glossy.mask, 0.0, 1.0)
                    if model_params.brdf_mode != "disabled":
                        brdf_image = torch.clamp(package.glossy.brdf, 0.0, 1.0)

                    normal_gt_image = torch.clamp(viewpoint.normal_image.to("cuda") / 2 + 0.5, 0.0, 1.0)
                    position_gt_image = torch.clamp(viewpoint.position_image.to("cuda"), 0.0, 1.0)
                    F0_gt_image = torch.clamp(viewpoint.F0_image.to("cuda"), 0.0, 1.0)
                    roughness_gt_image = torch.clamp(viewpoint.roughness_image.to("cuda"), 0.0, 1.0)
                    if model_params.brdf_mode != "disabled":
                        brdf_gt_image = torch.clamp(viewpoint.brdf_image.to("cuda"), 0.0, 1.0)
                        
                    if tb_writer and (idx < 5): 
                        save_image(torch.stack([roughness_image.cuda(), roughness_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_roughness.png", nrow=2, padding=0)
                        save_image(torch.stack([F0_image.cuda(), F0_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_F0.png", nrow=2, padding=0)
                        save_image(torch.stack([pred_image, gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_final.png", nrow=2, padding=0)
                        save_image(torch.stack([diffuse_image, diffuse_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_diffuse.png", nrow=2, padding=0)
                        save_image(torch.stack([glossy_image, glossy_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_glossy.png", nrow=2, padding=0)
                        save_image(torch.stack([position_image.cuda(), position_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_position.png", nrow=2, padding=0)
                        save_image(torch.stack([normal_image.cuda(), normal_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_normal.png", nrow=2, padding=0)
                        save_image(torch.stack([mask_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_mask.png", nrow=2, padding=0)
                        if model_params.brdf_mode != "disabled":
                            save_image(torch.stack([brdf_image, brdf_gt_image.cuda()]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_brdf.png", nrow=2, padding=0)

                    glossy_l1_test += l1_loss(glossy_image, glossy_gt_image).mean().double()
                    glossy_psnr_test += psnr(glossy_image, glossy_gt_image).mean().double()
                    l1_test += l1_loss(pred_image, gt_image).mean().double()
                    psnr_test += psnr(pred_image, gt_image).mean().double()

                if model_params.brdf_mode == "static_lut":
                    save_image(torch.stack([ gaussians.get_brdf_lut ]).abs(), os.path.join(tb_writer.log_dir, f"{config['name']}_view/lut_iter_{iteration:09}.png"), nrow=1)
                elif model_params.brdf_mode == "finetuned_lut":
                    
                    save_image(torch.stack([ gaussians._brdf_lut, gaussians.get_brdf_lut ]), os.path.join(tb_writer.log_dir, f"{config['name']}_view/lut_iter_{iteration:09}.png"), nrow=1)
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
                    f.write(f"{iteration}, {diffuse_psnr_test:02f}, {glossy_psnr_test:02f}, {psnr_test:02f}\n")

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
parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 100, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 30_000, 60_000, 90_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 1_000, 2_500, 7_000, 15_000, 30_000, 60_000, 90_000])
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

print("Optimizing " + args.model_path)

# Initialize system state (RNG)
safe_state(args.quiet)

# Start GUI server, configure and run training
if args.viewer:
    network_gui.init(args.ip, args.port)

first_iter = 0
tb_writer = prepare_output_and_logger(model_params) 

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

ema_loss_for_log = 0.0
first_iter += 1

for iteration in tqdm(range(first_iter, opt_params.iterations + 1), desc="Training progress"):      
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            net_image_bytes = None
            custom_cam, do_training, pipe_params.convert_SHs_python, pipe_params.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                net_image = render(custom_cam, gaussians, pipe_params, background, scaling_modifer)["render"] 
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, model_params.source_path)
            if do_training and ((iteration < int(opt_params.iterations)) or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None

    iter_start.record()
    gaussians.update_learning_rate(iteration)

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

    bg = torch.rand((3), device="cuda") if opt_params.random_background else background

    # *** run fused forward + backprop
    render(viewpoint_cam, gaussians, pipe_params, bg, raytracer=raytracer, iteration=iteration) 
    
    # if model_params.num_warmup_iters > 0 and iteration == model_params.num_warmup_iters:
    #     raytracer = GaussianRaytracer(gaussians, viewpoint_stack[0])

    # todo clamp the min opacities so they don't go under ALPHA_THRESHOLD

    iter_end.record()

    with torch.no_grad():
        # Log and save
        training_report(tb_writer, iteration, 0.0) #! buggy iter_start.elapsed_time(iter_end), RuntimeError: CUDA error: device not ready
        if iteration in args.save_iterations:
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        # Densification
        if False:
            if iteration < opt_params.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > opt_params.densify_from_iter and iteration % opt_params.densification_interval == 0:
                    size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt_params.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    # glossy_gaussians.densify_and_prune_glossy(opt_params.densify_grad_threshold, 0.005, scene.cameras_extent)
                    raytracer.rebuild_bvh()
                
                if iteration % opt_params.opacity_reset_interval == 0 or (model_params.white_background and iteration == opt_params.densify_from_iter):
                    gaussians.reset_opacity() 

        # Optimizer step
        if iteration < opt_params.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=False) # todo not sure if this set_to_none=False is still required
            raytracer.zero_grad()

        if iteration in args.checkpoint_iterations:
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


# All done
print("\nTraining complete.")

os.system(f"python render.py --start_checkpoint {scene.model_path}/chkpnt{iteration}.pth " + " ".join(sys.argv[1:]))
