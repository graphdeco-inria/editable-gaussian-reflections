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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(modelParams: ModelParams, optParams: OptimizationParams, pipeParams: PipelineParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    if modelParams.split_spec_diff:
        dynModelParams = copy.deepcopy(modelParams)
        dynModelParams.convert_mlp = True 
        dynModelParams.dynamic_gaussians = True
        dynModelParams.dynamic_diffuse = True
        modelParams.diffuse_only = True

        dyn_gaussians = GaussianModel(dynModelParams, modelParams.sh_degree)
        dyn_scene = Scene(dynModelParams, dyn_gaussians, dynamic=True)
        dyn_scene.model_path = modelParams.model_path
    else:
        dyn_gaussians = None
        dyn_scene = None #!!! this causes double loading of the images

    tb_writer = prepare_output_and_logger(modelParams) # todo: make sure cfg_args is ok

    gaussians = GaussianModel(modelParams, modelParams.sh_degree)
    scene = Scene(modelParams, gaussians)
    gaussians.training_setup(optParams)

    if modelParams.split_spec_diff:
        dyn_gaussians.training_setup(optParams)

    if modelParams.split_spec_diff:
        dyn_scene.model_path = scene.model_path

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, optParams)
        dyn_gaussians.restore(model_params, optParams, dynamic=True)

    bg_color = [1, 1, 1] if modelParams.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, optParams.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, optParams.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipeParams.convert_SHs_python, pipeParams.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipeParams, background, scaling_modifer)["render"] 
                    # todo dyn gaussians
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, modelParams.source_path)
                if do_training and ((iteration < int(optParams.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if modelParams.split_spec_diff:
            dyn_gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            if modelParams.split_spec_diff:
                dyn_gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipeParams.debug = True

        bg = torch.rand((3), device="cuda") if optParams.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipeParams, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        if args.split_spec_diff:
            dyn_render_pkg = render(viewpoint_cam, dyn_gaussians, pipeParams, bg)
            dyn_image, dyn_viewspace_point_tensor, dyn_visibility_filter, dyn_radii = dyn_render_pkg["render"], dyn_render_pkg["viewspace_points"], dyn_render_pkg["visibility_filter"], dyn_render_pkg["radii"]

        # Loss
        if args.split_spec_diff:
            gt_diffuse_image = viewpoint_cam.diffuse_image.cuda()
            Ll1_diffuse = l1_loss(image, gt_diffuse_image)
            loss_diffuse = (1.0 - optParams.lambda_dssim) * Ll1_diffuse + optParams.lambda_dssim * (1.0 - ssim(image, gt_diffuse_image))

            gt_glossy_image = viewpoint_cam.glossy_image.cuda()
            Ll1_glossy = l1_loss(dyn_image, gt_glossy_image)
            loss_glossy = (1.0 - optParams.lambda_dssim) * Ll1_glossy + optParams.lambda_dssim * (1.0 - ssim(dyn_image, gt_glossy_image))

            Ll1 = Ll1_diffuse + Ll1_glossy
            loss = loss_diffuse + loss_glossy # todo: logging
            loss.backward()
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - optParams.lambda_dssim) * Ll1 + optParams.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == optParams.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, dyn_scene, render, (pipeParams, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if args.split_spec_diff:
                    dyn_scene.save(iteration)

            # Densification
            if iteration < optParams.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if args.split_spec_diff: 
                    dyn_gaussians.max_radii2D[dyn_visibility_filter] = torch.max(dyn_gaussians.max_radii2D[dyn_visibility_filter], dyn_radii[dyn_visibility_filter])
                    dyn_gaussians.add_densification_stats(dyn_viewspace_point_tensor, dyn_visibility_filter)

                if iteration > optParams.densify_from_iter and iteration % optParams.densification_interval == 0:
                    size_threshold = 20 if iteration > optParams.opacity_reset_interval else None
                    gaussians.densify_and_prune(optParams.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    #!!!!!!!
                    # if args.split_spec_diff:  
                    #     dyn_gaussians.densify_and_prune(optParams.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % optParams.opacity_reset_interval == 0 or (modelParams.white_background and iteration == optParams.densify_from_iter):
                    gaussians.reset_opacity() 
                    if args.split_spec_diff: 
                        dyn_gaussians.reset_opacity()

            # Optimizer step
            if iteration < optParams.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if args.split_spec_diff:
                    dyn_gaussians.optimizer.step()
                    dyn_gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if args.split_spec_diff:
                    torch.save((dyn_gaussians.capture(), iteration), scene.model_path + "/chkpnt_dyn" + str(iteration) + ".pth")

def prepare_output_and_logger(args: ModelParams):    
    if not args.model_path:
        unique_str = args.label
    args.model_path = os.path.join("./output/", unique_str)

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, dyn_scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if scene.gaussians.modelParams.split_spec_diff:
        # Report test and samples of training set
        if iteration in testing_iterations:
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
                        package = renderFunc(viewpoint, scene.gaussians, *renderArgs, render_depth=True)
                        diffuse_image = torch.clamp(package["render"], 0.0, 1.0)
                        depth_image = package["depth"]
                        glossy_package = renderFunc(viewpoint, dyn_scene.gaussians, *renderArgs)
                        glossy_image = torch.clamp(glossy_package["render"], 0.0, 1.0)
                        diffuse_gt_image = torch.clamp(viewpoint.diffuse_image.to("cuda"), 0.0, 1.0)
                        glossy_gt_image = torch.clamp(viewpoint.glossy_image.to("cuda"), 0.0, 1.0)
                        pred_image = torch.clamp((glossy_package["render"]**1.6 + package["render"]**1.6)**(1/1.6), 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 5): 
                            os.makedirs(tb_writer.log_dir + "/" + f"{config['name']}_view", exist_ok=True)
                            os.makedirs(tb_writer.log_dir + "/" + f"{config['name']}_view_depth", exist_ok=True)

                            save_image(torch.stack([diffuse_image, diffuse_gt_image, glossy_image, glossy_gt_image, pred_image, gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}.jpg", nrow=2)
                            save_image(depth_image, tb_writer.log_dir + "/" + f"{config['name']}_view_depth/depth_iter_{iteration:09d}_view_{idx}.jpg")
                        
                        l1_test += l1_loss(pred_image, gt_image).mean().double()
                        psnr_test += psnr(pred_image, gt_image).mean().double()

                        diffuse_l1_test += l1_loss(diffuse_image, diffuse_gt_image).mean().double()
                        diffuse_psnr_test += psnr(diffuse_image, diffuse_gt_image).mean().double()

                        glossy_l1_test += l1_loss(glossy_image, glossy_gt_image).mean().double()
                        glossy_psnr_test += psnr(glossy_image, glossy_gt_image).mean().double()
                    
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
                # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

            torch.cuda.empty_cache()
    else:
        # Report test and samples of training set
        if iteration in testing_iterations:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        package = renderFunc(viewpoint, scene.gaussians, *renderArgs, render_depth=True)
                        image = torch.clamp(package["render"], 0.0, 1.0)
                        depth_image = package["depth"]
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 5): 
                            os.makedirs(tb_writer.log_dir + "/" + f"{config['name']}_view", exist_ok=True)
                            os.makedirs(tb_writer.log_dir + "/" + f"{config['name']}_view_depth", exist_ok=True)

                            save_image(torch.stack([image, gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}.jpg")
                            save_image(depth_image, tb_writer.log_dir + "/" + f"{config['name']}_view_depth/depth_iter_{iteration:09d}_view_{idx}.jpg")
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])          
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    if tb_writer:
                        os.makedirs(tb_writer.log_dir + "/" + f"{config['name']}_view", exist_ok=True)

                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

            if tb_writer:
                tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 1_000, 5_000, 10_000, 20_000, 30_000, 60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
