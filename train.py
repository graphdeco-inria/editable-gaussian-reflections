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
from gaussian_renderer import render, network_gui
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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(model_params: ModelParams, optParams: OptimizationParams, pipeParams: PipelineParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(model_params) # todo: make sure cfg_args is ok

    if model_params.split_spec_diff:
        model_params.diffuse_only = True
        dualModelParams = copy.deepcopy(model_params)
        dualModelParams.dual = True

        dual_gaussians = GaussianModel(dualModelParams, model_params.sh_degree)
        dual_scene = Scene(dualModelParams, dual_gaussians, dual=True)
    else:
        dual_gaussians = None
        dual_scene = None #! this causes double loading of the images

    if not args.skip_primal:
        # gaussians = GaussianModel(model_params, model_params.sh_degree)
        gaussians = SurfelModel(model_params, 0)
        scene = Scene(model_params, gaussians)
        gaussians.training_setup(optParams)

    if model_params.split_spec_diff:
        dual_gaussians.training_setup(optParams)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, optParams)
        dual_gaussians.restore(model_params, optParams, dual=True)

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
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
                    # todo send dual gaussians pass
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, model_params.source_path)
                if do_training and ((iteration < int(optParams.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        if not args.skip_primal:
            gaussians.update_learning_rate(iteration)
        if model_params.split_spec_diff:
            dual_gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            if args.skip_primal:
                viewpoint_stack = dual_scene.getTrainCameras().copy()
            else:
                viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipeParams.debug = True

        bg = torch.rand((3), device="cuda") if optParams.random_background else background

        if model_params.skip_primal:
            image = viewpoint_cam.diffuse_image.cuda()
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipeParams, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        if args.split_spec_diff:
            dual_render_pkg = render(viewpoint_cam, dual_gaussians, pipeParams, bg)
            dual_image, dual_viewspace_point_tensor = dual_render_pkg["render"], dual_render_pkg["viewspace_points"]
            #, dual_visibility_filter, dual_radii
            #, dual_render_pkg["visibility_filter"], dual_render_pkg["radii"]

        if args.split_spec_diff:
            gt_glossy_image = viewpoint_cam.glossy_image.cuda()
            Ll1_glossy = l1_loss(dual_image, gt_glossy_image)
            loss_glossy = (1.0 - optParams.lambda_dssim) * Ll1_glossy + optParams.lambda_dssim * (1.0 - ssim(dual_image, gt_glossy_image))

            if model_params.skip_primal:
                Ll1_diffuse = 0.0
                loss_diffuse = 0.0
            else:
                gt_diffuse_image = viewpoint_cam.diffuse_image.cuda()
                Ll1_diffuse = l1_loss(image, gt_diffuse_image)
                loss_diffuse = (1.0 - optParams.lambda_dssim) * Ll1_diffuse + optParams.lambda_dssim * (1.0 - ssim(image, gt_diffuse_image))
            
            Ll1 = Ll1_diffuse + Ll1_glossy

            
            loss = loss_diffuse + loss_glossy
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - optParams.lambda_dssim) * Ll1 + optParams.lambda_dssim * (1.0 - ssim(image, gt_image))

        if not args.skip_primal:
            # regularization
            lambda_normal = optParams.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = optParams.lambda_dist if iteration > 3000 else 0.0

            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            # loss
            total_loss = loss + dist_loss + normal_loss

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

            # if tb_writer is not None:
            #     tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
            #     tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, None if args.skip_primal else scene, dual_scene, render, (pipeParams, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                if not args.skip_primal:
                    scene.save(iteration)
                if args.split_spec_diff:
                    dual_scene.save(iteration)

            # Densification
            if iteration < optParams.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                if not args.skip_primal:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    
                # if args.split_spec_diff: 
                #     dual_gaussians.max_radii2D[dual_visibility_filter] = torch.max(dual_gaussians.max_radii2D[dual_visibility_filter], dual_radii[dual_visibility_filter])
                #     dual_gaussians.add_densification_stats(dual_viewspace_point_tensor, dual_visibility_filter)

                if iteration > optParams.densify_from_iter and iteration % optParams.densification_interval == 0 and not args.skip_primal:
                    size_threshold = 20 if iteration > optParams.opacity_reset_interval else None
                    gaussians.densify_and_prune(optParams.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    if args.split_spec_diff and args.densify_dual:  
                        dual_gaussians.densify_without_pruning(optParams.densify_grad_threshold, 0.005, scene.cameras_extent)
                        print("Number of gaussians: ", dual_gaussians.get_xyz.shape[0])
                
                if iteration % optParams.opacity_reset_interval == 0 or (model_params.white_background and iteration == optParams.densify_from_iter):
                    if not args.skip_primal:
                        gaussians.reset_opacity() 
                    if args.split_spec_diff: 
                        dual_gaussians.reset_opacity()

            # Optimizer step
            if iteration < optParams.iterations:
                if not args.skip_primal:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if args.split_spec_diff:
                    dual_gaussians.optimizer.step()
                    dual_gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if args.split_spec_diff:
                    torch.save((dual_gaussians.capture(), iteration), scene.model_path + "/chkpnt_dual" + str(iteration) + ".pth")

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, dual_scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if dual_scene.gaussians.model_params.split_spec_diff:
        # Report test and samples of training set
        if iteration in testing_iterations:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras' : dual_scene.getTestCameras()}, 
                                {'name': 'train', 'cameras' : [dual_scene.getTrainCameras()[idx % len(dual_scene.getTrainCameras())] for idx in range(5, 30, 5)]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    
                    glossy_l1_test = 0.0
                    glossy_psnr_test = 0.0
                    
                    diffuse_l1_test = 0.0
                    diffuse_psnr_test = 0.0

                    for idx, viewpoint in enumerate(config['cameras']):
                        if not args.skip_primal:
                            package = renderFunc(viewpoint, scene.gaussians, *renderArgs, render_depth=True)

                            #2DGS code below
                            from utils.general_utils import colormap
                            render_pkg = package
                            depth = render_pkg["surf_depth"]
                            norm = depth.max()
                            depth = depth / norm
                            depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        # tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        # image = diffuse_image
                        # tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        os.makedirs(tb_writer.log_dir + "/" + f"{config['name']}_view", exist_ok=True)

                        # our code below
                        if not args.skip_primal:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)
                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)

                            diffuse_image = torch.clamp(package["render"], 0.0, 1.0)
                            glossy_package = renderFunc(viewpoint, dual_scene.gaussians, *renderArgs)
                            glossy_image = torch.clamp(glossy_package["render"], 0.0, 1.0)
                            diffuse_gt_image = torch.clamp(viewpoint.diffuse_image.to("cuda"), 0.0, 1.0)
                            normal_gt_image = torch.clamp(viewpoint.normal_image.to("cuda") / 2 + 0.5, 0.0, 1.0)
                            glossy_gt_image = torch.clamp(viewpoint.glossy_image.to("cuda"), 0.0, 1.0)
                            pred_image = glossy_package["render"] + package["render"]

                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            if tb_writer and (idx < 5): 
                                save_image(torch.stack([diffuse_image, diffuse_gt_image, glossy_image, glossy_gt_image, pred_image, gt_image, rend_normal, normal_gt_image, surf_normal, normal_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}.png", nrow=2)
                            diffuse_l1_test += l1_loss(diffuse_image, diffuse_gt_image).mean().double()
                            diffuse_psnr_test += psnr(diffuse_image, diffuse_gt_image).mean().double()


                        else:
                            glossy_package = renderFunc(viewpoint, dual_scene.gaussians, *renderArgs)
                            glossy_image = torch.clamp(glossy_package["render"], 0.0, 1.0)
                            diffuse_gt_image = torch.clamp(viewpoint.diffuse_image.to("cuda"), 0.0, 1.0)
                            glossy_gt_image = torch.clamp(viewpoint.glossy_image.to("cuda"), 0.0, 1.0)

                            pred_image = glossy_image
                            gt_image = glossy_gt_image

                        roughness_image = torch.clamp(glossy_package["roughness"], 0.0, 1.0)
                        normal_image = torch.clamp(glossy_package["normal"] / 2 + 0.5, 0.0, 1.0) 
                        reflectivity_image = torch.clamp(glossy_package["reflectivity"], 0.0, 1.0)

                        roughness_gt_image = torch.clamp(viewpoint.roughness_image.to("cuda"), 0.0, 1.0)
                        albedo_gt_image = torch.clamp(viewpoint.albedo_image.to("cuda"), 0.0, 1.0)
                        normal_gt_image = torch.clamp(viewpoint.normal_image.to("cuda") / 2 + 0.5, 0.0, 1.0)
                        metalness_gt_image = torch.clamp(viewpoint.metalness_image.to("cuda"), 0.0, 1.0)
                        reflectivity_gt_image = (1.0 - metalness_gt_image) * 0.04 + metalness_gt_image * albedo_gt_image
                        if tb_writer and (idx < 5): 
                            save_image(torch.stack([roughness_image.cuda(), roughness_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_roughness.png", nrow=2)
                            save_image(torch.stack([normal_image.cuda(), normal_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_normal.png", nrow=2)
                            save_image(torch.stack([reflectivity_image.cuda(), reflectivity_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}_reflectivity.png", nrow=2)
                            save_image(torch.stack([glossy_image, glossy_gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}.png", nrow=2)
                        glossy_l1_test += l1_loss(glossy_image, glossy_gt_image).mean().double()
                        glossy_psnr_test += psnr(glossy_image, glossy_gt_image).mean().double()
                        l1_test += l1_loss(pred_image, gt_image).mean().double()
                        psnr_test += psnr(pred_image, gt_image).mean().double()

                    
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

            # if tb_writer:
            #     # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

            torch.cuda.empty_cache()
    else:
        assert False # code below wasnt adapted to 2dgs properly
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
                            depth = render_pkg["surf_depth"]
                            norm = depth.max()
                            depth = depth / norm
                            depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                            tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                            try:
                                rend_alpha = render_pkg['rend_alpha']
                                rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                                surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                                tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                                rend_dist = render_pkg["rend_dist"]
                                rend_dist = colormap(rend_dist.cpu().numpy()[0])
                                tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                            except:
                                pass
                                os.makedirs(tb_writer.log_dir + "/" + f"{config['name']}_view", exist_ok=True)
                                os.makedirs(tb_writer.log_dir + "/" + f"{config['name']}_view_depth", exist_ok=True)

                                save_image(torch.stack([image, gt_image]), tb_writer.log_dir + "/" + f"{config['name']}_view/iter_{iteration:09}_{idx}.png")
                                save_image(depth_image, tb_writer.log_dir + "/" + f"{config['name']}_view_depth/depth_iter_{iteration:09d}_view_{idx}.png")
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
    parser.add_argument('--flip_camera', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 30_000, 60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 1_000, 7_000, 15_000, 30_000])
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
