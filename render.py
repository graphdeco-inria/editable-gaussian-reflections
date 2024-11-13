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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, GaussianRaytracer
from gaussian_renderer import render, network_gui, render
import copy
import imageio
import shutil
import math 
import numpy as np 
import torch.nn.functional as F


def render_set(model_params, model_path, split, iteration, views, gaussians, pipeline, background, raytracer, glossy_gaussians=None):
    render_path = os.path.join(model_path, split, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, split, "ours_{}".format(iteration), "gt")
    diffuse_render_path = os.path.join(model_path, split, "ours_{}".format(iteration), "diffuse_renders")
    diffuse_gts_path = os.path.join(model_path, split, "ours_{}".format(iteration), "diffuse_gt")
    glossy_render_path = os.path.join(model_path, split, "ours_{}".format(iteration), "glossy_renders")
    glossy_gts_path = os.path.join(model_path, split, "ours_{}".format(iteration), "glossy_gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(diffuse_render_path, exist_ok=True)
    makedirs(diffuse_gts_path, exist_ok=True)
    makedirs(glossy_render_path, exist_ok=True)
    makedirs(glossy_gts_path, exist_ok=True)

    all_renders = []
    all_gts = []

    all_diffuse_renders = []
    all_diffuse_gts = []

    all_glossy_renders = []
    all_glossy_gts = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if "env" in args.mode:
            if idx == 0:
                view0 = view
                view0.FoVx = 2.0944 * 2 #!!!
                view0.FoVy = -2.0944 * 2 #?? why negative
            view = view0

            R_colmap_init = view.R
            _R_blender = -R_colmap_init
            _R_blender[:, 0] = -_R_blender[:, 0]
            R_blender = _R_blender
            T_blender = -R_colmap_init @ view.T
            
            if "env_rot" in args.mode:
                theta = 2 * math.pi * idx / (len(views) - 1)
                rotation = torch.tensor((
                    (math.cos(theta), -math.sin(theta), 0.0),
                    (math.sin(theta), math.cos(theta), 0.0),
                    (0.0, 0.0, 1.0)
                ))
                if idx > 0:
                    R_blender = rotation.to(torch.float64) @  np.array(((-0.9882196187973022, 0.10767492651939392, -0.10875695198774338),
        (-0.10844696313142776, 0.008747747167944908, 0.9940638542175293),
        (0.10798710584640503, 0.994147777557373, 0.003032323671504855)))
            elif "env_move" in args.mode:
                theta = 0
                rotation = torch.tensor((
                    (math.cos(theta), -math.sin(theta), 0.0),
                    (math.sin(theta), math.cos(theta), 0.0),
                    (0.0, 0.0, 1.0)
                ))
                R_blender = rotation.to(torch.float64) @  np.array(((-0.9882196187973022, 0.10767492651939392, -0.10875695198774338),
                    (-0.10844696313142776, 0.008747747167944908, 0.9940638542175293),
                    (0.10798710584640503, 0.994147777557373, 0.003032323671504855))
                            )

            if args.mode == "env_rot_1":
                T_blender = np.array([0.0, -0.2, 0.2])
            elif args.mode == "env_rot_2":
                T_blender = np.array([1.3, -2.0, 0.0])
            elif args.mode == "env_move_1":
                t = idx / (len(views) - 1)
                T_blender = (1.0 - t) * np.array([0.0, -0.2, 0.2]) + t * np.array([1.3, -2.0, 0.0])
            elif args.mode == "env_move_2":
                t = idx / (len(views) - 1)
                T_blender = (1.0 - t) * np.array([0.0, -0.2, 0.2]) + t * np.array([1.3, -0.3, 0.0])

            R_colmap = -R_blender
            R_colmap[:, 0] = -R_colmap[:, 0]
            T_colmap = -R_colmap.T @ T_blender

            view.R = np.array(R_colmap) 
            view.T = np.array(T_colmap)
            
            view.update()
            print(view.world_view_transform)

        package = render(view, gaussians, pipeline, background,raytracer=raytracer)
        
        diffuse_gt_image = torch.clamp(view.diffuse_image.to("cuda"), 0.0, 1.0)
        glossy_gt_image = torch.clamp(view.glossy_image.to("cuda"), 0.0, 1.0)
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            
        torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(package.diffuse.render, os.path.join(diffuse_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(diffuse_gt_image, os.path.join(diffuse_gts_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(package.glossy.render, os.path.join(glossy_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(glossy_gt_image, os.path.join(glossy_gts_path, '{0:05d}'.format(idx) + ".png"))

        def format_image(image):
            image = F.interpolate(image[None], (image.shape[-2] // 2 * 2, image.shape[-1] // 2 * 2), mode="bilinear")[0]
            return (image.clamp(0, 1) * 255).to(torch.uint8).moveaxis(0, -1).cpu()

        pred_image = torch.clamp(package.diffuse.render + package.glossy.render, 0.0, 1.0)
        torchvision.utils.save_image(pred_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        all_renders.append(format_image(pred_image))
        all_gts.append(format_image(gt_image))
    
        all_diffuse_renders.append(format_image(package.diffuse.render))
        all_diffuse_gts.append(format_image(diffuse_gt_image))

        all_glossy_renders.append(format_image(package.diffuse.render))
        all_glossy_gts.append(format_image(glossy_gt_image))

    os.makedirs(os.path.join(model_params.model_path, "videos/"), exist_ok=True)
    if not args.skip_video:
        print("Writing videos...")
        path = os.path.join(model_params.model_path, f"{{dir}}{split}_{{name}}.mp4")
        
        for label, quality in [("hq", "18"), ("lq", "30")]:
            kwargs = dict(fps=30, options={"crf": quality})

            torchvision.io.write_video(path.format(name=f"renders_{label}", dir="videos/"), torch.stack(all_renders), **kwargs)
            torchvision.io.write_video(path.format(name=f"gts_{label}", dir="videos/"), torch.stack(all_gts), **kwargs)
            torchvision.io.write_video(path.format(name=f"comparison_{label}", dir="videos/"), torch.cat([torch.stack(all_renders), torch.stack(all_gts)], dim=2), **kwargs)
            
            torchvision.io.write_video(path.format(name=f"diffuse_renders_{label}", dir="videos/"), torch.stack(all_diffuse_renders), **kwargs)
            torchvision.io.write_video(path.format(name=f"diffuse_gts_{label}", dir="videos/"), torch.stack(all_diffuse_gts), **kwargs)
            torchvision.io.write_video(path.format(name=f"diffuse_comparison_{label}", dir="videos/"), torch.cat([torch.stack(all_diffuse_renders), torch.stack(all_diffuse_gts)], dim=2), **kwargs)
            
            torchvision.io.write_video(path.format(name=f"glossy_renders_{label}", dir="videos/"), torch.stack(all_glossy_renders), **kwargs)
            torchvision.io.write_video(path.format(name=f"glossy_gts_{label}", dir="videos/"), torch.stack(all_glossy_gts), **kwargs)
            torchvision.io.write_video(path.format(name=f"glossy_comparison_{label}", dir="videos/"), torch.cat([torch.stack(all_glossy_renders), torch.stack(all_glossy_gts)], dim=2), **kwargs)

        if split == "test":
            shutil.copy(
                path.format(name=f"comparison_lq", dir="videos/"), 
                path.format(name="comparison_lq", dir="")
            )
            shutil.copy(
                path.format(name=f"comparison_hq", dir="videos/"), 
                path.format(name="comparison_hq", dir="")
            )

def render_sets(model_params: ModelParams, iteration: int, pipeline: PipelineParams):
    dynModelParams = copy.deepcopy(model_params)
    dynModelParams.dynamic_gaussians = True
    dynModelParams.dynamic_diffuse = True
    dynModelParams.diffuse_only = False
    model_params.diffuse_only = True
        
    with torch.no_grad():
        gaussians = GaussianModel(model_params)
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)

        if args.red_region:
            bbox_min = [0.22, -0.5, -0.22]
            bbox_max = [0.46, -0.13, -0.05]

            mask = (gaussians.get_xyz < torch.tensor(bbox_max, device="cuda")).all(dim=-1).logical_and((gaussians.get_xyz > torch.tensor(bbox_min, device="cuda")).all(dim=-1))
            gaussians._features_dc[mask] = torch.tensor([1.0, 0.0, 0.0], device="cuda")

        bg_color = [0.5, 0.5, 0.5] if args.sliced else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        raytracer = GaussianRaytracer(gaussians, scene.getTrainCameras()[0])

        if args.train_views:
            render_set(model_params, model_params.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, raytracer)
        else:
            render_set(model_params, model_params.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, raytracer) 
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--train_views", action="store_true")
    parser.add_argument("--sliced", action="store_true")
    parser.add_argument("--render_scene", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", type=str, choices=["normal", "env_rot_1", "env_rot_2", "env_move_1", "env_move_2"], default="test")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--red_region", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))