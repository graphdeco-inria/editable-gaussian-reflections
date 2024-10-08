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
from gaussian_renderer import GaussianModel
import copy

def render_set(model_params, model_path, name, iteration, views, gaussians, pipeline, background, dual_gaussians=None):
    if dual_gaussians is None:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if args.render_scene:
                mat_blender = torch.tensor([[-0.9325725436210632, -0.07478659600019455, -0.3531506359577179, -0.0006426088511943817],
                    [-0.35823866724967957, 0.31213051080703735, 0.8799088597297668, -0.11529811471700668],
                    [0.04442369565367699, 0.9470911026000977, -0.31787580251693726, 0.05930082127451897],
                    [0.0, 0.0, 0.0, 1.0]])
                R_blender = mat_blender[0:3, 0:3]
                T_blender = mat_blender[0:3, 3]
                R_colmap = -R_blender
                R_colmap[:, 0] = -R_colmap[:, 0]
                R_colmap = -R_blender.T @ T_blender

                view.R = R_colmap
                view.T = T_colmap
                view.update()

                idx = 999

            rendering = render(view, gaussians, pipeline, background, secondary_view=views[0] if args.fixed_pov else None, nomask=args.render_scene)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

            if args.render_scene:
                break
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        diffuse_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diffuse_renders")
        diffuse_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "diffuse_gt")
        glossy_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "glossy_renders")
        glossy_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "glossy_gt")

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(diffuse_render_path, exist_ok=True)
        makedirs(diffuse_gts_path, exist_ok=True)
        makedirs(glossy_render_path, exist_ok=True)
        makedirs(glossy_gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if not model_params.skip_primal:
                package = render(view, gaussians, pipeline, background, secondary_view=views[0] if args.fixed_pov else None, render_depth=True)
                diffuse_image = torch.clamp(package["render"], 0.0, 1.0)
            glossy_package = render(view, dual_gaussians, pipeline, background, secondary_view=views[0] if args.fixed_pov else None)
            glossy_image = torch.clamp(glossy_package["render"], 0.0, 1.0)
            diffuse_gt_image = torch.clamp(view.diffuse_image.to("cuda"), 0.0, 1.0)
            glossy_gt_image = torch.clamp(view.glossy_image.to("cuda"), 0.0, 1.0)
            gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
                
            torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
            if not model_params.skip_primal:
                torchvision.utils.save_image(diffuse_image, os.path.join(diffuse_render_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(diffuse_gt_image, os.path.join(diffuse_gts_path, '{0:05d}'.format(idx) + ".png"))

            torchvision.utils.save_image(glossy_image, os.path.join(glossy_render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(glossy_gt_image, os.path.join(glossy_gts_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(pred_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # pred_image = torch.clamp((glossy_package["render"]**1.6 + package["render"]**1.6)**(1/1.6), 0.0, 1.0)

def render_sets(model_params: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    if model_params.split_spec_diff:
        dynModelParams = copy.deepcopy(model_params)
        dynModelParams.convert_mlp = True 
        dynModelParams.dynamic_gaussians = True
        dynModelParams.dynamic_diffuse = True
        dynModelParams.diffuse_only = False
        model_params.diffuse_only = True
        
    with torch.no_grad():
        gaussians = GaussianModel(model_params, model_params.sh_degree)
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)

        if model_params.split_spec_diff:
            model_params.diffuse_only = True
            dualModelParams = copy.deepcopy(model_params)
            dualModelParams.dual = True
            dual_gaussians = GaussianModel(dualModelParams, model_params.sh_degree)
            dual_scene = Scene(dualModelParams, dual_gaussians, load_iteration=iteration, dual=True)

            kwargs=dict(dual_gaussians=dual_gaussians)
        else:
            kwargs=dict()

        bg_color = [0.5, 0.5, 0.5] if args.sliced else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(model_params, model_params.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, **kwargs)

        if not skip_test:
            label = "test"
            if args.sliced:
                label += "_sliced"
            if args.fixed_pov:
                label += "_fixed"
            render_set(model_params, model_params.model_path, label, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, **kwargs)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--sliced", action="store_true")
    parser.add_argument("--render_scene", action="store_true")
    parser.add_argument("--fixed_pov", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--flip_cams", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)