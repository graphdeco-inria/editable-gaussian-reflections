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

def render_set(modelParams, model_path, name, iteration, views, gaussians, pipeline, background, dyn_gaussians=None):
    if dyn_gaussians is None:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background, secondary_view=views[0] if args.fixed_pov else None)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
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
            package = render(view, gaussians, pipeline, background, secondary_view=views[0] if args.fixed_pov else None, render_depth=True)
            diffuse_image = torch.clamp(package["render"], 0.0, 1.0)
            glossy_package = render(view, dyn_gaussians, pipeline, background, secondary_view=views[0] if args.fixed_pov else None)
            glossy_image = torch.clamp(glossy_package["render"], 0.0, 1.0)
            diffuse_gt_image = torch.clamp(view.diffuse_image.to("cuda"), 0.0, 1.0)
            glossy_gt_image = torch.clamp(view.glossy_image.to("cuda"), 0.0, 1.0)
            pred_image = torch.clamp((glossy_package["render"]**1.6 + package["render"]**1.6)**(1/1.6), 0.0, 1.0)
            gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
                
            torchvision.utils.save_image(pred_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
            torchvision.utils.save_image(diffuse_image, os.path.join(diffuse_render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(diffuse_gt_image, os.path.join(diffuse_gts_path, '{0:05d}'.format(idx) + ".png"))

            torchvision.utils.save_image(glossy_image, os.path.join(glossy_render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(glossy_gt_image, os.path.join(glossy_gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(modelParams: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    if modelParams.split_spec_diff:
        dynModelParams = copy.deepcopy(modelParams)
        dynModelParams.convert_mlp = True 
        dynModelParams.dynamic_gaussians = True
        dynModelParams.dynamic_diffuse = True
        dynModelParams.diffuse_only = False
        modelParams.diffuse_only = True
        
    with torch.no_grad():
        gaussians = GaussianModel(modelParams, modelParams.sh_degree)
        scene = Scene(modelParams, gaussians, load_iteration=iteration, shuffle=False)

        def slice(gaussians):
            slice_size = 0.1
            mask = (gaussians._xyz[:, 1] < slice_size) & (gaussians._xyz[:, 1] > -slice_size)
            gaussians._xyz = gaussians._xyz[mask]
            gaussians._features_dc = gaussians._features_dc[mask]
            gaussians._features_rest = gaussians._features_rest[mask]
            gaussians._scaling = gaussians._scaling[mask]
            gaussians._rotation = gaussians._rotation[mask]
            gaussians._opacity = gaussians._opacity[mask]
        
        if args.sliced:
            slice(gaussians)

        if modelParams.split_spec_diff:
            dyn_gaussians = GaussianModel(dynModelParams, modelParams.sh_degree)
            dyn_scene = Scene(dynModelParams, dyn_gaussians, dynamic=True, load_iteration=iteration, shuffle=False)
            if args.sliced:
                slice(dyn_gaussians)
            kwargs=dict(dyn_gaussians=dyn_gaussians)
        else:
            kwargs=dict()

        bg_color = [0.5,0.5,0.5] if args.sliced else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(modelParams, modelParams.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, **kwargs)

        if not skip_test:
            label = "test"
            if args.sliced:
                label += "_sliced"
            if args.fixed_pov:
                label += "_fixed"
            render_set(modelParams, modelParams.model_path, label, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, **kwargs)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--sliced", action="store_true")
    parser.add_argument("--fixed_pov", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)