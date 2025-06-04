import glob 
import tyro
import os
from PIL import Image 
from dataclasses import dataclass
from tqdm import tqdm 
from typing import *
import json
from torchvision.utils import save_image
import numpy as np
import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_tensor


@dataclass
class Conf:
    scenes: List[int] = tuple([
        # "multichromeball_identical_kitchen_v2", 
        # "multichromeball_kitchen_v2", 
        # "multichromeball_tint_kitchen_v2", 
        # "multichromeball_value_kitchen_v2",
        "shiny_bedroom", 
        "shiny_kitchen", 
        "shiny_livingroom", 
        "shiny_office"
    ])
    methods: List[int] = tuple([
        "3dgs",
        "2dgs",
        "gaussian_shader", 
        "3dgs_dr", 
        "ref_gaussian", 
        # "priors",
        "envgs_network",
        "envgs_gt",
        "ours_network",
        "ours",
    ])
    render_passes: List[int] = tuple([
        # "normal", 
        "diffuse", 
        "glossy",
        "render", 
    ])

    ground_truth_pattern: str = "gts/{scene}/test/tonemapped_{render_pass}/render_{i:04d}.png"
    predictions_path_pattern = {
        "3dgs": "3dgs_results_v2/{scene}/test/ours_30000/{render_pass}/{i:05d}.png",
        "2dgs": "2dgs_results_v2/{scene}/test/ours_30000/{render_pass}/{i:05d}.png",
        "gaussian_shader": "gaussian_shader_results_v2/{scene}/test/ours_30000/{render_pass}/{i:05d}.png",
        "3dgs_dr": "3dgs_dr_results_v2/{scene}/test/ours_97000/renders/{render_pass}/{i:05d}.png", 
        "ref_gaussian": "ref_gaussian_results_v2/{scene}/test/renders/{render_pass}/{i:05d}.png",
        
        "envgs_network" : "envgs/renders_prnormals/envgs_{scene}/{render_pass}/frame0000_camera{i:04d}.png",
        "envgs_gt" : "envgs/renders_gtnormals/envgs_{scene}/{render_pass}/frame0000_camera{i:04d}.png",

        "priors" : "real_datasets_v3_filmic/{scene}/test/{render_pass}/render_{i:04d}.png",

        "ours_network" : "ours_results_from_priors_final_v1/{scene}/test/ours_8000/{render_pass}/{i:05d}_{render_pass}.png",
        
        "ours" : "our_results_v4/{scene}/test/ours_8000/{render_pass}/{i:05d}_{render_pass}.png",
    }
    render_pass_naming_schemes = {
        "3dgs": {
            "render": "renders",
            "diffuse": "diffuse",
            "glossy": "residual",
            "normal": "normal",
        },
        "2dgs": {
            "render": "renders",
            "diffuse": "diffuse",
            "glossy": "residual",
            "normal": "normals",
        },
        "r3dg": {
            "render": "renders",
            "diffuse": "diffuse",
            "glossy": "specular",
            "normal": "normal",
        },
        "gaussian_shader": {
            "render": "renders",
            "diffuse": "diffuse_color",
            "glossy": "residual",
            "normal": "normal",
        },
        "3dgs_dr": {
            "render": "rgb",
            "diffuse": "diffuse",
            "glossy": "glossy",
            "normal": "normal",
        },
        "ref_gaussian": {
            "render": "rgb",
            "diffuse": "diffuse",
            "glossy": "glossy",
            "normal": "normal",
        },
        "envgs_network": {
            "render": "RENDER",
            "diffuse": "DIFFUSE",
            "glossy": "CUSTOM",
            "normal": "NORMAL",
        },
        "envgs_gt": {
            "render": "RENDER",
            "diffuse": "DIFFUSE",
            "glossy": "CUSTOM",
            "normal": "NORMAL",
        },
        "ours_network": {
            "render": "render",
            "diffuse": "diffuse",
            "glossy": "glossy",
            "normal": "normal",
        },
        "ours": {
            "render": "render",
            "diffuse": "diffuse",
            "glossy": "glossy",
            "normal": "normal",
        },
        "priors": {
            "render": "image",
            "diffuse": "diffuse",
            "glossy": "glossy",
            "normal": "normal",
        }
    }

    num_frames: int = 100
    allow_resize: bool = False

    image_height: Optional[int] = 768


if __name__ == "__main__":
    conf = tyro.cli(Conf)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = dict(
        psnr=PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(device), 
        lpips=LearnedPerceptualImagePatchSimilarity(normalize=True).to(device),
        ssim=StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    )

    base_path = os.path.dirname(os.path.abspath(__file__))

    scores_by_scene = {}

    for scene in conf.scenes:
        scores = {}
        for method in conf.methods:
            scores[method] = { render_pass: { key: 0.0 for key in metrics.keys() } for render_pass in conf.render_passes }
            
            for i in tqdm(range(conf.num_frames)):
                images = {}

                for render_pass in conf.render_passes:
                    if method in ["3dgs"] and render_pass == "normal":
                        for metric, metric_fn in metrics.items():
                            scores[method][render_pass][metric] = float("nan")
                        continue

                    gt_path = base_path + "/" + conf.ground_truth_pattern.format(scene=scene, i=i, render_pass=render_pass) 
                    pred_path = base_path + "/" + conf.predictions_path_pattern[method].format(scene=scene, i=i, render_pass=conf.render_pass_naming_schemes[method][render_pass])
                    
                    gt = Image.open(gt_path).convert("RGB")
                    pred = Image.open(pred_path).convert("RGB")

                    if conf.image_height is not None:
                        if gt.size[1] != conf.image_height:
                            gt = gt.resize((int(conf.image_height*1.5), conf.image_height), Image.LANCZOS)
                        if conf.allow_resize:
                            pred = pred.resize((int(conf.image_height*1.5), conf.image_height), Image.LANCZOS)
                        else:
                            assert pred.size[1] == conf.image_height, method

                    pred = to_tensor(pred)
                    gt = to_tensor(gt)

                    images[render_pass] = (pred, gt)

                for render_pass in conf.render_passes:
                    for metric, metric_fn in metrics.items():
                        scores[method][render_pass][metric] += metric_fn(images[render_pass][0][None].to(device), images[render_pass][1][None].to(device)).item() / conf.num_frames

        scores_by_scene[scene] = scores
        with open(base_path + f"/scores.json", "w") as f:
            json.dump(scores_by_scene, f, indent=4)

