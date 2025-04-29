import glob 

import tyro
import os
from PIL import Image 

from dataclasses import dataclass
from tqdm import tqdm 
from typing import *
import json

@dataclass
class Conf:
    scenes: List[int] = tuple([
        "multichromeball_identical_kitchen_v2", 
        "multichromeball_kitchen_v2", 
        "multichromeball_tint_kitchen_v2", 
        "multichromeball_value_kitchen_v2",
        "shiny_bedroom", 
        "shiny_kitchen", 
        "shiny_livingroom", 
        "shiny_office"

    ])
    methods: List[int] = tuple([
        # "3dgs",
        # "2dgs",
        # "r3dg", 
        # "gaussian_shader", 
        # "3dgs_dr", 
        # "ref_gaussian", 
        "ours",
        # "ours1m",
    ])
    render_passes: List[int] = tuple([
        # "normal", 
        "diffuse", 
        "glossy",
        "render", 
    ])

    ground_truth_pattern: str = "gts/{scene}/test/tonemapped_{render_pass}/render_{i:04d}.png"
    predictions_path_pattern = {
        "3dgs": "3dgs_results/{scene}/test/ours_30000/{render_pass}/{i:05d}.png",
        "2dgs": "2dgs_results/{scene}/test/ours_30000/{render_pass}/{i:05d}.png",
        "r3dg" : "r3dg_results/ours/{scene}/neilf/test/ours_50000/{render_pass}/{i:05d}.png",
        "gaussian_shader": "gaussian_shader_results/{scene}/test/ours_30000/{render_pass}/{i:05d}.png",
        "3dgs_dr": "3dgs_dr_results/{scene}/test/ours_97000/renders/{render_pass}/{i:05d}.png", 
        "ref_gaussian": "ref_gaussian_results/{scene}_config_A/test/renders/{render_pass}/{i:05d}.png",
        # "ours" : "our_results/{scene}/test/ours_30000/{render_pass}/{i:05d}_{render_pass}.png", 
        "ours" : "ours/{scene}/test/ours_6000/{render_pass}/{i:05d}_{render_pass}.png"
    }
    render_pass_naming_schemes = {
        # "3dgs": {
        #     "render": "renders",
        #     "diffuse": "diffuse",
        #     "glossy": "residual",
        #     "normal": "normal",
        # },
        # "2dgs": {
        #     "render": "renders",
        #     "diffuse": "diffuse",
        #     "glossy": "residual",
        #     "normal": "normals",
        # },
        # "r3dg": {
        #     "render": "renders",
        #     "diffuse": "diffuse",
        #     "glossy": "specular",
        #     "normal": "normal",
        # },
        # "gaussian_shader": {
        #     "render": "renders",
        #     "diffuse": "diffuse_color",
        #     "glossy": "specular_color",
        #     "normal": "normal",
        # },
        # "3dgs_dr": {
        #     "render": "rgb",
        #     "diffuse": "diffuse",
        #     "glossy": "glossy",
        #     "normal": "normal",
        # },
        # "ref_gaussian": {
        #     "render": "rgb",
        #     "diffuse": "diffuse",
        #     "glossy": "glossy",
        #     "normal": "normal",
        # },
        "ours": {
            "render": "render",
            "diffuse": "diffuse",
            "glossy": "glossy",
            "normal": "normal",
        }
    }

    directions_to_eval: List[int] = tuple([i for i in range(25) if i not in [2, 3, 19, 20, 21, 22, 24]])
    matching: Literal["none", "histogram", "meanstd"] = "none"
    normalize: bool = False
    num_frames: int = 100

    resolution: Optional[int] = None


if __name__ == "__main__":

    conf = tyro.cli(Conf)

    import torch
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchvision.transforms.functional import to_tensor
    from torchvision.transforms.functional import to_tensor
    import numpy as np
    from kornia.color.lab import rgb_to_lab, lab_to_rgb
    from skimage.exposure import match_histograms

    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = dict(
        psnr=PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(device), 
        lpips=LearnedPerceptualImagePatchSimilarity(normalize=True).to(device),
        ssim=StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device), 
        # kid=None
    )

    base_path = os.path.dirname(os.path.abspath(__file__))

    scores_by_scene = {}

    for scene in conf.scenes:
        scores = {}
        for method in conf.methods:
            scores[method] = { render_pass: { key: 0.0 for key in metrics.keys() } for render_pass in conf.render_passes }

            for render_pass in conf.render_passes:
                print(scene, method, render_pass)
                n_samples=0
                for i in tqdm(range(conf.num_frames)):
                    if method in ["3dgs"] and render_pass == "normal":
                        for metric, metric_fn in metrics.items():
                            scores[method][render_pass][metric] = float("nan")
                        continue

                    gt_path = base_path + "/" + conf.ground_truth_pattern.format(scene=scene, i=i, dir=dir, render_pass=render_pass) 
                    pred_path = base_path + "/" + conf.predictions_path_pattern[method].format(scene=scene, i=i, dir=dir, render_pass=conf.render_pass_naming_schemes[method][render_pass])
                    if conf.matching != "none":
                        recolored_path = base_path + f"/recolored_{conf.matching}_" + conf.predictions_path_pattern[method].format(scene=scene, i=i, dir=dir, render_pass="rgb")

                    gt = Image.open(gt_path).convert("RGB")
                    pred = Image.open(pred_path).convert("RGB")

                    if conf.resolution is not None:
                        assert pred.size[0] >= conf.resolution
                        assert pred.size[0] / pred.size[1] == 1.5
                        pred = pred.resize((int(conf.resolution*1.5), conf.resolution), Image.LANCZOS)
                        gt = gt.resize((int(conf.resolution*1.5), conf.resolution), Image.LANCZOS)
                    
                    if conf.matching == "histogram":
                        pred = match_histograms(pred,gt,channel_axis=0) 
                        Image.fromarray(pred).save(recolored_path)

                    pred = to_tensor(pred)
                    gt = to_tensor(gt)

                    if conf.matching == "meanstd":
                        def match_color(reference, image):
                            a_lab = rgb_to_lab(reference)
                            b_lab = rgb_to_lab(image)

                            def match_statistics(image, reference):
                                return (image - image.mean()) / image.std() * reference.std() + reference.mean()

                            c_lab_L = match_statistics(b_lab[0:1], a_lab[0:1])
                            c_lab_A = match_statistics(b_lab[1:2], a_lab[1:2])
                            c_lab_B = match_statistics(b_lab[2:3], a_lab[2:3])
                            c_lab = torch.cat([c_lab_L, c_lab_A, c_lab_B])

                            return lab_to_rgb(c_lab) 

                        pred = match_color(gt, pred)
                        os.makedirs(os.path.dirname(recolored_path), exist_ok=True)
                        Image.fromarray((pred * 255).moveaxis(0, -1).cpu().numpy().astype(np.uint8)).save(recolored_path)

                    for metric, metric_fn in metrics.items():
                        scores[method][render_pass][metric] += metric_fn(pred[None].to(device), gt[None].to(device)).item()
                    n_samples += 1

                for metric, metric_fn in metrics.items():
                    if metric is not None and n_samples != 0:
                        scores[method][render_pass][metric] /= n_samples

        scores_by_scene[scene] = scores
        with open(base_path + f"/scores.json", "w") as f:
            json.dump(scores_by_scene, f, indent=4)
