import glob 
import tyro
import os
from PIL import Image 
from dataclasses import dataclass, field
from tqdm import tqdm 
from typing import *
import json
from torchvision.utils import save_image
import numpy as np
import torch
from tyro.conf import arg
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_tensor, to_pil_image


@dataclass
class Conf:
    model_path: Annotated[str, arg(aliases=["-m"])]

    render_passes: List[int] = field(default_factory=lambda: [ "diffuse", "specular", "render" ])
    metrics: List[int] = field(default_factory=lambda: [ "psnr" ])

    ground_truth_pattern: str = "{model_path}/test/ours_8000/{render_pass}_gt/{i:05d}_{render_pass}.png" 
    predictions_path_pattern = "{model_path}/test/ours_8000/{render_pass}/{i:05d}_{render_pass}.png" 

    num_frames: int = 100


if __name__ == "__main__":
    conf = tyro.cli(Conf)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    metrics = {}
    if "psnr" in conf.metrics:
        metrics["psnr"] = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(device)
    if "ssim" in conf.metrics:
        metrics["ssim"] = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(device)
    if "lpips" in conf.metrics:
        metrics["lpips"] = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    base_path = os.path.dirname(os.path.abspath(__file__))

    scores = { render_pass: { key: 0.0 for key in metrics.keys() } for render_pass in conf.render_passes }
    
    # * Eval scores for each frame
    for i in tqdm(range(conf.num_frames)):
        images = {}

        for render_pass in conf.render_passes:
            gt_path = base_path + "/" + conf.ground_truth_pattern.format(i=i, render_pass=render_pass, model_path=conf.model_path)
            pred_path = base_path + "/" + conf.predictions_path_pattern.format(i=i, render_pass=render_pass, model_path=conf.model_path)
            
            gt = Image.open(gt_path).convert("RGB")
            pred = Image.open(pred_path).convert("RGB")

            pred = to_tensor(pred)[None].to(device)
            gt = to_tensor(gt)[None].to(device)

            for metric, metric_fn in metrics.items():
                scores[render_pass][metric] += metric_fn(pred, gt).item() / conf.num_frames

    # * Round all scores
    for render_pass in conf.render_passes:
        for metric in scores[render_pass].keys():
            scores[render_pass][metric] = round(scores[render_pass][metric], 2)
    
    # * Print and save scores
    print(json.dumps(scores, indent=4))
    with open(os.path.join(conf.model_path, "metrics.json"), "w") as f:
        json.dump(scores, f, indent=4)

