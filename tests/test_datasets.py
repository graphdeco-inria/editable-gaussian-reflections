import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from gaussian_tracing.arguments import ModelParams
from gaussian_tracing.dataset import BlenderDataset, BlenderPriorDataset


def vis_tensors(images, image_path):
    if not os.path.isdir(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

    images = [image.detach().cpu().numpy() for image in images]
    images = [image.clip(0.0, 1.0) for image in images]
    grid = np.hstack(images)
    grid = Image.fromarray((grid * 255).astype(np.uint8))
    grid.save(image_path)


def test_blender_prior_dataset():
    scene_list = ["shiny_kitchen", "shiny_livingroom", "shiny_office", "shiny_bedroom"]
    output_dir = "./output/tests"
    os.makedirs(output_dir, exist_ok=True)

    max_tolerances = {
        "image": 0.01,
        "diffuse": 0.01,
        "glossy": 0.01,
        "position": 0.01,
    }

    for scene_name in tqdm(scene_list):
        model_params = ModelParams(parser=argparse.ArgumentParser())
        model_params.resolution = 512

        data_dir = f"data/renders/{scene_name}"
        dataset0 = BlenderDataset(model_params, data_dir)
        data_dir = f"data/real_datasets_v3_filmic/renders_priors/{scene_name}"
        dataset1 = BlenderPriorDataset(model_params, data_dir)
        cam_info0 = dataset0[0]
        cam_info1 = dataset1[0]

        for k, v in asdict(cam_info0).items():
            if "image" not in k:
                continue
            if not isinstance(v, torch.Tensor):
                continue

            image0 = getattr(cam_info0, k)
            image1 = getattr(cam_info1, k)
            assert image0.shape == image1.shape

            vis_tensors(
                [image0, image1],
                os.path.join(output_dir, scene_name, f"{k}_compare.png"),
            )

            median_diff = (image0 - image1).abs().median()
            if k in max_tolerances:
                assert median_diff < max_tolerances[k], f"{k} is not close"
