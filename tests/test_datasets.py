import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams
from scene.dataset import BlenderDataset, BlenderPriorDataset
from scene.dataset.points_utils import get_point_cloud


def vis_tensor(image, image_path):
    if not os.path.isdir(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

    image = image.detach().cpu().numpy()
    # image = (image - image.min()) / (image.max() - image.min())
    image = image.clip(0.0, 1.0)
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(image_path)


def test_blender_prior_dataset():
    scene_list = ["shiny_kitchen", "shiny_livingroom", "shiny_office", "shiny_bedroom"]
    output_dir = "./output/tests"
    os.makedirs(output_dir, exist_ok=True)

    max_tolerances = {
        "image": 0.05,
        "diffuse": 0.05,
        "glossy": 0.01,
        "position": 0.05,
    }

    for scene_name in tqdm(scene_list):
        model_params = ModelParams(parser=argparse.ArgumentParser())
        data_dir = f"data/renders/{scene_name}"
        point_cloud = get_point_cloud(data_dir)
        dataset0 = BlenderDataset(model_params, data_dir, point_cloud)
        data_dir = f"data/shiny_dataset_priors/{scene_name}"
        dataset1 = BlenderPriorDataset(model_params, data_dir, point_cloud)
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

            vis_tensor(image0, os.path.join(output_dir, scene_name, f"{k}_0.png"))
            vis_tensor(image1, os.path.join(output_dir, scene_name, f"{k}_1.png"))

            mean_diff = (image0 - image1).abs().mean()
            if k in max_tolerances:
                assert mean_diff < max_tolerances[k], f"{k} is not close"
