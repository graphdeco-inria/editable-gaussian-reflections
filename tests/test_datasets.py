import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams
from scene.dataset import BlenderDataset, BlenderPriorDataset


def vis_tensor(image, image_path):
    if not os.path.isdir(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

    image = image.detach().cpu().numpy()
    # image = (image - image.min()) / (image.max() - image.min())
    image = image.clip(0.0, 1.0)
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(image_path)


def test_blender_prior_dataset():
    model_params = ModelParams(parser=argparse.ArgumentParser())
    output_dir = "./output/tests"
    os.makedirs(output_dir, exist_ok=True)

    scene_list = ["shiny_kitchen", "shiny_livingroom", "shiny_office", "shiny_bedroom"]
    for scene_name in scene_list:
        data_dir = f"data/renders/{scene_name}"
        dataset0 = BlenderDataset(model_params, data_dir)
        data_dir = f"data/shiny_dataset_priors/{scene_name}"
        dataset1 = BlenderPriorDataset(model_params, data_dir)
        cam_info0 = dataset0[0]
        cam_info1 = dataset1[0]

        for key, value in asdict(cam_info0).items():
            if not isinstance(value, torch.Tensor):
                continue
            if "image" not in key:
                continue

            image0 = getattr(cam_info0, key)
            image1 = getattr(cam_info1, key)
            vis_tensor(image0, os.path.join(output_dir, scene_name, f"{key}_0.png"))
            vis_tensor(image1, os.path.join(output_dir, scene_name, f"{key}_1.png"))

        image_diff = cam_info0.image - cam_info1.image
        assert image_diff.abs().mean() < 0.05, "image is not close"

        diffuse_diff = cam_info0.diffuse_image - cam_info1.diffuse_image
        assert diffuse_diff.abs().mean() < 0.05, "diffuse is not close"
        glossy_diff = cam_info0.glossy_image - cam_info1.glossy_image
        assert glossy_diff.abs().mean() < 0.05, "glossy is not close"

        position_diff = cam_info0.position_image - cam_info1.position_image
        assert position_diff.abs().mean() < 0.1, "position is not close"
