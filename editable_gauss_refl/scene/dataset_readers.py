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

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from editable_gauss_refl.config import Config
from editable_gauss_refl.dataset import (
    BlenderDataset,
    BlenderPriorDataset,
    ColmapPriorDataset,
)
from editable_gauss_refl.utils.graphics_utils import BasicPointCloud, getWorld2View2
from editable_gauss_refl.utils.ply_utils import read_ply

from .cameras import Camera


@dataclass
class SceneInfo:
    point_cloud: BasicPointCloud
    train_cameras: List[Camera]
    test_cameras: List[Camera]
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cameras: List[Camera]) -> dict:
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cameras:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def get_dataset(cfg: Config, data_dir: str, split: str):
    if os.path.exists(os.path.join(data_dir, "transforms_train.json")):
        # * This is a bit of a kludge: we check for depth format to distinguish between g.t. render and prdicted buffer for the blender scenes
        if os.path.isfile(os.path.join(data_dir, split, "depth", "depth_0000.tiff")):
            dataset = BlenderDataset(
                data_dir,
                split=split,
                resolution=cfg.resolution,
                max_images=cfg.max_images,
            )
        else:
            dataset = BlenderPriorDataset(
                data_dir,
                split=split,
                resolution=cfg.resolution,
                max_images=cfg.max_images,
            )
    elif os.path.exists(data_dir):
        dataset = ColmapPriorDataset(
            data_dir,
            split=split,
            resolution=cfg.resolution,
            max_images=cfg.max_images,
            do_eval=False
        )
    else:
        raise FileNotFoundError(f"Data directory {data_dir} not found.")
    return dataset


def read_dataset(dataset, num_workers=16):
    max_workers = min(num_workers, os.cpu_count() // 2)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0 if "NO_WORKERS" in os.environ else max_workers,
        collate_fn=lambda x: x,
        persistent_workers=False if "NO_WORKERS" in os.environ else True,
    )
    cameras = []
    for cam_info_batch in tqdm(dataloader):
        camera = Camera.from_cam_info(cam_info_batch[0])
        cameras.append(camera)
    return cameras


def readSceneInfo(cfg: Config, data_dir: str) -> SceneInfo:
    print("Reading Training Transforms")
    train_dataset = get_dataset(
        cfg,
        data_dir,
        split="train",
    )
    train_cameras = read_dataset(train_dataset)
    print("Reading Test Transforms")
    test_dataset = get_dataset(
        cfg,
        data_dir,
        split="test",
    )
    test_cameras = read_dataset(test_dataset)

    points, colors = read_ply(os.path.join(data_dir, f"point_cloud_{cfg.init_type}.ply"))

    point_cloud = BasicPointCloud(
        points=points,
        colors=colors,
        normals=np.zeros_like(points),
    )

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        nerf_normalization=getNerfppNorm(train_cameras),
        ply_path=os.path.join(data_dir, "sparse/0/points3D.ply"),
    )
    return scene_info
