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

from gaussian_tracing.arguments import ModelParams
from gaussian_tracing.dataset import BlenderDataset, BlenderPriorDataset, ColmapDataset
from gaussian_tracing.dataset.points_utils import make_random_point_cloud
from gaussian_tracing.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_tracing.utils.ply_utils import read_ply

from .cameras import Camera


@dataclass
class SceneInfo:
    point_cloud: BasicPointCloud
    extra_point_cloud: BasicPointCloud
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


def get_dataset(model_params: ModelParams, data_dir: str, split: str):
    if os.path.exists(os.path.join(data_dir, "transforms_train.json")):
        if os.path.isdir(os.path.join(data_dir, "train", "preview")) or os.path.isdir(
            os.path.join(data_dir, "priors", "preview")
        ):
            dataset = BlenderPriorDataset(model_params, data_dir, split=split)
        else:
            dataset = BlenderDataset(model_params, data_dir, split=split)
    else:
        dataset = ColmapDataset(model_params, data_dir, split=split)
    return dataset


def read_dataset(dataset, num_workers=16):
    max_workers = min(num_workers, os.cpu_count() or 1)
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


def readSceneInfo(model_params: ModelParams, data_dir: str) -> SceneInfo:
    print("Reading Training Transforms")
    train_dataset = get_dataset(
        model_params,
        data_dir,
        split="train",
    )
    train_cameras = read_dataset(train_dataset)
    print("Reading Test Transforms")
    test_dataset = get_dataset(
        model_params,
        data_dir,
        split="test",
    )
    test_cameras = read_dataset(test_dataset)

    if "USE_COLMAP_INIT" in os.environ:
        points, colors = read_ply(os.path.join(data_dir, "point_cloud_sfm.ply"))
    else:
        points, colors = read_ply(os.path.join(data_dir, "point_cloud_dense.ply"))

    point_cloud = BasicPointCloud(
        points=points,
        colors=colors,
        normals=np.zeros_like(points),
    )
    extra_point_cloud = make_random_point_cloud(model_params)

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        extra_point_cloud=extra_point_cloud,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        nerf_normalization=getNerfppNorm(train_cameras),
        ply_path=os.path.join(data_dir, "sparse/0/points3D.ply"),
    )
    return scene_info
