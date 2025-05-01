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
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams
from scene.dataset import BlenderDataset, BlenderPriorDataset, ColmapDataset
from scene.dataset.points_utils import get_point_cloud, make_random_point_cloud
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import getWorld2View2


@dataclass
class SceneInfo:
    point_cloud: BasicPointCloud
    extra_point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


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
    cam_infos = []
    for cam_info_batch in tqdm(dataloader):
        # Use deepcopy to avoid too many open files error
        cam_info = deepcopy(cam_info_batch[0])
        cam_infos.append(cam_info)
        del cam_info_batch
    return cam_infos


def readColmapSceneInfo(model_params: ModelParams, data_dir: str) -> SceneInfo:
    point_cloud = get_point_cloud(data_dir)
    extra_point_cloud = make_random_point_cloud(model_params)

    train_dataset = ColmapDataset(
        model_params,
        data_dir,
        point_cloud,
        split="train",
    )
    test_dataset = ColmapDataset(
        model_params,
        data_dir,
        point_cloud,
        split="test",
    )
    print("Reading Training Transforms")
    train_cam_infos = read_dataset(train_dataset)
    print("Reading Test Transforms")
    test_cam_infos = read_dataset(test_dataset)

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        extra_point_cloud=extra_point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=getNerfppNorm(train_cam_infos),
        ply_path=os.path.join(data_dir, "sparse/0/points3D.ply"),
    )
    return scene_info


def readBlenderSceneInfo(model_params: ModelParams, data_dir: str) -> SceneInfo:
    point_cloud = get_point_cloud(data_dir)
    extra_point_cloud = make_random_point_cloud(model_params)

    train_dataset = BlenderDataset(
        model_params,
        data_dir,
        point_cloud,
        split="train",
    )
    test_dataset = BlenderDataset(
        model_params,
        data_dir,
        point_cloud,
        split="test",
    )
    print("Reading Training Transforms")
    train_cam_infos = read_dataset(train_dataset)
    print("Reading Test Transforms")
    test_cam_infos = read_dataset(test_dataset)

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        extra_point_cloud=extra_point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=getNerfppNorm(train_cam_infos),
        ply_path=os.path.join(data_dir, "sparse/0/points3D.ply"),
    )
    return scene_info


def readBlenderPriorSceneInfo(model_params: ModelParams, data_dir: str) -> SceneInfo:
    point_cloud = get_point_cloud(data_dir)
    extra_point_cloud = make_random_point_cloud(model_params)

    if "PRIOR_DATASET_TRAIN_ONLY" in os.environ:
        train_dataset = BlenderPriorDataset(
            model_params, data_dir, point_cloud, split="train", dirname="priors"
        )
        train_cam_infos = read_dataset(train_dataset)

        scene_info = SceneInfo(
            point_cloud=point_cloud,
            extra_point_cloud=extra_point_cloud,
            train_cameras=train_cam_infos,
            test_cameras=train_cam_infos,  #!!!
            nerf_normalization=getNerfppNorm(train_cam_infos),
            ply_path=os.path.join(data_dir, "sparse/0/points3D.ply"),
        )

        return scene_info
    else:
        train_dataset = BlenderPriorDataset(
            model_params,
            data_dir,
            point_cloud,
            split="train",
        )
        test_dataset = BlenderPriorDataset(
            model_params,
            data_dir,
            point_cloud,
            split="test",
        )
        print("Reading Training Transforms")
        train_cam_infos = read_dataset(train_dataset)
        print("Reading Test Transforms")
        test_cam_infos = read_dataset(test_dataset)

    scene_info = SceneInfo(
        point_cloud=point_cloud,
        extra_point_cloud=extra_point_cloud,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=getNerfppNorm(train_cam_infos),
        ply_path=os.path.join(data_dir, "sparse/0/points3D.ply"),
    )
    return scene_info
