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

import json
import math
import os
import random
import shutil

import numpy as np
import torch
from diff_gaussian_tracing import raytracer_config

from arguments import ModelParams
from scene.cameras import Camera
from scene.dataset_readers import (
    readSceneInfo,
)
from scene.gaussian_model import BasicPointCloud, GaussianModel
from utils.system_utils import searchForMaxIteration


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        model_params: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        glossy=False,
        extend_point_cloud=None,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_params = model_params
        self.model_path = model_params.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.glossy = glossy

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        data_dir = model_params.source_path
        scene_info = readSceneInfo(model_params, data_dir)
        scene_info.train_cameras = scene_info.train_cameras[
            :: model_params.keep_every_kth_view
        ]

        if "HARD_SPARSE" in os.environ:
            cameras = scene_info.train_cameras
            cameras = [cameras[0], cameras[49], cameras[100], cameras[199]]
            scene_info.train_cameras = cameras

        if model_params.sparseness != -1:
            cameras = sorted(scene_info.train_cameras, key=lambda x: x.image_path)
            cameras = cameras[:50] + cameras[-50:]
            # take every kth cameras where k = args.sparseness
            scene_info.train_cameras = cameras[:: model_params.sparseness]

        scene_info.train_cameras = scene_info.train_cameras[
            model_params.skip_n_images :
        ]

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = scene_info.train_cameras
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = scene_info.test_cameras

        print(f"I have {len(self.train_cameras[resolution_scales[0]])} cameras")

        self.autoadjust_zplanes()

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(
                scene_info.point_cloud,
                self.cameras_extent,
                self.train_cameras[1.0][0].FoVy,
                self.max_zfar,
                raytracer_config,
            )

        self.gaussians.scene = self

    def select_points_to_prune_near_cameras(
        self, points, scales, sigma=int(os.getenv("PRUNING_SIGMA", 0))
    ):
        # Delete all gaussians that would intesect a sphere around each camera at 3 stddev
        # The sphere radius is determined by the distance to the closest point

        points_to_prune = torch.zeros(points.shape[0], dtype=torch.bool, device="cuda")

        if "SKIPZNEAR" in os.environ:
            return points_to_prune

        for camera in self.train_cameras[1.0]:
            if isinstance(camera.R, torch.Tensor):
                R = camera.R.cuda().float()
            else:
                R = torch.from_numpy(camera.R).cuda().float()
            if isinstance(camera.camera_center, torch.Tensor):
                T = camera.camera_center
            else:
                T = torch.from_numpy(camera.camera_center)

            points_dist_to_camera = (points - T).norm(dim=1)
            too_close = (
                points_dist_to_camera - sigma * scales.amax(dim=1) < camera.znear
            )

            points_to_prune |= too_close

        return points_to_prune

    @torch.no_grad()
    def autoadjust_zplanes(self):
        for camera in self.train_cameras[1.0] + self.test_cameras[1.0]:
            R = torch.from_numpy(camera.R).cuda().float()
            T = camera.camera_center
            distances = (
                camera.position_image - camera.camera_center[:, None, None]
            ).norm(dim=0)
            camera.znear = distances.amin() * self.model_params.znear_scaledown
            camera.zfar = distances.amax() * self.model_params.zfar_scaleup
            camera.update()

        # Assert that for all cameras, image_height is equal and FoVy is equal
        train_cameras = self.train_cameras[1.0]
        first_train_camera = train_cameras[0]
        for camera in train_cameras:
            assert camera.image_height == first_train_camera.image_height, (
                "All train cameras must have the same image_height"
            )
            assert camera.FoVy == first_train_camera.FoVy, (
                "All train cameras must have the same FoVy"
            )

        self.max_zfar = max([x.zfar for x in self.train_cameras[1.0]]).item()
        self.max_pixel_blur_sigma = (
            self.gaussians.model_params.lod_max_world_size_blur
            / (
                2
                * math.tan(self.train_cameras[1.0][0].FoVy / 2)
                / self.train_cameras[1.0][0].image_height
                * self.max_zfar
            )
        )

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
