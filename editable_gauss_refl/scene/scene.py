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
import random

import torch

from editable_gauss_refl.cfg import Config
from editable_gauss_refl.scene.dataset_readers import (
    readSceneInfo,
)
from editable_gauss_refl.scene.gaussian_model import GaussianModel


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        cfg: Config,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        glossy=False,
        extend_point_cloud=None,
        model_path=None,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg
        self.model_path = model_path or cfg.model_path
        self.gaussians = gaussians
        self.glossy = glossy

        self.loaded_iter = load_iteration

        self.train_cameras = {}
        self.test_cameras = {}

        data_dir = self.cfg.source_path
        scene_info = readSceneInfo(cfg, data_dir)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling

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
            )

        self.gaussians.scene = self

    def select_points_to_prune_near_cameras(self, points, scales):
        # Delete all gaussians that would intesect a sphere around each camera at 3 stddev
        # The sphere radius is determined by the distance to the closest point

        points_to_prune = torch.zeros(points.shape[0], dtype=torch.bool, device="cuda")

        for camera in self.train_cameras[1.0]:
            if isinstance(camera.camera_center, torch.Tensor):
                T = camera.camera_center
            else:
                T = torch.from_numpy(camera.camera_center)

            points_dist_to_camera = (points - T).norm(dim=1)
            too_close = points_dist_to_camera < camera.znear

            points_to_prune |= too_close

        return points_to_prune

    @torch.no_grad()
    def autoadjust_zplanes(self):
        for camera in self.train_cameras[1.0] + self.test_cameras[1.0]:
            camera.znear = camera.depth_image.amin() * self.cfg.znear_scaledown
            camera.zfar = camera.depth_image.amax() * self.cfg.zfar_scaleup
            camera.update()

        # Assert that for all cameras, image_height is equal and FoVy is equal
        train_cameras = self.train_cameras[1.0]
        first_train_camera = train_cameras[0]
        for camera in train_cameras:
            assert camera.image_height == first_train_camera.image_height, "All train cameras must have the same image_height"
            assert camera.FoVy == first_train_camera.FoVy, "All train cameras must have the same FoVy"

        self.max_zfar = max([x.zfar for x in self.train_cameras[1.0]]).item()

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
