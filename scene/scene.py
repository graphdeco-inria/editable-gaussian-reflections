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
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import *
from scene.gaussian_model import GaussianModel
# from scene.surfel_model import SurfelModel

from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
 
class Scene:
    gaussians: GaussianModel

    def __init__(self, model_params: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], glossy=False, extend_point_cloud=None):
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
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        source_path_synthetic = model_params.source_path.replace("colmap/", "renders/").replace("/train", "")
        assert os.path.exists(os.path.join(source_path_synthetic, "transforms_train.json")), source_path_synthetic
        scene_info_synthetic = readNerfSyntheticInfo(model_params, source_path_synthetic, model_params.white_background, model_params.eval)
        
        source_path_colmap = model_params.source_path + "/train"
        assert os.path.exists(os.path.join(source_path_colmap, "sparse")), source_path_colmap
        scene_info = readColmapSceneInfo(model_params, source_path_colmap, model_params.images, model_params.eval)

        self.scene_info = scene_info
        scene_info.train_cameras = scene_info.train_cameras[::model_params.keep_every_kth_view]

        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, model_params)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, model_params)

        print(f"I have {len(self.train_cameras)} cameras")

        scene_info.point_cloud = BasicPointCloud(
            np.concatenate([scene_info.point_cloud.points, scene_info_synthetic.point_cloud.points]),
            np.concatenate([scene_info.point_cloud.colors, scene_info_synthetic.point_cloud.colors]),
            np.concatenate([scene_info.point_cloud.normals, scene_info_synthetic.point_cloud.normals]),
        )

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           f"{'glossy_' if self.glossy else ''}point_cloud.ply"), self.glossy)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, f"{'glossy_' if self.glossy else ''}point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]