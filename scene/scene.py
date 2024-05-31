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
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
 
class Scene:
    gaussians : GaussianModel

    def __init__(self, modelParams : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], dynamic=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.modelParams = modelParams
        self.model_path = modelParams.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.dynamic = dynamic

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(modelParams.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](modelParams, modelParams.source_path, modelParams.images, modelParams.eval)
        elif os.path.exists(os.path.join(modelParams.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](modelParams, modelParams.source_path, modelParams.white_background, modelParams.eval)
        else:
            assert False, "Could not recognize scene type!"

        scene_info.train_cameras = scene_info.train_cameras[::modelParams.keep_every_kth_view]

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
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
        
        if not self.dynamic:
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, modelParams)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, modelParams)

        print(f"I have {len(self.train_cameras)} cameras")

        if self.loaded_iter:
            if self.modelParams.convert_mlp:
                self.gaussians.mlp = torch.load(
                    os.path.join(self.model_path,
                                    "point_cloud",
                                    "iteration_" + str(self.loaded_iter),
                                    f"{'dyn_' if self.dynamic else ''}mlp.pt")
                )
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           f"{'dyn_' if self.dynamic else ''}point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, f"{'dyn_' if self.dynamic else ''}point_cloud.ply"))
        if self.modelParams.convert_mlp:
            torch.save(self.gaussians.mlp, os.path.join(point_cloud_path, f"{'dyn_' if self.dynamic else ''}mlp.pt"))
            if self.modelParams.use_tcnn:
                import io
                buff = io.BytesIO()
                torch.save(self.gaussians.mlp.state_dict()['params'], buff)
                with open(os.path.join(point_cloud_path, f"{'dyn_' if self.dynamic else ''}mlp.bin"), "wb") as f:
                    f.write(buff.getvalue())


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]