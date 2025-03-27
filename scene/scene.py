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
import math 

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
                camlist.extend(scene_info_synthetic.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if model_params.sparseness != -1:
            cameras = sorted(scene_info.train_cameras, key=lambda x: x.image_path)
            cameras = cameras[:50] + cameras[-50:]
            # take every kth cameras where k = args.sparseness
            scene_info.train_cameras = cameras[::model_params.sparseness]

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, model_params)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info_synthetic.test_cameras, resolution_scale, model_params)

        print(f"I have {len(self.train_cameras)} cameras")

        ## Remove all init points that are too close to the camera
        if gaussians.model_params.znear_init_pruning or gaussians.model_params.znear_densif_pruning:
            self.autoadjust_zplanes()
        
        if gaussians.model_params.znear_init_pruning:
            points = torch.from_numpy(scene_info_synthetic.point_cloud.points).cuda().float()
            points_to_prune = self.select_points_to_prune_near_cameras(points)
            
            print(f"Pruned {points_to_prune.float().mean() * 100:.2f}% of the extra init points since they are too close to the cameras.")
            extra_points = scene_info_synthetic.point_cloud.points[(~points_to_prune).cpu().numpy()]
            extra_colors = scene_info_synthetic.point_cloud.colors[(~points_to_prune).cpu().numpy()]
            extra_normals = scene_info_synthetic.point_cloud.normals[(~points_to_prune).cpu().numpy()]
        else:
            extra_points = scene_info_synthetic.point_cloud.points
            extra_colors = scene_info_synthetic.point_cloud.colors
            extra_normals = scene_info_synthetic.point_cloud.normals

        self.autoadjust_zplanes()
            
        import sys
        sys.path.append(gaussians.model_params.raytracer_version)
        import raytracer_config
        if raytracer_config.MAX_BOUNCES > 0 and "SKIP_EXTRA_POINTS" not in os.environ:
            scene_info.point_cloud = BasicPointCloud(
                np.concatenate([scene_info.point_cloud.points, extra_points]),
                np.concatenate([scene_info.point_cloud.colors, extra_colors]),
                np.concatenate([scene_info.point_cloud.normals, extra_normals])
            )

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.train_cameras[1.0][0].FoVy, self.max_zfar, raytracer_config)
        
        self.gaussians.scene = self

    def select_points_to_prune_near_cameras(self, points):
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

            if False:
                points_world = gaussians.get_xyz
                R_c2w_blender = -R 
                R_c2w_blender[:, 0] = -R_c2w_blender[:, 0]
                points_local = (R_c2w_blender.T @ (points_world - T).T).T
                
                x_size = math.tan(camera.FoVx / 2)
                y_size = math.tan(camera.FoVy / 2)
                
                x = points_local[:, 0]
                y = points_local[:, 1]
                z = points_local[:, 2]
                in_frustrum_cone = (x / z > -x_size) & (x / z < x_size) & (y / z > -y_size) & (y / z < y_size)
                points_to_prune = ~in_frustrum_cone | (in_frustrum_cone & (-z > camera.znear))
            else:
                points_dist_to_camera = (points - T).norm(dim=1)
                too_close = points_dist_to_camera < camera.znear

                points_to_prune |= too_close

        return points_to_prune

    @torch.no_grad()
    def autoadjust_zplanes(self):
        for camera in self.train_cameras[1.0] + self.test_cameras[1.0]:
            R = torch.from_numpy(camera.R).cuda().float()
            T = camera.camera_center
            distances = (camera.position_image - camera.camera_center[:, None, None]).norm(dim=0)
            camera.znear = distances.amin() * self.model_params.znear_scaledown
            camera.zfar = distances.amax() * self.model_params.zfar_scaleup
            camera.update()

        # Assert that for all cameras, image_height is equal and FoVy is equal
        train_cameras = self.train_cameras[1.0]
        test_cameras = self.test_cameras[1.0]
        first_train_camera = train_cameras[0]
        first_test_camera = test_cameras[0]
        for camera in train_cameras:
            assert camera.image_height == first_train_camera.image_height, "All train cameras must have the same image_height"
            assert camera.FoVy == first_train_camera.FoVy, "All train cameras must have the same FoVy"
        for camera in test_cameras:
            assert camera.image_height == first_test_camera.image_height, "All test cameras must have the same image_height"
            assert camera.FoVy == first_test_camera.FoVy, "All test cameras must have the same FoVy"

        self.max_zfar = max([x.zfar for x in self.train_cameras[1.0]]).item()
        self.max_pixel_blur_sigma = self.gaussians.model_params.lod_max_world_size_blur / (2 * math.tan(self.train_cameras[1.0][0].FoVy / 2) / self.train_cameras[1.0][0].image_height * self.max_zfar) 

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]