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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import os 

from scene.tonemapping import *

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 diffuse_image,
                 glossy_image,
                 position_image,
                 normal_image,
                 roughness_image,
                 metalness_image,
                 base_color_image,
                 brdf_image,
                 specular_image,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.original_image = tonemap(diffuse_image + glossy_image)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        #*** optimized as tonemapped values, will need to be inverse the tonemapping before adding both passes
        self.diffuse_image = tonemap(diffuse_image) 
        self.glossy_image = tonemap(glossy_image) 

        if "CLAMP01" in os.environ:
            self.diffuse_image = self.diffuse_image.clamp(0.0, 1) 
            self.glossy_image = self.glossy_image.clamp(0.0, 1) 

        if "CLAMP21" in os.environ:
            self.diffuse_image = self.diffuse_image.clamp(0.2, 1) 
            self.glossy_image = self.glossy_image.clamp(0.2, 1) 

        if "CLAMP28" in os.environ:
            self.diffuse_image = self.diffuse_image.clamp(0.2, 0.8) 
            self.glossy_image = self.glossy_image.clamp(0.2, 0.8) 

        if "CLAMP51" in os.environ:
            self.diffuse_image = self.diffuse_image.clamp(0.5, 1) 
            self.glossy_image = self.glossy_image.clamp(0.5, 1) 

        self.normal_image = normal_image
        self.position_image = position_image
        self.roughness_image = roughness_image
        self.brdf_image = brdf_image
        self.F0_image = ((1.0 - metalness_image) * 0.08 * specular_image + metalness_image * base_color_image).cuda()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.trans = trans
        self.scale = scale

        self.R = R
        self.T = T

        self.update()

    #!! todo try using the updates znear and far
    def update(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.cpu().inverse().cuda()[3, :3]


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.R = world_view_transform[:3, :3]

