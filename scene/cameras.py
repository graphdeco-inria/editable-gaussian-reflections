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

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 diffuse_image,
                 glossy_image,
                 position_image,
                 normal_image,
                 roughness_image,
                 metalness_image,
                 albedo_image,
                 brdf_image,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 ):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.diffuse_image = diffuse_image * (1.0 - metalness_image) #!!!
        self.glossy_image = glossy_image
        
        self.normal_image = normal_image
        self.position_image = position_image

        self.roughness_image = roughness_image
        self.metalness_image = metalness_image
        self.albedo_image = albedo_image
        self.brdf_image = brdf_image

        self.F0_image = self.albedo_image * self.metalness_image # + 0.08 * self.specular_image

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = self.diffuse_image + self.glossy_image
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.R = R
        self.T = T

        self.update()

        incident = self.position_image.moveaxis(0, -1) - self.camera_center.cpu()
        incident = incident / incident.norm(dim=-1, keepdim=True)
        self.reflection_ray_image = (incident - 2 * (incident * self.normal_image.moveaxis(0, -1)).sum(dim=-1).unsqueeze(-1) * self.normal_image.moveaxis(0, -1)).moveaxis(-1, 0)


    def update(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.focal_x = self.image_width / (2 * np.tan(self.FoVx * 0.5))
        self.focal_y = self.image_height / (2 * np.tan(self.FoVy * 0.5))

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

