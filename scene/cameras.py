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
                 base_color_image,
                 brdf_image,
                 specular_image,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 random_pool=False
                 ):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.diffuse_image = diffuse_image
        self.glossy_image = glossy_image

        self._normal_image = normal_image
        self._position_image = position_image
        self._roughness_image = roughness_image
        self._brdf_image = brdf_image
        self._F0_image = ((1.0 - metalness_image) * 0.08 * specular_image + metalness_image * base_color_image).cuda()

        # self.base_color_image = base_color_image
        # self.metalness_image = metalness_image 
        # self.specular_image = specular_image

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

        incident = position_image.moveaxis(0, -1) - self.camera_center
        incident = incident / incident.norm(dim=-1, keepdim=True)
        self._reflection_ray_image = (incident - 2 * (incident * normal_image.moveaxis(0, -1)).sum(dim=-1).unsqueeze(-1) * normal_image.moveaxis(0, -1)).moveaxis(-1, 0)

        self.random_pool = random_pool

    def sample_position_image(self):
        if self.random_pool:
            if torch.is_grad_enabled():
                return _random_pool(self._position_image)
            else:
                return torch.nn.functional.interpolate(self._position_image[None], scale_factor=1/3, mode='nearest')[0]
        else:
            return self._position_image

    def sample_normal_image(self):
        if self.random_pool:
            if torch.is_grad_enabled():
                return _random_pool(self._normal_image)
            else:
                return torch.nn.functional.interpolate(self._normal_image[None], scale_factor=1/3, mode='nearest')[0]
        else:
            return self._normal_image
        
    def sample_roughness_image(self):
        if self.random_pool:
            if torch.is_grad_enabled():
                return _random_pool(self._roughness_image)
            else:
                return torch.nn.functional.interpolate(self._roughness_image[None], scale_factor=1/3, mode='nearest')[0]
        else:
            return self._roughness_image
        
    def sample_brdf_image(self):
        if self.random_pool:
            if torch.is_grad_enabled():
                return _random_pool(self._brdf_image)
            else:
                return torch.nn.functional.interpolate(self._brdf_image[None], scale_factor=1/3, mode='nearest')[0]
        else:
            return self._brdf_image
        
    def sample_F0_image(self):
        if self.random_pool:
            if torch.is_grad_enabled():
                return _random_pool(self._F0_image)
            else:
                return torch.nn.functional.interpolate(self._F0_image[None], scale_factor=1/3, mode='nearest')[0]
        else:
            return self._F0_image

    def sample_reflection_ray_image(self):
        if self.random_pool:
            if torch.is_grad_enabled():
                return _random_pool(self._reflection_ray_image)
            else:
                return torch.nn.functional.interpolate(self._reflection_ray_image[None], scale_factor=1/3, mode='nearest')[0]
        else:
            return self._reflection_ray_image
        
    def update(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.focal_x = self.image_width / (2 * np.tan(self.FoVx * 0.5))
        self.focal_y = self.image_height / (2 * np.tan(self.FoVy * 0.5))


def _random_pool(x):
    patches = torch.nn.functional.unfold(x.unsqueeze(1), kernel_size=(3, 3), stride=(3, 3), padding=0) 
    indices = torch.randint(0, 9, (patches.shape[2],), device=x.device)
    return patches[:, indices, torch.arange(patches.shape[2])].reshape(x.shape[0], x.shape[1]//3, x.shape[2]//3)



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

