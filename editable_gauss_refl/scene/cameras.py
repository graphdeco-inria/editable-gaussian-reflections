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

import numpy as np
import torch
from torch import nn

from editable_gauss_refl.utils.graphics_utils import getProjectionMatrix, getWorld2View2
from editable_gauss_refl.utils.tonemapping import untonemap


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        glossy_image,
        diffuse_image,
        depth_image,
        normal_image,
        roughness_image,
        f0_image,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.image_width = diffuse_image.shape[2]
        self.image_height = diffuse_image.shape[1]

        image_holding_device = os.getenv("IMAGE_HOLDING_DEVICE", "cpu")

        if image.dtype == torch.uint8:
            image = untonemap(image.cuda().half() / 255.0)
            diffuse_image = untonemap(diffuse_image.cuda().half() / 255.0)
            glossy_image = untonemap(glossy_image.cuda().half() / 255.0)
        if normal_image.dtype == torch.uint8:
            normal_image = normal_image.half() / 255.0 * 2.0 - 1.0
        if depth_image.dtype == torch.uint8:
            assert False
        if roughness_image.dtype == torch.uint8:
            roughness_image = roughness_image.half() / 255.0
        if f0_image.dtype == torch.uint8:
            f0_image = f0_image.half() / 255.0
        if roughness_image.shape[-1] == 3:
            roughness_image = roughness_image[..., :1]
        if depth_image.shape[-1] == 3:
            depth_image = depth_image[..., :1]

        self._original_image = image.half().to(image_holding_device)
        self._diffuse_image = diffuse_image.half().to(image_holding_device)
        self._glossy_image = glossy_image.half().to(image_holding_device)
        self._normal_image = normal_image.half().to(image_holding_device)
        self._depth_image = depth_image.half().to(image_holding_device)
        self._roughness_image = roughness_image.half().to(image_holding_device)
        self._f0_image = f0_image.half().to(image_holding_device)

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

    @classmethod
    def from_cam_info(cls, cam_info):
        return cls(
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image=cam_info.image.moveaxis(-1, 0),
            gt_alpha_mask=None,
            image_name=cam_info.image_name,
            uid=cam_info.uid,
            data_device="cuda",
            diffuse_image=cam_info.diffuse_image.moveaxis(-1, 0),
            glossy_image=cam_info.glossy_image.moveaxis(-1, 0),
            depth_image=cam_info.depth_image.moveaxis(-1, 0),
            normal_image=cam_info.normal_image.moveaxis(-1, 0),
            roughness_image=cam_info.roughness_image.moveaxis(-1, 0),
            f0_image=cam_info.f0_image.moveaxis(-1, 0),
        )

    @property
    def original_image(self):
        return self._original_image.float().cuda()

    @property
    def diffuse_image(self):
        return self._diffuse_image.float().cuda()

    @property
    def glossy_image(self):
        return self._glossy_image.float().cuda()

    @property
    def normal_image(self):
        return self._normal_image.float().cuda()

    @property
    def depth_image(self):
        return self._depth_image.float().cuda()

    @property
    def roughness_image(self):
        return self._roughness_image.float().cuda()

    @property
    def f0_image(self):
        return self._f0_image.float().cuda()

    def update(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.cpu().inverse().cuda()[3, :3]


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
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
