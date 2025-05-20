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
from torchvision.utils import save_image

from scene.tonemapping import *
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


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
        diffuse_image,
        glossy_image,
        position_image,
        normal_image,
        roughness_image,
        metalness_image,
        base_color_image,
        brdf_image,
        specular_image,
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

        if "CLAMP_BASE_COLOR" in os.environ:
            base_color_image.clamp_(0, 1)

        image_holding_device = os.getenv("IMAGE_HOLDING_DEVICE", "cuda")

        EXPOSURE = float(os.getenv("EXPOSURE", 3.5))
        if "TONEMAP_INPUT" in os.environ:
            # *** optimized as tonemapped values, will need to be inverse the tonemapping before adding both passes
            self._original_image = (
                (tonemap(diffuse_image * EXPOSURE + glossy_image * EXPOSURE))
                .half()
                .to(image_holding_device)
            )
            self._diffuse_image = (
                (tonemap(diffuse_image * EXPOSURE)).half().to(image_holding_device)
            )
            self._glossy_image = (
                (tonemap(glossy_image * EXPOSURE)).half().to(image_holding_device)
            )
            if "DONT_CLAMP_TARGETS" not in os.environ:
                self._original_image = torch.clamp(self._original_image, 0.0, 1.0)
                self._diffuse_image = torch.clamp(self._diffuse_image, 0.0, 1.0)
                self._glossy_image = torch.clamp(self._glossy_image, 0.0, 1.0)
        else:
            self._original_image = (
                (diffuse_image + glossy_image).half().to(image_holding_device)
            )
            self._diffuse_image = diffuse_image.half().to(image_holding_device)
            self._glossy_image = glossy_image.half().to(image_holding_device)
            self._original_image *= EXPOSURE
            self._diffuse_image *= EXPOSURE
            self._glossy_image *= EXPOSURE

            if "CLAMP_TARGETS" in os.environ:
                self._original_image = torch.clamp(self._original_image, 0.0, 1.0)
                self._diffuse_image = torch.clamp(self._diffuse_image, 0.0, 1.0)
                self._glossy_image = torch.clamp(self._glossy_image, 0.0, 1.0)

        if "DIFFUSE_IS_RENDER" in os.environ:
            self._diffuse_image = self._original_image

        self._normal_image = normal_image.half().to(image_holding_device)
        self._position_image = position_image.half().to(image_holding_device)
        self._roughness_image = (roughness_image).half().to(image_holding_device)
        if "ZERO_ROUGHNESS" in os.environ:
            self._roughness_image = self._roughness_image * 0
        self._brdf_image = brdf_image.half().to(image_holding_device)

        self._F0_image = (
            (
                (1.0 - metalness_image)
                * float(os.getenv("DIELECTIC_REFL", 0.08))
                * specular_image
                + metalness_image * base_color_image
            )
            .half()
            .to(image_holding_device)
        )

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.trans = trans
        self.scale = scale

        self.R = R
        self.T = T

        self.update()

        self._depth_image = torch.norm(
            self._position_image
            - self.camera_center.unsqueeze(-1)
            .unsqueeze(-1)
            .to(self._position_image.device),
            dim=0,
        )

    @classmethod
    def from_cam_info(cls, cam_info):
        return cls(
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image=cam_info.image,
            gt_alpha_mask=None,
            image_name=cam_info.image_name,
            uid=cam_info.uid,
            data_device="cuda",
            diffuse_image=cam_info.diffuse_image.moveaxis(-1, 0),
            glossy_image=cam_info.glossy_image.moveaxis(-1, 0),
            position_image=cam_info.position_image.moveaxis(-1, 0),
            normal_image=cam_info.normal_image.moveaxis(-1, 0),
            roughness_image=cam_info.roughness_image.moveaxis(-1, 0),
            metalness_image=cam_info.metalness_image.moveaxis(-1, 0),
            base_color_image=cam_info.base_color_image.moveaxis(-1, 0),
            brdf_image=cam_info.brdf_image.moveaxis(-1, 0),
            specular_image=cam_info.specular_image.moveaxis(-1, 0),
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
    def position_image(self):
        return self._position_image.float().cuda()

    @property
    def depth_image(self):
        return self._depth_image.float().cuda()

    @property
    def roughness_image(self):
        return self._roughness_image.float().cuda()

    @property
    def brdf_image(self):
        return self._brdf_image.float().cuda()

    @property
    def F0_image(self):
        return self._F0_image.float().cuda()

    def update(self):
        self.world_view_transform = (
            torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale))
            .transpose(0, 1)
            .cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
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
