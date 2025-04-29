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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    resolution = args.resolution

    assert isinstance(cam_info.image, torch.Tensor) 
    height = cam_info.image.shape[0]
    width = cam_info.image.shape[1]
    aspect_ratio = width / height

    def downsize(x):
        if resolution != x.shape[-2]:
            return (
                torch.nn.functional.interpolate(
                    x[None].cuda().float(),
                    (resolution, int(resolution * aspect_ratio)),
                    mode="area",
                )[0]
                .half()
                .cpu()
            )
        else:
            return x.cuda()

    diffuse_image = downsize(cam_info.diffuse_image.moveaxis(-1, 0))
    glossy_image = downsize(cam_info.glossy_image.moveaxis(-1, 0))
    position_image = downsize(cam_info.position_image.moveaxis(-1, 0))
    normal_image = downsize(cam_info.normal_image.moveaxis(-1, 0))
    roughness_image = downsize(cam_info.roughness_image.moveaxis(-1, 0))
    metalness_image = downsize(cam_info.metalness_image.moveaxis(-1, 0))
    base_color_image = downsize(cam_info.base_color_image.moveaxis(-1, 0))
    specular_image = downsize(cam_info.specular_image.moveaxis(-1, 0))
    brdf_image = downsize(cam_info.brdf_image.moveaxis(-1, 0))
    gt_image = downsize(cam_info.image.moveaxis(-1, 0))
    loaded_mask = None

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        diffuse_image=diffuse_image,
        glossy_image=glossy_image,
        position_image=position_image,
        normal_image=normal_image,
        roughness_image=roughness_image,
        metalness_image=metalness_image,
        base_color_image=base_color_image,
        brdf_image=brdf_image,
        specular_image=specular_image,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
