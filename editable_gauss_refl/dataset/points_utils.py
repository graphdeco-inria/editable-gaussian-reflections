import os

import numpy as np
import torch

from editable_gauss_refl.utils.graphics_utils import BasicPointCloud

from .colmap_loader import read_points3D_binary, read_points3D_text


def get_point_cloud(data_dir) -> BasicPointCloud:
    bin_path = os.path.join(data_dir, "sparse/0/points3D.bin")
    txt_path = os.path.join(data_dir, "sparse/0/points3D.txt")
    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except Exception:
        xyz, rgb, _ = read_points3D_text(txt_path)
    pcd = BasicPointCloud(
        points=xyz,
        colors=rgb / 255.0,
        normals=np.zeros_like(xyz),
    )
    return pcd


def make_skybox(radius=1.0, num_skybox_pts=10_000):
    theta = 2.0 * torch.pi * torch.rand(num_skybox_pts)
    phi = torch.arccos(1.0 - 1.4 * torch.rand(num_skybox_pts))
    points = torch.zeros((num_skybox_pts, 3))
    points[:, 0] = radius * torch.cos(theta) * torch.sin(phi)
    points[:, 1] = radius * torch.sin(theta) * torch.sin(phi)
    points[:, 2] = radius * torch.cos(phi)

    colors = torch.ones_like(points) * 0.5
    return points.numpy(), colors.numpy()
