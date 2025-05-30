import os

import numpy as np
import torch

from gaussian_tracing.utils.graphics_utils import BasicPointCloud

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


def make_random_point_cloud(model_params) -> BasicPointCloud:
    num_rand_pts = model_params.num_farfield_init_points
    # glossy_bbox_size_mult = model_params.glossy_bbox_size_mult
    init_extra_point_diffuse = model_params.init_extra_point_diffuse

    print(f"Generating random point cloud ({num_rand_pts})...")
    rand_xyz = (
        np.random.random((num_rand_pts, 3)) * 2.6 - 1.3
    )  # * glossy_bbox_size_mult
    if "GRAY_EXTRA_POINTS" in os.environ:
        init_rgb = np.ones_like(rand_xyz) * init_extra_point_diffuse
    else:
        init_rgb = np.random.random((num_rand_pts, 3))
    pcd = BasicPointCloud(
        points=rand_xyz,
        colors=init_rgb,
        normals=np.zeros_like(rand_xyz),
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
