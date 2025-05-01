import os

import numpy as np

from scene.gaussian_model import BasicPointCloud

from .colmap_loader import read_points3D_binary, read_points3D_text


def get_point_cloud(data_dir) -> BasicPointCloud:
    bin_path = os.path.join(data_dir, "sparse/0/points3D.bin")
    txt_path = os.path.join(data_dir, "sparse/0/points3D.txt")
    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
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
