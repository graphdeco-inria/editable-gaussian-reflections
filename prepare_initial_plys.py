import os

import numpy as np
import torch
import tyro
from tqdm import tqdm

from gaussian_tracing.arguments import TyroConfig
from gaussian_tracing.dataset.colmap_parser import ColmapParser
from gaussian_tracing.scene.dataset_readers import get_dataset
from gaussian_tracing.utils.depth_utils import (
    transform_distance_to_position_image,
    transform_points,
)
from gaussian_tracing.utils.ply_utils import save_ply
from gaussian_tracing.utils.tonemapping import tonemap


def main(cfg: TyroConfig):
    colmap_parser = ColmapParser(cfg.source_path)
    print("SFM Point Cloud:", colmap_parser.points.shape)
    ply_path = os.path.join(cfg.source_path, "point_cloud_sfm.ply")
    save_ply(ply_path, colmap_parser.points, colmap_parser.points_rgb)

    dataset = get_dataset(cfg, cfg.source_path, split="train")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x,
    )

    points_all = []
    colors_all = []

    for idx, cam_info_batch in enumerate(tqdm(dataloader)):
        cam_info = cam_info_batch[0]
        w2c = np.eye(4)
        w2c[:3, :3] = np.transpose(cam_info.R)
        w2c[:3, 3] = cam_info.T
        c2w = np.linalg.inv(w2c)

        distance_image = cam_info.depth_image
        position_image = transform_distance_to_position_image(
            distance_image, cam_info.FovX, cam_info.FovY
        )
        points = position_image.reshape(-1, 3)
        points = transform_points(points, c2w)
        colors = tonemap(cam_info.diffuse_image.reshape(-1, 3))

        points_all.append(points.cpu())
        colors_all.append(colors.cpu())

    # Stack everything into tensors
    points = torch.cat(points_all)  # (N, 3)
    colors = torch.cat(colors_all)  # (N, 3)

    # Compute voxel indices
    voxel_coords = (points * cfg.voxel_scale).round().int()

    # Find unique voxel positions
    unique_coords, inverse_indices, counts = torch.unique(
        voxel_coords, dim=0, return_inverse=True, return_counts=True
    )

    # Accumulate color contributions per voxel
    accum_colors = torch.zeros((unique_coords.shape[0], 3), dtype=colors.dtype)
    accum_colors.index_add_(0, inverse_indices, colors)

    # Average colors per voxel
    avg_colors = accum_colors / counts.unsqueeze(1)

    # Select top x% densest voxels
    # threshold = torch.quantile(counts.float(), 0.02)
    # print("Percentile threshold:", threshold.item())
    mask = counts >= 2.0
    unique_coords = unique_coords[mask]
    avg_colors = avg_colors[mask]

    # Extract points and colors
    points_np = (unique_coords.float() / cfg.voxel_scale).numpy()  # (n, 3)
    colors_np = avg_colors.numpy()  # (n, 3)
    print("Dense Point Cloud:", points_np.shape)
    ply_path = os.path.join(cfg.source_path, "point_cloud_dense.ply")
    save_ply(ply_path, points_np, colors_np)


if __name__ == "__main__":
    cfg = tyro.cli(TyroConfig)
    main(cfg)
