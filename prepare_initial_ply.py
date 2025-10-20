import argparse
import os
from dataclasses import dataclass

import torch
import tyro
from tqdm import tqdm

from gaussian_tracing.cfg import ModelParams
from gaussian_tracing.scene.dataset.colmap_parser import ColmapParser
from gaussian_tracing.scene.dataset_readers import get_dataset
from gaussian_tracing.scene.tonemapping import tonemap
from gaussian_tracing.utils.ply_utils import save_ply


@dataclass
class Config:
    # Path to scene
    source_path: str = "data/renders/shiny_kitchen"
    # Path to output
    output_dir: str = "output/plys/shiny_kitchen"
    # Scale to bin
    scale: float = 50.0
    # Resolution
    resolution: int = 128


def main(cfg: Config):
    if not os.path.isdir(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)

    model_params = ModelParams(parser=argparse.ArgumentParser())
    model_params.resolution = cfg.resolution
    colmap_parser = ColmapParser(cfg.source_path)
    print("SFM Point Cloud:", colmap_parser.points.shape)
    ply_path = os.path.join(cfg.output_dir, "point_cloud_sfm.ply")
    save_ply(ply_path, colmap_parser.points, colmap_parser.points_rgb)

    dataset = get_dataset(model_params, cfg.source_path, split="train")
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
        points = cam_info.position_image.reshape(-1, 3)
        colors = tonemap(cam_info.diffuse_image.reshape(-1, 3))

        points_all.append(points.cpu())
        colors_all.append(colors.cpu())

    # Stack everything into tensors
    points = torch.cat(points_all)  # (N, 3)
    colors = torch.cat(colors_all)  # (N, 3)

    # Compute voxel indices
    voxel_coords = (points * cfg.scale).round().int()

    # Find unique voxel positions
    unique_coords, inverse_indices, counts = torch.unique(voxel_coords, dim=0, return_inverse=True, return_counts=True)

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
    points_np = (unique_coords.float() / cfg.scale).numpy()  # (n, 3)
    colors_np = avg_colors.numpy()  # (n, 3)
    print("Dense Point Cloud:", points_np.shape)
    ply_path = os.path.join(cfg.output_dir, "point_cloud_dense.ply")
    save_ply(ply_path, points_np, colors_np)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
