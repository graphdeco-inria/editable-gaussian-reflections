import argparse
import os
from dataclasses import dataclass
import numpy as np 

import torch
import tyro
from tyro.conf import arg
from tqdm import tqdm
from typing import Optional, Annotated, Literal

from editable_gauss_refl.dataset.colmap_parser import ColmapParser
from editable_gauss_refl.scene.dataset_readers import get_dataset
from editable_gauss_refl.utils.tonemapping import untonemap
from editable_gauss_refl.utils.ply_utils import save_ply

from editable_gauss_refl.utils.general_utils import set_seeds
from editable_gauss_refl.utils.depth_utils import compute_primary_ray_directions
import json


@dataclass
class PrepareInitialPLYCLI:
    source_path: Annotated[str, arg(aliases=["-s"])] # * e.g. "data/renders/shiny_kitchen"
    mode: Literal["sfm", "dense"] = "dense"
    filename: str = "point_cloud_{mode}.ply"

    voxel_scale: float = 400.0
    resolution: int = 128
    
    max_images: Optional[int] = None


if __name__ == "__main__":
    cli = tyro.cli(PrepareInitialPLYCLI)

    set_seeds()

    if cli.mode == "sfm":
        colmap_parser = ColmapParser(cli.source_path)
        print("SFM Point Cloud:", colmap_parser.points.shape)
        ply_path = os.path.join(cli.source_path, cli.filename.format(mode="sfm"))
        # * Deliberately avoid invert tonemapping since it struggles with very dark colors leading to bright spots/floaters
        save_ply(ply_path, colmap_parser.points, colmap_parser.points_rgb / 255.0)
    else:
        dataset = get_dataset(cli, cli.source_path, split="train")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4 if "NO_WORKERS" not in os.environ else 0,
            collate_fn=lambda x: x,
        )

        points_all = []
        colors_all = []

        for idx, cam_info_batch in enumerate(tqdm(dataloader)):
            cam_info = cam_info_batch[0]
            depth = cam_info.depth_image
            R_blender = -torch.from_numpy(cam_info.R).clone()
            R_blender[:, 0] = -R_blender[:, 0]

            ray_dirs = compute_primary_ray_directions(
                cam_info.depth_image.shape[0],
                cam_info.depth_image.shape[1],
                cam_info.FovY,
                R_blender[:3, :3],
            )
            origin = torch.from_numpy(-cam_info.R @ cam_info.T)
            positions = origin + ray_dirs * depth
            points = positions.reshape(-1, 3)

            colors = cam_info.diffuse_image.reshape(-1, 3)
            if cam_info.diffuse_image.dtype == torch.uint8:
                colors = untonemap(colors.float() / 255.0)

            points_all.append(points.cpu())
            colors_all.append(colors.cpu())

        # Stack everything into tensors
        points = torch.cat(points_all)  # (N, 3)
        colors = torch.cat(colors_all)  # (N, 3)

        # Compute voxel indices
        voxel_coords = (points * cli.voxel_scale).round().int()

        # Find unique voxel positions
        unique_coords, inverse_indices, counts = torch.unique(voxel_coords, dim=0, return_inverse=True, return_counts=True)

        # Accumulate color contributions per voxel
        accum_colors = torch.zeros((unique_coords.shape[0], 3), dtype=colors.dtype)
        accum_colors.index_add_(0, inverse_indices, colors)

        # Average colors per voxel
        avg_colors = accum_colors / counts.unsqueeze(1)

        # Select valid voxels
        # threshold = torch.quantile(counts.float(), 0.02)
        # print("Percentile threshold:", threshold.item())
        mask = counts >= 2.0
        unique_coords = unique_coords[mask]
        avg_colors = avg_colors[mask]

        # Extract points and colors
        points_np = (unique_coords.float() / cli.voxel_scale).numpy()  # (n, 3)
        colors_np = avg_colors.numpy()  # (n, 3)
        print("Dense Point Cloud:", points_np.shape)
        ply_path = os.path.join(cli.source_path, cli.filename.format(mode="dense"))
        save_ply(ply_path, points_np, colors_np)
