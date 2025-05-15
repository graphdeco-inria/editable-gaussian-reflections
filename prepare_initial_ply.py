import os
import argparse
from dataclasses import dataclass

import torch
import numpy as np
import tyro
from tqdm import tqdm
import plyfile
from collections import defaultdict

from arguments import ModelParams
from scene.dataset import ColmapDataset
from scene.dataset.points_utils import (
    get_point_cloud,
    make_skybox,
)
from scene.tonemapping import tonemap


@dataclass
class Config:
    # Path to scene
    source_path: str = "data/real_datasets_v2_filmic/neural_catacaustics_priors/compost"
    # Scale to bin
    scale: float = 50.0

def read_ply(data_dir):
    # Read the .ply file
    plydata = plyfile.PlyData.read(os.path.join(data_dir, 'point_cloud.ply'))
    vertex_data = plydata['vertex'].data
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
    colors = colors / 255.0
    print(points.shape)
    point_cloud = BasicPointCloud(
        points=points,
        colors=colors,
        normals=np.zeros_like(points),
    )
    extra_point_cloud = BasicPointCloud(
        points=np.zeros((0, 3)),
        colors=np.zeros((0, 3)),
        normals=np.zeros((0, 3)),
    )


def main(cfg: Config):
    model_params = ModelParams(parser=argparse.ArgumentParser())
    model_params.resolution = 256
    point_cloud = get_point_cloud(cfg.source_path)
    dataset = ColmapDataset(model_params, cfg.source_path, point_cloud)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x,
    )

    points_dict = defaultdict(int)
    colors_dict = defaultdict(lambda: torch.zeros(3))
    # normals_dict = defaultdict(lambda: torch.zeros(3))
    for idx, cam_info_batch in enumerate(tqdm(dataloader)):
        cam_info = cam_info_batch[0]
        if idx == 10:
            break

        points = cam_info.position_image.reshape(-1, 3)
        colors = tonemap(cam_info.diffuse_image.reshape(-1, 3))
        # normals = cam_info.normal_image.reshape(-1, 3)
        for point, color in zip(points, colors):
            point_round = (point * cfg.scale).round().int()
            point_hash = tuple(point_round.numpy().tolist())

            c = points_dict[point_hash]
            colors_dict[point_hash] = (colors_dict[point_hash] * c + color) / (c + 1)
            points_dict[point_hash] += 1

    # threshold = torch.quantile(points_dict.values(), 2)
    threshold = 0.0
    points_list = []
    colors_list = []
    for k in points_dict.keys():
        if points_dict[k] < threshold:
            continue
        points_list.append(k)
        colors_list.append(colors_dict[k])
    points = np.array(points_list) / cfg.scale
    colors = (np.array(colors_list) * 255.0).astype(np.uint8)
    print(points.shape)

    # Create structured array
    vertex = np.array(
        [(*point, *color) for point, color in zip(points, colors)],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    # Create a PlyElement
    ply_element = plyfile.PlyElement.describe(vertex, "vertex")

    # Write to PLY file
    ply_path = os.path.join(cfg.source_path, "point_cloud.ply")
    plyfile.PlyData([ply_element], text=True).write(ply_path)

    
if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
