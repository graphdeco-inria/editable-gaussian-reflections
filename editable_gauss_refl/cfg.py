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

from dataclasses import dataclass, field
from typing import Annotated

from tyro.conf import arg


@dataclass
class Config:
    viewer: bool = False
    viewer_mode: str = "local"
    val_view: int = 75
    test_iterations: list[int] = field(default_factory=lambda: [1, 750, 3000, 8000])
    save_iterations: list[int] = field(default_factory=lambda: [1, 750, 3000, 8000])
    iterations: int = 8000

    # * Dataset params
    source_path: Annotated[str, arg(aliases=["-s"])] = ""
    model_path: Annotated[str, arg(aliases=["-m"])] = ""
    resolution: Annotated[int, arg(aliases=["-r"])] = 768
    eval: bool = False
    max_images: int | None = None
    do_depth_fit: bool = False

    # * Model params
    white_background: bool = False
    data_device: str = "cuda"
    scene_extent_init_radius: float = 4.0
    scene_extent_multiplier: float = 5.0
    num_feat_per_gaussian_channel: int = 16
    max_images: int = 9999999
    min_opacity: float = 0.005
    min_weight: float = 0.1
    disable_znear_densif_pruning: bool = False
    znear_scaledown: float = 0.8
    zfar_scaleup: float = 1.5
    transmittance_threshold: float = 0.01
    alpha_threshold: float = 0.005
    exp_power: int = 3
    no_bounces_until_iter: int = 750

    # * Init params
    init_num_pts: int = 100_000
    init_num_pts_farfield = 75_000
    init_opa: float = 0.1
    init_opa_farfield: float = 0.1
    init_scale: float = 1.0
    init_scale_farfield: float = 0.1
    init_roughness: float = 0.1
    init_f0: float = 0.04
    init_diffuse_farfield: float = 0.2

    # * Loss weights
    loss_weight_diffuse: float = 5.0
    loss_weight_glossy: float = 3.0
    loss_weight_depth: float = 2.5
    loss_weight_normal: float = 2.5
    loss_weight_f0: float = 1.0
    loss_weight_roughness: float = 1.0

    # * Optimization params
    xyz_lr_max_steps: int = 32_000
    xyz_lr_init: float = 0.00016
    xyz_lr_final: float = 0.0000016
    xyz_lr_delay_mult: float = 0.01
    xyz_lr: float = 0.0025
    normal_lr: float = 0.0025
    roughness_lr: float = 0.0025
    f0_lr: float = 0.0025
    diffuse_lr: float = 0.005
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    scale_decay: float = 0.9999
    pruning_interval: int = 125
    pruning_start_iter: int = 1250
    beta_1: float = 0.9
    beta_2: float = 0.999
