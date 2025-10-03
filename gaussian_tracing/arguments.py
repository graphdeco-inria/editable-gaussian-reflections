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
from dataclasses import dataclass, field
from typing import Annotated, List, Optional, Literal
from tyro.conf import arg



@dataclass
class TyroConfig:
    viewer: bool = False
    viewer_mode: str = "local"
    detect_anomaly: bool = False
    flip_camera: bool = False
    val_views: list[int] = field(default_factory=lambda: [75, 175])
    test_iterations: list[int] = field(
        default_factory=lambda: [4, 6_000, 12_000, 18_000, 24_000, 32_000]
    )
    save_iterations: list[int] = field(
        default_factory=lambda: [4, 6_000, 12_000, 18_000, 24_000, 32_000]
    )
    quiet: bool = False
    
    iterations: int = 32_000

    # * Dataset params
    source_path: Annotated[str, arg(aliases=["-s"])] = ""
    model_path: Annotated[str, arg(aliases=["-m"])] = ""
    resolution: int = 512
    eval: bool = False
    max_images: int | None = None
    do_depth_fit: bool = False

    # * Model params
    white_background: bool = False
    data_device: str = "cuda"
    scene_extent_init_radius: float = 4.0
    scene_extent_multiplier: float = 5.0
    num_feat_per_gaussian_channel: int = 16
    raytracer_version: str = ""  # "build_v0.1_attached_brdf"
    max_images: int = 9999999
    min_opacity: float = 0.005
    min_weight: float = 0.1
    disable_znear_densif_pruning: bool = False
    znear_scaledown: float = 0.8
    zfar_scaleup: float = 1.5
    transmittance_threshold: float = 0.01
    alpha_threshold: float = 0.005
    exp_power: int = 3
    no_bounces_until_iter: int = 3_000

    # * Render params
    iteration: int = -1
    spp: int = 128
    train_views: bool = False
    denoise: bool = True
    modes: list[Literal["regular", "env_rot_1", "env_move_1", "env_move_2"]] = field(
        default_factory=lambda: ["regular"]
    )
    skip_video: bool = False
    skip_save_frames: bool = False

    # * Init params
    init_num_pts: int = 100_000
    init_num_pts_farfield = 75_000
    init_opa: float = 0.1
    init_opa_farfield: float = 0.1
    init_scale: float = 0.1  # 1.0 for 3dgs, 0.1 for mcmc
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
    timestretch: float = 0.25
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
    pruning_interval: int = 500
    beta_1: float = 0.9
    beta_2: float = 0.999

