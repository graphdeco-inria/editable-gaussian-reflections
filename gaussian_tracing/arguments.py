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

class ModelParams:
    def __init__(self):
        self.white_background = False
        self.data_device = "cuda"
        self.scene_extent_init_radius = 4.0
        self.scene_extent_multiplier = 5.0
        self.num_feat_per_gaussian_channel = 16

        self.raytracer_version = ""  # "build_v0.1_attached_brdf"

        self.max_images = 9999999

        self.min_opacity = 0.005
        self.min_weight = 0.1

        self.znear_scaledown = 0.8
        self.zfar_scaleup = 1.5

        self.transmittance_threshold = 0.01
        self.alpha_threshold = 0.005
        self.exp_power = 3

        self.no_bounces_until_iter = 3_000


class OptimizationParams:
    def __init__(self):
        self.xyz_lr_max_steps = 32_000

        # flat schedule
        self.xyz_lr_init = 0.00016
        self.xyz_lr_final = 0.0000016
        self.xyz_lr_delay_mult = 0.01

        self.timestretch = 0.25

        self.xyz_lr = 0.0025
        self.normal_lr = 0.0025
        self.roughness_lr = 0.0025
        self.f0_lr = 0.0025
        self.diffuse_lr = 0.005

        self.opacity_lr = 0.025  
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01

        self.scale_decay = 0.9999

        self.pruning_interval = 500

        self.beta_1 = 0.9
        self.beta_2 = 0.999  # important to be lower than 0.999


@dataclass
class TyroConfig:
    # Model params
    model_params: ModelParams = field(default_factory=lambda: ModelParams())
    # Optimization params
    opt_params: OptimizationParams = field(default_factory=lambda: OptimizationParams())

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
    # Checkpoint iterations
    checkpoint_iterations: list[int] = field(default_factory=lambda: [])
    # Start checkpoint
    start_checkpoint: str | None = None
    # Total iterations
    iterations: int = 32_000

    # * Dataset params
    source_path: Annotated[str, arg(aliases=["-s"])] = ""
    model_path: Annotated[str, arg(aliases=["-m"])] = ""
    resolution: int = 512
    eval: bool = False
    max_images: int | None = None
    do_depth_fit: bool = False

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

    # 
    disable_znear_densif_pruning: bool = False
