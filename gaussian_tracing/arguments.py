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
from typing import Literal

# DIFFUSE_LOSS_WEIGHT
# REFLECTION_LOSS_WEIGHT
# NORMAL_LOSS_WEIGHT
# POSITION_LOSS_WEIGHT
# BRDF_PARAMS_LOSS_WEIGHT


class ModelParams:
    def __init__(self):
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.scene_extent_init_radius = 4.0
        self.scene_extent_multiplier = 5.0
        self.num_feat_per_gaussian_channel = 16

        self.brdf_mode: Literal["disabled", "gt", "static_lut"] = (
            "gt"  # "finetuned_lut" is legacy, no longer works
        )
        self.use_attached_brdf = False
        self.detach_normals = False
        self.detach_position = False
        self.detach_roughness = False
        self.detach_F0 = False

        self.use_masks = False
        self.precomp_ray = False

        self.raytracer_version = ""  # "build_v0.1_attached_brdf"

        self.disable_bounce_grads = False

        self.keep_every_kth_view = 1
        self.max_images = 9999999
        self.num_farfield_init_points = 75_000

        self.min_opacity = 0.005
        self.min_weight = 0.1

        self.znear_densif_pruning = "REAL_SCENE" not in os.environ
        self.znear_scaledown = 0.8
        self.zfar_scaleup = 1.5

        self.force_mcmc_custom_init = False
        self.downsampling_mode = "area"

        self.raytrace_primal = False

        self.f0_decay = False

        self.opacity_pruning_threshold = 0.005  # 0.051 #
        self.cap_max = -1  # for mcmc

        self.use_opacity_resets = False
        self.init_scale_factor = 1.0  # 1.0 for 3dgs, 0.1 for mcmc
        if "REAL_SCENE" in os.environ:
            self.init_scale_factor = 0.1
        self.init_scale_factor_farfield = 0.1
        self.init_opacity = 0.1  # 0.1 for 3dgs, 0.5 for mcmc
        self.init_opacity_farfield = 0.1
        self.init_roughness = 0.1
        self.init_f0 = 0.04
        self.init_extra_point_diffuse = 0.2

        self.warmup_diffuse_loss_weight = 10000.0
        self.diffuse_loss_weight = 5.0
        self.glossy_loss_weight = 3.0
        if "LEGACY_WEIGHT" in os.environ:
            self.glossy_loss_weight = 0.001
        self.normal_loss_weight = 2.5
        self.position_loss_weight = 2.5
        if "REAL_SCENE" in os.environ:
            self.position_loss_weight = 0.0
        self.f0_loss_weight = 1.0
        self.roughness_loss_weight = 1.0
        self.specular_loss_weight = 1.0
        self.albedo_loss_weight = 1.0
        self.metalness_loss_weight = 1.0
        self.regular_loss_weight = 0.0

        if "ONLY_DIFFUSE_LOSS" in os.environ:
            self.diffuse_loss_weight = 5.0
            self.glossy_loss_weight = 0.0
            self.normal_loss_weight = 0.0
            self.position_loss_weight = 0.0
            self.f0_loss_weight = 0.0
            self.roughness_loss_weight = 0.0
            self.specular_loss_weight = 0.0
            self.albedo_loss_weight = 0.0
            self.metalness_loss_weight = 0.0

        # level of detail args below
        self.lod_prob_blur_targets = 1.0
        self.lod_init_scale = 0.005
        self.lod_init_frac_extra_points = 0.25
        self.lod_clamp_minsize = True
        self.lod_clamp_minsize_factor = 1.0
        self.lod_max_world_size_blur = 0.05
        self.lod_force_blur_sigma = -1.0
        self.lod_schedule_power = 1.0

        self.use_diffuse_target = False
        self.use_glossy_target = False

        self.sparseness = -1
        self.hard_sparse = -1

        self.warmup_until_iter = 0
        if "LEGACY_SCHEDULE" in os.environ:
            self.no_bounces_until_iter = 6_000
            self.max_one_bounce_until_iter = 12_000
        else:
            self.no_bounces_until_iter = 3_000
            self.max_one_bounce_until_iter = -1

        self.diffuse_loss_weight_after_rebalance = 5.0
        self.glossy_loss_weight_after_rebalance = 5.0
        self.rebalance_losses_at_iter = -1
        if "REAL_SCENE" in os.environ:
            self.rebalance_losses_at_iter = -1
        if "LEGACY_WEIGHT" in os.environ:
            self.rebalance_losses_at_iter = 18_000
        self.enable_regular_loss_at_iter = -1

        self.skip_n_images = 0


class PipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.depth_ratio = 0.0


class OptimizationParams:
    def __init__(self):
        self.position_lr_max_steps = 32_000

        # flat schedule
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01

        self.timestretch = 0.25

        self.normal_lr = 0.0025
        self.position_lr = 0.0025
        self.roughness_lr = 0.0025
        self.f0_lr = 0.0025
        self.diffuse_lr = 0.005

        self.lod_mean_lr = 0.005 / 100
        self.lod_scale_lr = 0.005 / 100 * 5

        self.opacity_lr = 0.025  #! was 0.05 which does not match 3dgs
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self._brdf_lut_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05

        self.noise_lr = 5e5
        self.scale_reg = 0.0  # 0.01 in mcmc
        self.opacity_reg = 0.0  # 0.01 in mcmc

        self.scale_decay = 0.9999
        self.opacity_decay = 1.0
        self.lod_mean_decay = 1.0
        self.lod_scale_decay = 1.0

        self.densif_scaledown_clones = False
        self.densif_jitter_clones = False

        self.densification_interval = 500
        self.opacity_reset_interval = (
            999999999999  # Doesn't change metrics in 3dgs, may cause issues
        )
        self.densify_from_iter = 999999999999  # was 500 in 3dgs, 1500 when doing LOD
        self.densify_until_iter = 15_000  # was 25k in mcmc
        self.densify_grad_threshold = 0.0002

        self.densif_use_top_k = True
        self.densif_final_num_gaussians = 800_000
        self.densif_size_ranking_weight = 0.0
        self.densif_opacity_ranking_weight = 0.0
        self.densif_lod_ranking_weight = 0.0
        self.densif_no_pruning_large_radii = False
        self.densif_use_fixed_split_clone_ratio = True
        self.densif_split_clone_ratio = 0.5
        self.densif_num_gaussians_per_step = 1_000

        self.prune_even_without_densification = True

        self.densif_no_pruning = False
        self.densif_no_cloning = False
        self.densif_no_splitting = True
        self.densif_pruning_only = False

        self.densif_skip_big_points_ws = False

        self.sh_slowdown_factor = 20.0
        self.random_background = False

        self.beta_1 = 0.9
        self.beta_2 = 0.999  # important to be lower than 0.999


@dataclass
class TyroConfig:
    # Model params
    model_params: ModelParams = field(default_factory=lambda: ModelParams())
    # Optimization params
    opt_params: OptimizationParams = field(default_factory=lambda: OptimizationParams())
    # Pipeline params
    pipe_params: PipelineParams = field(default_factory=lambda: PipelineParams())

    # IP address
    ip: str = "127.0.0.1"
    # Port
    port: int = 8000
    # Enable viewer
    viewer: bool = False
    # Viewer mode
    viewer_mode: str = "local"
    # Detect anomaly
    detect_anomaly: bool = False
    # Flip camera
    flip_camera: bool = False
    # Validation views
    val_views: list[int] = field(default_factory=lambda: [75, 175])
    # Test iterations
    test_iterations: list[int] = field(
        default_factory=lambda: [4, 6_000, 12_000, 18_000, 24_000, 32_000]
    )
    # Save iterations
    save_iterations: list[int] = field(
        default_factory=lambda: [4, 6_000, 12_000, 18_000, 24_000, 32_000]
    )
    # Quiet
    quiet: bool = False
    # Checkpoint iterations
    checkpoint_iterations: list[int] = field(default_factory=lambda: [])
    # Start checkpoint
    start_checkpoint: str | None = None
    # Total iterations
    iterations: int = 32_000

    # Source path
    source_path: str = ""
    # Model path
    model_path: str = ""
    # Resolution
    resolution: int = 512
    # Initial scale factor
    init_scale_factor: float = 0.1
    # Evaluation
    eval: bool = False
    # Max images
    max_images: int | None = None

    # Iteration
    iteration: int = -1
    # Maximum number of bounces
    max_bounces: int = -1
    # Samples per pixel
    spp: int = 128
    # Supersampling
    supersampling: int = 1
    # Use train views
    train_views: bool = False
    # Skip denoiser
    skip_denoiser: bool = False
    # Rendering modes
    modes: list[str] = field(
        default_factory=lambda: ["regular", "env_rot_1", "env_move_1", "env_move_2"]
    )
    # Blur sigmas [None, 4.0, 16.0]
    blur_sigmas: list[float | None] = field(default_factory=lambda: [None])
    # Skip video
    skip_video: bool = False
    # Red region
    red_region: bool = False
    # Skip save frames
    skip_save_frames: bool = False
