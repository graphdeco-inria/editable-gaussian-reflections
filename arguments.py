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

from argparse import ArgumentParser, Namespace
import sys
import os
from typing import *


# DIFFUSE_LOSS_WEIGHT
# REFLECTION_LOSS_WEIGHT
# NORMAL_LOSS_WEIGHT
# POSITION_LOSS_WEIGHT
# BRDF_PARAMS_LOSS_WEIGHT 

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + ("no_" + key if value else key), ("-" + key[0:1]), action="store_false" if value else "store_true", dest=key)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + ("no_" + key if value else key),  action="store_false" if value else "store_true", dest=key)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = 1536
        
        # // 2 ## Yohan: on quarter res this crashes at 600 iters and I don't know why => this is because this gives the width, not the height! so it results in non-pow2 sizes for the height
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.glossy_bbox_size_mult = 8.0
        self.scene_extent_multiplier = 5.0
        self.num_feat_per_gaussian_channel = 16 
         
        self.brdf_mode: Literal["disabled", "gt", "static_lut"] = "gt"  # "finetuned_lut" is legacy, no longer works
        self.use_attached_brdf = False
        self.detach_normals = False 
        self.detach_position = False
        self.detach_roughness = False
        self.detach_F0 = False

        self.use_masks = False
        self.precomp_ray = False

        self.raytracer_version = "build" #"build_v0.1_attached_brdf"

        self.disable_bounce_grads = False

        self.keep_every_kth_view = 1
        self.max_images = 9999999
        self.num_farfield_init_points = 50_000 # 100_000

        self.min_opacity = 0.005

        self.znear_init_pruning = True
        self.znear_densif_pruning = True
        self.znear_scaledown = 0.8
        self.zfar_scaleup = 1.5

        self.force_mcmc_custom_init = False
        self.add_mcmc_noise = False

        self.downsampling_mode = "area"

        self.linear_space = True
        self.exposure = 1.0
        self.raytrace_primal = False

        self.opacity_pruning_threshold = 0.005 # 0.051 # 
        self.cap_max = -1 # for mcmc

        self.use_opacity_resets = False

        self.init_scale_factor = 1.0 # 1.0 for 3dgs, 0.1 for mcmc 
        self.init_opacity = 0.1 # 0.1 for 3dgs, 0.5 for mcmc

        self.diffuse_loss_weight = 1.0
        self.glossy_loss_weight = 0.0001 # ! was 0.001
        self.normal_loss_weight = 1.0
        self.position_loss_weight = 1.0
        self.f0_loss_weight = 1.0
        self.roughness_loss_weight = 1.0
        self.specular_loss_weight = 1.0
        self.albedo_loss_weight = 1.0
        self.metalness_loss_weight = 1.0

        if "DIFFUSE_IS_RENDER" in os.environ:
            self.diffuse_loss_weight = 1.0
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
        self.no_bounces_until_iter = 3_000
        self.max_one_bounce_until_iter = 7_000
        self.diffuse_loss_weight_after_rebalance = 1.0
        self.glossy_loss_weight_after_rebalance = 1.0
        self.rebalance_losses_at_iter = 15000
        self.enable_regular_loss_at_iter = -1

        self.num_samples = 1

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.depth_ratio = 0.0
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.slowdown = 1
        
        self.normal_lr = 0.0025
        self.position_lr = 0.0025
        self.roughness_lr = 0.0025
        self.f0_lr = 0.0025
        self.feature_lr = 0.0025

        self.lod_mean_lr = 0.005 / 100
        self.lod_scale_lr = 0.005 / 100 * 5

        self.opacity_lr = 0.025 #! was 0.05 which does not match 3dgs
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self._brdf_lut_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05
        
        self.noise_lr = 5e5
        self.scale_reg = 0.0 # 0.01 in mcmc
        self.opacity_reg = 0.0 # 0.01 in mcmc

        self.scale_decay = 1.0 
        self.opacity_decay = 1.0 
        self.lod_mean_decay = 1.0
        self.lod_scale_decay = 1.0

        self.densif_scaledown_clones = False
        self.densif_jitter_clones = False

        self.densification_interval = 100 # # was 100 in 3dgs, 500 to fix LOD densification
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500 # was 500 in 3dgs, 1500 when doing LOD
        self.densify_until_iter = 15_000 # was 25k in mcmc
        self.densify_grad_threshold = 0.0002

        self.densif_use_top_k = True
        self.densif_final_num_gaussians = 500_000 #! 1M
        self.densif_size_ranking_weight = 0.0
        self.densif_opacity_ranking_weight = 0.0
        self.densif_lod_ranking_weight = 0.0
        self.densif_no_pruning_large_radii = False
        self.densif_use_fixed_split_clone_ratio = True
        self.densif_split_clone_ratio = 0.5

        self.densif_no_pruning = False
        self.densif_no_cloning = False
        self.densif_no_splitting = True 
        self.densif_pruning_only = False

        self.densif_skip_big_points_ws = False

        self.sh_slowdown_factor = 20.0
        self.random_background = False

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "model_params")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
