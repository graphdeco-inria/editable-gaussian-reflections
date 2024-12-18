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
        self._resolution = 768
        
        # // 2 ## Yohan: on quarter res this crashes at 600 iters and I don't know why => this is because this gives the width, not the height! so it results in non-pow2 sizes for the height
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.glossy_bbox_size_mult = 4.0
        self.num_feat_per_gaussian_channel = 16 
        
        self.brdf_mode: Literal["disabled", "gt", "static_lut", "finetuned_lut"] = "gt"
        self.use_attached_brdf = False
        self.detach_normals = False 
        self.detach_position = False
        self.detach_roughness = False
        self.detach_F0 = False

        self.use_masks = False
        self.precomp_ray = False

        self.raytracer_version = "build" #"build_v0.1_attached_brdf"

        self.disable_bounce_grads = False

        self.freeze_brdf = False
        self.optimize_roughness = False
        self.freeze_optimize_roughness = False
        self.optimize_position = False
        self.optimize_normals = False
        self.freeze_optimize_normals = False
        self.gaussian_subsets = False
        self.keep_every_kth_view = 1
        self.max_images = 9999999
        self.num_init_points = 100_000 # 100_000
        self.opacity_modulation = False

        self.mcmc_densify = False
        self.mcmc_densify_disable_custom_init = False
        self.mcmc_skip_relocate = False
        self.force_mcmc_custom_init = False

        self.warmup = -1

        self.remap_position = False 
        # self.aux_randn_init = False

        self.ray_offset = 0.0 

        self.num_bounces = 1
        self.random_pool_props = False
        self.downsampling_mode = "nearest" #area

        self.linear_space = True
        self.exposure = 5 # note: use 10 for one bounce image


        self.raytrace_primal = False

        self.opacity_pruning_threshold = 0.005 # 0.051 # 
        self.cap_max = -1 # for mcmc

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
        self.normal_lr = 0.0025
        self.position_lr = 0.0025
        self.brdf_params_lr = 0.0025
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self._brdf_lut_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05
        
        self.noise_lr = 5e5
        self.scale_reg = 0.01
        self.opacity_reg = 0.01

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25_000 # was 15k, increased for mcmc
        self.densify_grad_threshold = 0.0002

        self.sh_slowdown_factor = 20.0
        self.random_background = False

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
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
