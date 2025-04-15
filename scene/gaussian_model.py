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

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from arguments import ModelParams
from scene.tonemapping import *
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.diffuse_activation = lambda x: x

        self.is_dirty = False # for viewer

    def __init__(self, model_params: ModelParams):
        self.model_params = model_params
        self._xyz = torch.empty(0)
        self._normal = torch.empty(0)
        self._position = torch.empty(0)
        self._roughness = torch.empty(0)
        self._f0 = torch.empty(0)
        self._diffuse = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._lod_mean = torch.empty(0)
        self._lod_scale = torch.empty(0)
        self._round_counter = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum_diffuse = torch.empty(0)
        self.xyz_gradient_accum_glossy = torch.empty(0)
        self.comes_from_colmap_pc = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.schedule = None

        self.num_densification_steps = 0

        if "lut" in self.model_params.brdf_mode:
            brdf_lut_path = "data/ibl_brdf_lut.png"
            brdf_lut = cv2.imread(brdf_lut_path)
            brdf_lut = cv2.cvtColor(brdf_lut, cv2.COLOR_BGR2RGB)
            brdf_lut = brdf_lut.astype(np.float32)
            brdf_lut /= 255.0
            brdf_lut = torch.tensor(brdf_lut).to("cuda")
            self._brdf_lut = brdf_lut.permute((2, 0, 1))
            if self.model_params.brdf_mode == "finetuned_lut":
                self._brdf_lut_residual = nn.Parameter(torch.zeros_like(self._brdf_lut))

    def capture(self):
        return (
            self._xyz,
            self._normal,
            self._position,
            self._roughness,
            self._f0,
            self._diffuse,
            self._scaling,
            self._rotation,
            self._opacity,
            self._lod_mean,
            self._lod_scale,
            self._round_counter,
            self.max_radii2D,
            self.xyz_gradient_accum_diffuse,
            self.xyz_gradient_accum_glossy,
            self.comes_from_colmap_pc,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._normal,
            self._position,
            self._roughness,
            self._f0,
            self._diffuse,
            self._scaling,
            self._rotation,
            self._opacity,
            self._lod_mean,
            self._lod_scale,
            self._round_counter,
            self.max_radii2D,
            xyz_gradient_accum_diffuse,
            xyz_gradient_accum_glossy,
            comes_from_colmap_pc,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum_diffuse = xyz_gradient_accum_diffuse
        self.xyz_gradient_accum_glossy = xyz_gradient_accum_glossy
        self.comes_from_colmap_pc = comes_from_colmap_pc
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

        self.init_empty_grads()

    def init_empty_grads(self):
        self._xyz.grad = torch.zeros_like(self._xyz)
        self._normal.grad = torch.zeros_like(self._normal)
        self._position.grad = torch.zeros_like(self._position)
        self._roughness.grad = torch.zeros_like(self._roughness)
        self._f0.grad = torch.zeros_like(self._f0)
        self._opacity.grad = torch.zeros_like(self._opacity)
        self._lod_mean.grad = torch.zeros_like(self._lod_mean)
        self._lod_scale.grad = torch.zeros_like(self._lod_scale)
        self._round_counter.grad = torch.zeros_like(self._round_counter)
        self._diffuse.grad = torch.zeros_like(self._diffuse)
        self._scaling.grad = torch.zeros_like(self._scaling)
        self._rotation.grad = torch.zeros_like(self._rotation)

    @property
    def get_brdf_lut(self):
        if self.model_params.brdf_mode == "finetuned_lut":
            return self._brdf_lut * torch.exp(self._brdf_lut_residual)
        else:
            assert self.model_params.brdf_mode == "static_lut"
            return self._brdf_lut

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def _get_scaling(self):
        return self._scaling # for override for the viewer
    
    @property
    def _get_rotation(self):
        return self._rotation # for override for the viewer

    @property
    def get_diffuse(self):
        return self._diffuse

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_normal(self):
        return self._normal

    @property
    def get_position(self):
        return self._position

    @property
    def get_roughness(self):
        return self._roughness

    @property
    def get_f0(self):
        return self._f0

    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            1 / self.get_scaling, 1 / scaling_modifier, self.get_rotation
        )

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_lod_mean(self):
        return self._lod_mean

    @property
    def get_lod_scale(self):
        return torch.exp(self._lod_scale)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(
        self,
        pcd: BasicPointCloud,
        spatial_lr_scale: float,
        fovY_radians: float,
        max_camera_zfar: float,
        raytracer_config,
    ):
        self.spatial_lr_scale = spatial_lr_scale

        if raytracer_config.USE_LEVEL_OF_DETAIL:
            init_N = len(pcd.points)
            num_new_points = int(
                self.model_params.lod_init_frac_extra_points * len(pcd.points)
            )
            indices = np.random.choice(len(pcd.points), num_new_points, replace=False)
            new_points = np.concatenate([pcd.points, pcd.points[indices]])
            new_colors = np.concatenate([pcd.colors, pcd.colors[indices]])
            new_normals = np.concatenate([pcd.normals, pcd.normals[indices]])
            pcd = BasicPointCloud(new_points, new_colors, new_normals)

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        fused_normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        if self.model_params.force_mcmc_custom_init:
            dist2 = torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            scales = torch.log(torch.sqrt(dist2) * 0.1)[..., None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            opacities = inverse_sigmoid(
                0.5
                * torch.ones(
                    (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
                )
            )
        else:
            dist2 = torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            scales = torch.log(torch.sqrt(dist2) * self.model_params.init_scale_factor)[
                ..., None
            ].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            opacities = inverse_sigmoid(
                self.model_params.init_opacity
                * torch.ones(
                    (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
                )
            )

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        self._position = nn.Parameter(fused_point_cloud.clone())
        self._normal = nn.Parameter(fused_normal.clone())
        self._roughness = nn.Parameter(
            torch.ones((fused_point_cloud.shape[0], 1), device="cuda") * self.model_params.init_roughness
        )
        self._f0 = nn.Parameter(
            torch.ones((fused_point_cloud.shape[0], 3), device="cuda") * self.model_params.init_f0
        )  
        if "SKIP_TONEMAPPING" not in os.environ:
            self._diffuse = nn.Parameter(untonemap(fused_color.clone()).clamp(0, 1))
        else:
            self._diffuse = nn.Parameter(fused_color.clone())
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self._lod_scale = nn.Parameter(
            torch.log(
                torch.ones((fused_point_cloud.shape[0], 1), device="cuda")
                * self.model_params.lod_init_scale
            )
        )
        self._lod_mean = nn.Parameter(
            torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        )
        self._round_counter = torch.zeros(
            (fused_point_cloud.shape[0], 1), device="cuda"
        )

        if (
            raytracer_config.USE_LEVEL_OF_DETAIL
            and "DISABLE_LOD_INIT" not in os.environ
        ):
            with torch.no_grad():
                self._lod_mean.copy_(
                    torch.distributions.Beta(1.5, 5.0).sample(
                        (self._lod_mean.shape[0],)
                    )[:, None]
                    * self.model_params.lod_max_world_size_blur
                )
                self._lod_mean[:init_N] = 0.0
            if self.model_params.lod_clamp_minsize:
                with torch.no_grad():
                    self._scaling.data.clamp_(
                        min=torch.log(
                            self._lod_mean * self.model_params.lod_clamp_minsize_factor
                        )
                    )

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._xyz.grad = torch.zeros_like(self._xyz)
        self._position.grad = torch.zeros_like(self._position)
        self._normal.grad = torch.zeros_like(self._normal)
        self._roughness.grad = torch.zeros_like(self._roughness)
        self._f0.grad = torch.zeros_like(self._f0)
        self._opacity.grad = torch.zeros_like(self._opacity)
        self._lod_mean.grad = torch.zeros_like(self._lod_mean)
        self._lod_scale.grad = torch.zeros_like(self._lod_scale)
        self._diffuse.grad = torch.zeros_like(self._diffuse)
        self._scaling.grad = torch.zeros_like(self._scaling)
        self._rotation.grad = torch.zeros_like(self._rotation)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum_diffuse = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_glossy = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.comes_from_colmap_pc = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") #!!!!! mark properly
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self._xyz.grad = torch.zeros_like(self._xyz)
        self._position.grad = torch.zeros_like(self._position)
        self._normal.grad = torch.zeros_like(self._normal)
        self._roughness.grad = torch.zeros_like(self._roughness)
        self._f0.grad = torch.zeros_like(self._f0)
        self._opacity.grad = torch.zeros_like(self._opacity)
        self._lod_mean.grad = torch.zeros_like(self._lod_mean)
        self._lod_scale.grad = torch.zeros_like(self._lod_scale)
        self._diffuse.grad = torch.zeros_like(self._diffuse)
        self._scaling.grad = torch.zeros_like(self._scaling)
        self._rotation.grad = torch.zeros_like(self._rotation)

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._position],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "position",
            },
            {"params": [self._normal], "lr": training_args.normal_lr, "name": "normal"},
            {
                "params": [self._roughness],
                "lr": training_args.roughness_lr,
                "name": "roughness",
            },
            {"params": [self._f0], "lr": training_args.f0_lr, "name": "f0"},
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {"params": [self._diffuse], "lr": training_args.feature_lr, "name": "f_dc"},
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self._lod_mean],
                "lr": training_args.lod_mean_lr,
                "name": "lod_mean",
            },
            {
                "params": [self._lod_scale],
                "lr": training_args.lod_scale_lr,
                "name": "lod_scale",
            },
        ]

        if self.model_params.brdf_mode == "finetuned_lut":
            l.append(
                {
                    "params": [self._brdf_lut_residual],
                    "lr": training_args.brdf_lut_lr,
                    "name": "brdf_lut_residual",
                }
            )
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=(training_args.beta_1, training_args.beta_2))
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._diffuse.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        lod_mean = self._lod_mean.detach().cpu().numpy()
        lod_scale = self._lod_scale.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        position = self._position.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        f0 = self._f0.detach().cpu().numpy()

        all_attributes = [
            "x",
            "y",
            "z",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "pos_0",
            "pos_1",
            "pos_2",
            "normal_0",
            "normal_1",
            "normal_2",
            "roughness",
            "f0_0",
            "f0_1",
            "f0_2",
            "lod_mean",
            "lod_scale",
        ]
        dtype_full = [(attribute, "f4") for attribute in all_attributes]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                f_dc,
                opacities,
                scale,
                rotation,
                position,
                normal,
                roughness,
                f0,
                lod_mean,
                lod_scale,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        if self.model_params.brdf_mode == "finetuned_lut":
            brdf_lut_path = path.replace("point_cloud.ply", "brdf_lut_residuals.pt")
            torch.save(self._brdf_lut_residual, brdf_lut_path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.1)
        )  #! was 0.01
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        self._opacity.grad = torch.zeros_like(self._opacity)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        diffuse = np.zeros((xyz.shape[0], 3))
        diffuse[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        diffuse[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        diffuse[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        pos_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("pos")
        ]
        pos_names = sorted(pos_names, key=lambda x: int(x.split("_")[-1]))
        positions = np.zeros((xyz.shape[0], len(pos_names)))
        for idx, attr_name in enumerate(pos_names):
            positions[:, idx] = np.asarray(plydata.elements[0][attr_name])

        normal_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("normal")
        ]
        normal_names = sorted(normal_names, key=lambda x: int(x.split("_")[-1]))
        normals = np.zeros((xyz.shape[0], len(normal_names)))
        for idx, attr_name in enumerate(normal_names):
            normals[:, idx] = np.asarray(plydata.elements[0][attr_name])

        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

        f0_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f0")
        ]
        f0_names = sorted(f0_names, key=lambda x: int(x.split("_")[-1]))
        f0 = np.zeros((xyz.shape[0], len(f0_names)))
        for idx, attr_name in enumerate(f0_names):
            f0[:, idx] = np.asarray(plydata.elements[0][attr_name])

        lod_mean = np.asarray(plydata.elements[0]["lod_mean"])[..., np.newaxis]
        lod_scale = np.asarray(plydata.elements[0]["lod_scale"])[..., np.newaxis]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._diffuse = nn.Parameter(
            torch.tensor(diffuse, dtype=torch.float, device="cuda")
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda")
        )
        self._lod_mean = nn.Parameter(torch.zeros((xyz.shape[0], 1), device="cuda"))
        self._lod_scale = nn.Parameter(torch.zeros((xyz.shape[0], 1), device="cuda"))
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda")
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda")
        )
        self._position = nn.Parameter(
            torch.tensor(positions, dtype=torch.float, device="cuda")
        )
        self._normal = nn.Parameter(
            torch.tensor(normals, dtype=torch.float, device="cuda")
        )
        self._roughness = nn.Parameter(
            torch.tensor(roughness, dtype=torch.float, device="cuda")
        )
        self._f0 = nn.Parameter(torch.tensor(f0, dtype=torch.float, device="cuda"))

        self._round_counter = torch.zeros(
            (xyz.shape[0], 1), dtype=torch.float, device="cuda"
        )

        if self.model_params.brdf_mode == "finetuned_lut":
            self._brdf_lut_residual = nn.Parameter(
                torch.load(path.replace("point_cloud.ply", "brdf_lut_residuals.pt")).to(
                    "cuda"
                )
            )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor)
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "brdf_lut_residual":
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask]))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask])
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._xyz.grad = torch.zeros_like(self._xyz)
        #
        self._position = optimizable_tensors["position"]
        self._position.grad = torch.zeros_like(self._position)
        #
        self._normal = optimizable_tensors["normal"]
        self._normal.grad = torch.zeros_like(self._normal)
        #
        self._roughness = optimizable_tensors["roughness"]
        self._roughness.grad = torch.zeros_like(self._roughness)
        #
        self._f0 = optimizable_tensors["f0"]
        self._f0.grad = torch.zeros_like(self._f0)
        #
        self._diffuse = optimizable_tensors["f_dc"]
        self._diffuse.grad = torch.zeros_like(self._diffuse)
        #
        self._opacity = optimizable_tensors["opacity"]
        self._opacity.grad = torch.zeros_like(self._opacity)
        #
        self._lod_mean = optimizable_tensors["lod_mean"]
        self._lod_mean.grad = torch.zeros_like(self._lod_mean)
        #
        self._lod_scale = optimizable_tensors["lod_scale"]
        self._lod_scale.grad = torch.zeros_like(self._lod_scale)
        #
        self._scaling = optimizable_tensors["scaling"]
        self._scaling.grad = torch.zeros_like(self._scaling)
        #
        self._rotation = optimizable_tensors["rotation"]
        self._rotation.grad = torch.zeros_like(self._rotation)

        self.xyz_gradient_accum_diffuse = self.xyz_gradient_accum_diffuse[valid_points_mask]
        self.xyz_gradient_accum_glossy = self.xyz_gradient_accum_glossy[valid_points_mask]
        self.comes_from_colmap_pc = self.comes_from_colmap_pc[valid_points_mask]

        self._round_counter = self._round_counter[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "brdf_lut_residual":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_position,
        new_normal,
        new_roughness_params,
        new_f0_params,
        new_diffuse,
        new_opacity,
        new_lod_mean,
        new_lod_scale,
        new_scaling,
        new_rotation,
        new_round_counter,
        reset_params=True,
    ):
        d = {
            "xyz": new_xyz,
            "position": new_position,
            "normal": new_normal,
            "roughness": new_roughness_params,
            "f0": new_f0_params,
            "f_dc": new_diffuse,  # keep the same name for compat
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "lod_mean": new_lod_mean,
            "lod_scale": new_lod_scale,
        }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        optimizable_tensors["xyz"].grad = torch.zeros_like(optimizable_tensors["xyz"])
        self._xyz = optimizable_tensors["xyz"]

        optimizable_tensors["position"].grad = torch.zeros_like(
            optimizable_tensors["position"]
        )
        self._position = optimizable_tensors["position"]

        optimizable_tensors["normal"].grad = torch.zeros_like(
            optimizable_tensors["normal"]
        )
        self._normal = optimizable_tensors["normal"]

        optimizable_tensors["roughness"].grad = torch.zeros_like(
            optimizable_tensors["roughness"]
        )
        self._roughness = optimizable_tensors["roughness"]

        optimizable_tensors["f0"].grad = torch.zeros_like(optimizable_tensors["f0"])
        self._f0 = optimizable_tensors["f0"]

        optimizable_tensors["f_dc"].grad = torch.zeros_like(optimizable_tensors["f_dc"])
        self._diffuse = optimizable_tensors["f_dc"]  # keep the same name for compat

        optimizable_tensors["opacity"].grad = torch.zeros_like(
            optimizable_tensors["opacity"]
        )
        self._opacity = optimizable_tensors["opacity"]

        optimizable_tensors["lod_mean"].grad = torch.zeros_like(
            optimizable_tensors["lod_mean"]
        )
        self._lod_mean = optimizable_tensors["lod_mean"]

        optimizable_tensors["lod_scale"].grad = torch.zeros_like(
            optimizable_tensors["lod_scale"]
        )
        self._lod_scale = optimizable_tensors["lod_scale"]

        optimizable_tensors["scaling"].grad = torch.zeros_like(
            optimizable_tensors["scaling"]
        )
        self._scaling = optimizable_tensors["scaling"]

        optimizable_tensors["rotation"].grad = torch.zeros_like(
            optimizable_tensors["rotation"]
        )
        self._rotation = optimizable_tensors["rotation"]

        self._round_counter = torch.cat((self._round_counter, new_round_counter), dim=0)
        self.comes_from_colmap_pc = torch.cat(
            (self.comes_from_colmap_pc, torch.zeros_like(new_round_counter)), dim=0
        ) #!!! fix this to update properly

        if reset_params:
            self.xyz_gradient_accum_diffuse = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_glossy = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # ---------------- below: our top kdensification

    def densify_and_split_top_k(self, opt, indices_to_split, N=2):
        if indices_to_split.sum().item() == 0:
            return

        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros(n_init_points, device="cuda").bool()
        selected_pts_mask[indices_to_split] = True

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_position = self._position[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
        new_f0 = self._f0[selected_pts_mask].repeat(N, 1)
        new_diffuse = self._diffuse[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_lod_mean = self._lod_mean[selected_pts_mask].repeat(N, 1)
        if "ON_SPLIT_SKIP_DONT_TOUCH_LOD" not in os.environ:
            new_lod_mean /= 0.8 * N
        new_lod_scale = self._lod_scale[selected_pts_mask].repeat(N, 1)
        new_round_counter = self._round_counter[selected_pts_mask].repeat(N, 1) + 1

        self.densification_postfix(
            new_xyz,
            new_position,
            new_normal,
            new_roughness,
            new_f0,
            new_diffuse,
            new_opacity,
            new_lod_mean,
            new_lod_scale,
            new_scaling,
            new_rotation,
            new_round_counter,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone_top_k(self, opt, indices_to_clone):
        if indices_to_clone.sum().item() == 0:
            return

        selected_pts_mask = torch.zeros(
            len(self.get_xyz), device=self.get_xyz.device
        ).bool()
        selected_pts_mask[indices_to_clone] = True

        if opt.densif_jitter_clones:
            stds = self.get_scaling[selected_pts_mask]
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = (
                torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                + self.get_xyz[selected_pts_mask]
            )
        else:
            new_xyz = self._xyz[selected_pts_mask]
        new_position = self._position[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_f0 = self._f0[selected_pts_mask]
        new_diffuse = self._diffuse[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        if opt.densif_scaledown_clones:
            new_scaling = self.scaling_inverse_activation(
                self.get_scaling[selected_pts_mask] / (0.8 * 2)
            )
        else:
            new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_lod_mean = self._lod_mean[selected_pts_mask]
        new_lod_scale = self._lod_scale[selected_pts_mask]
        new_round_counter = self._round_counter[selected_pts_mask] + 1

        self.densification_postfix(
            new_xyz,
            new_position,
            new_normal,
            new_roughness,
            new_f0,
            new_diffuse,
            new_opacity,
            new_lod_mean,
            new_lod_scale,
            new_scaling,
            new_rotation,
            new_round_counter,
        )

    def prune_znear_only(self, scene):
        prune_mask = scene.select_points_to_prune_near_cameras(self._xyz.data)
        self.prune_points(prune_mask)

    def prune(self, scene, opt, min_opacity, extent):
        # Prune
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if not opt.densif_skip_big_points_ws:
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_ws)

        if opt.densif_no_pruning:
            prune_mask = torch.zeros_like(prune_mask)

        if self.model_params.znear_densif_pruning:
            prune_mask |= scene.select_points_to_prune_near_cameras(self._xyz.data)

        self.prune_points(prune_mask)
    
    def densify_and_prune_top_k(self, scene, opt, min_opacity, extent):
        return self._densify_and_prune_top_k(scene, opt, min_opacity, extent, "diffuse")
        # return self._densify_and_prune_top_k(scene, opt, min_opacity, extent, "glossy")
    
    def _densify_and_prune_top_k(self, scene, opt, min_opacity, extent, mode: str):
        if "CHECK_NAN" in os.environ:
            if torch.isnan(self._xyz).any():
                print("NaN in xyz")
            if torch.isnan(self._scaling).any():
                print("NaN in scaling")
            if torch.isnan(self._rotation).any():
                print("NaN in rotation")
            if torch.isnan(self._opacity).any():
                print("NaN in opacity")
            if torch.isnan(self._lod_mean).any():
                print("NaN in lod_mean")
            if torch.isnan(self._lod_scale).any():
                print("NaN in lod_scale")
            if torch.isnan(self._diffuse).any():
                print("NaN in diffuse")
            if torch.isnan(self._roughness).any():
                print("NaN in roughness")
            if torch.isnan(self._f0).any():
                print("NaN in f0")
            if torch.isnan(self._position).any():
                print("NaN in position")
            if torch.isnan(self._normal).any():
                print("NaN in normal")
            if torch.isnan(self.xyz_gradient_accum_diffuse).any():
                print("NaN in xyz_gradient_accum")
            if torch.isnan(self.xyz_gradient_accum_glossy).any():
                print("NaN in xyz_gradient_accum_glossy")

        self.num_densification_steps += 1
        num_gaussians_before_pruning = self.get_xyz.shape[0]
        self.prune(scene, opt, min_opacity, extent)

        # Densify
        if self.schedule is None:
            self.schedule = get_count_array(self._xyz.shape[0], opt)
            num_gaussians_before_pruning = self._xyz.shape[0]  
            # * start the schedule after the first prune, in case the first prune is very agressive depending on init opacity

        gradient_accum = self.xyz_gradient_accum_diffuse if mode == "diffuse" else self.xyz_gradient_accum_glossy
        num_gaussians_before_densification = self.get_xyz.shape[0]
        grads = gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        padded_grad = torch.zeros((num_gaussians_before_densification), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        largest_axis_size = self.get_scaling.max(dim=1).values

        if opt.densif_pruning_only:
            trace = ""
        else:
            grad_ranking = (
                torch.argsort(torch.argsort(padded_grad))
                / num_gaussians_before_densification
            )
            size_ranking = (
                torch.argsort(torch.argsort(largest_axis_size))
                / num_gaussians_before_densification
            )
            opacity_ranking = (
                torch.argsort(torch.argsort(self.get_opacity.squeeze()))
                / num_gaussians_before_densification
            )
            lod_ranking = (
                torch.argsort(torch.argsort(-self.get_lod_mean.squeeze()))
                / num_gaussians_before_densification
            )

            score = (
                grad_ranking
                + size_ranking * opt.densif_size_ranking_weight
                + opacity_ranking * opt.densif_opacity_ranking_weight
                + opacity_ranking * opt.densif_opacity_ranking_weight
                + lod_ranking * opt.densif_lod_ranking_weight
            )

            if "GRAD_RANKING_ONLY" in os.environ:
                score = grad_ranking

            if opt.densif_lod_ranking_weight != 0.0:  #!!!! tmp
                print("lod ranking is enabled")
                score = (
                    grad_ranking.log()
                    + lod_ranking.log() * opt.densif_lod_ranking_weight
                )

            if "DONT_DENSIFY_HIGH_LOD" in os.environ:
                score[
                    self.get_lod_mean.squeeze()
                    > float(os.environ["DONT_DENSIFY_HIGH_LOD"])
                ] = 0.0
                # log(quantile(grad_score)) + w * log(quantile(-lod_score))

            target = self.schedule[self.num_densification_steps]
            k = target - num_gaussians_before_densification

            target = num_gaussians_before_densification + k

            indices_to_densify = torch.topk(score, k).indices

            if opt.densif_use_fixed_split_clone_ratio:
                scale_threshold = torch.quantile(
                    largest_axis_size[indices_to_densify],
                    q=1.0 - opt.densif_split_clone_ratio,
                )
            else:
                scale_threshold = self.percent_dense * extent

            assert not (opt.densif_no_cloning and opt.densif_no_splitting)

            if opt.densif_no_splitting:
                self.densify_and_clone_top_k(opt, indices_to_densify)
            elif opt.densif_no_cloning:
                self.densify_and_split_top_k(opt, indices_to_densify)
            else:
                indices_to_clone = indices_to_densify[
                    largest_axis_size[indices_to_densify] < scale_threshold
                ]
                indices_to_split = indices_to_densify[
                    largest_axis_size[indices_to_densify] >= scale_threshold
                ]

                self.densify_and_clone_top_k(opt, indices_to_clone)
                self.densify_and_split_top_k(opt, indices_to_split)

            trace = f"Step {self.num_densification_steps} :: Init: {num_gaussians_before_pruning}; After Pruning: {num_gaussians_before_densification}; After Densification: {self.get_xyz.shape[0]}; Target: {target}\n"
            print(trace)

        torch.cuda.empty_cache()

        return trace

    def add_densification_stats_3d(self, gradient_diffuse, gradient_glossy):
        update_filter = self._opacity.grad[:, 0] != 0

        self.xyz_gradient_accum_diffuse[update_filter] += gradient_diffuse[update_filter].norm(dim=-1, keepdim=True)
        self.xyz_gradient_accum_glossy[update_filter] += gradient_glossy[update_filter].norm(dim=-1, keepdim=True)

        self.denom[update_filter] += 1


def get_count_array(start_count, opt):
    # Eq. (2) of taming-3dgs

    num_steps = (
        opt.densify_until_iter - opt.densify_from_iter
    ) // opt.densification_interval
    slope_lower_bound = (opt.densif_final_num_gaussians - start_count) / (num_steps - 1)

    k = 2 * slope_lower_bound
    a = (opt.densif_final_num_gaussians - start_count - k * (num_steps - 1)) / (
        (num_steps - 1) * (num_steps - 1)
    )
    b = k
    c = start_count

    values = [int(1 * a * (x**2) + (b * x) + c) for x in range(num_steps)]

    return values
