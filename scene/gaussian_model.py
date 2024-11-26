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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from arguments import ModelParams
import cv2 

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

        if self.model_params.linear_space:
            self.diffuse_activation = torch.nn.functional.softplus
        else:
            self.diffuse_activation = torch.sigmoid


    def __init__(self, model_params: ModelParams):
        self.model_params = model_params
        self._xyz = torch.empty(0)
        self._normal = torch.empty(0)
        self._position = torch.empty(0)
        self._brdf_params = torch.empty(0)
        self._diffuse = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

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
            self._brdf_params,
            self._diffuse,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self._xyz, 
        self._normal,
        self._position,
        self._brdf_params,
        self._diffuse, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

        self._xyz.grad = torch.zeros_like(self._xyz)
        self._normal.grad = torch.zeros_like(self._normal)
        self._position.grad = torch.zeros_like(self._position)
        self._brdf_params.grad = torch.zeros_like(self._brdf_params)
        self._opacity.grad = torch.zeros_like(self._opacity)
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
    def get_diffuse(self):
        return self.diffuse_activation(self._diffuse)
    
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
    def get_brdf_params(self):
        return self._brdf_params
    
    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling,
                                          1 / scaling_modifier,
                                          self.get_rotation)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        fused_normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        if self.model_params.mcmc_densify and not self.model_params.mcmc_densify_disable_custom_init:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2)*0.1)[...,None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2 / float(os.getenv("SCALEDOWN", 1.0))))[...,None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        
        self._position = nn.Parameter(fused_point_cloud.clone()) 
        self._normal = nn.Parameter(fused_normal.clone())
        self._brdf_params = nn.Parameter(torch.cat([fused_color.clone(), torch.zeros_like(fused_color)[..., :1]], dim=-1))
        self._diffuse = nn.Parameter(fused_color.clone())
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._xyz.grad = torch.zeros_like(self._xyz)
        self._position.grad = torch.zeros_like(self._position)
        self._normal.grad = torch.zeros_like(self._normal)
        self._brdf_params.grad = torch.zeros_like(self._brdf_params)
        self._opacity.grad = torch.zeros_like(self._opacity)
        self._diffuse.grad = torch.zeros_like(self._diffuse)
        self._scaling.grad = torch.zeros_like(self._scaling)
        self._rotation.grad = torch.zeros_like(self._rotation)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self._xyz.grad = torch.zeros_like(self._xyz)
        self._position.grad = torch.zeros_like(self._position)
        self._normal.grad = torch.zeros_like(self._normal)
        self._brdf_params.grad = torch.zeros_like(self._brdf_params)
        self._opacity.grad = torch.zeros_like(self._opacity)
        self._diffuse.grad = torch.zeros_like(self._diffuse)
        self._scaling.grad = torch.zeros_like(self._scaling)
        self._rotation.grad = torch.zeros_like(self._rotation)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._position], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "position"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._brdf_params], 'lr': training_args.brdf_params_lr, "name": "brdf_params"},
            {'params': [self._diffuse], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        if self.model_params.brdf_mode == "finetuned_lut":
            l.append({'params': [self._brdf_lut_residual], 'lr': training_args.brdf_lut_lr, "name": "brdf_lut_residual"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._diffuse.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        position = self._position.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        brdf_params = self._brdf_params.detach().cpu().numpy()

        all_attributes = [
            'x', 'y', 'z', 
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity',
            'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3',
            'pos_0', 'pos_1', 'pos_2',
            'normal_0', 'normal_1', 'normal_2',
            'brdf_params_0', 'brdf_params_1', 'brdf_params_2', 'brdf_params_3'
        ]
        dtype_full = [(attribute, 'f4') for attribute in all_attributes]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, opacities, scale, rotation, position, normal, brdf_params), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self.model_params.brdf_mode == "finetuned_lut":
            brdf_lut_path = path.replace("point_cloud.ply", "brdf_lut_residuals.pt")
            torch.save(self._brdf_lut_residual, brdf_lut_path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        self._opacity.grad = torch.zeros_like(self._opacity)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        diffuse = np.zeros((xyz.shape[0], 3))
        diffuse[:, 0] = np.asarray(plydata.elements[0][f"f_dc_0"])
        diffuse[:, 1] = np.asarray(plydata.elements[0][f"f_dc_1"])
        diffuse[:, 2] = np.asarray(plydata.elements[0][f"f_dc_2"])
        
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        pos_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("pos")]
        pos_names = sorted(pos_names, key = lambda x: int(x.split('_')[-1]))
        positions = np.zeros((xyz.shape[0], len(pos_names)))
        for idx, attr_name in enumerate(pos_names):
            positions[:, idx] = np.asarray(plydata.elements[0][attr_name])

        normal_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("normal")]
        normal_names = sorted(normal_names, key = lambda x: int(x.split('_')[-1]))
        normals = np.zeros((xyz.shape[0], len(normal_names)))
        for idx, attr_name in enumerate(normal_names):
            normals[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        brdf_params_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("brdf_params")]
        brdf_params_names = sorted(brdf_params_names, key = lambda x: int(x.split('_')[-1]))
        brdf_params = np.zeros((xyz.shape[0], len(brdf_params_names)))
        for idx, attr_name in enumerate(brdf_params_names):
            brdf_params[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._diffuse = nn.Parameter(torch.tensor(diffuse, dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda"))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda"))
        self._position = nn.Parameter(torch.tensor(positions, dtype=torch.float, device="cuda"))
        self._normal = nn.Parameter(torch.tensor(normals, dtype=torch.float, device="cuda"))
        self._brdf_params = nn.Parameter(torch.tensor(brdf_params, dtype=torch.float, device="cuda"))

        if self.model_params.brdf_mode == "finetuned_lut":
            self._brdf_lut_residual = nn.Parameter(torch.load(path.replace("point_cloud.ply", "brdf_lut_residuals.pt")).to("cuda"))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor)
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "brdf_lut_residual":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask]))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask])
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._position = optimizable_tensors["position"]
        self._normal = optimizable_tensors["normal"]
        self._brdf_params = optimizable_tensors["brdf_params"]
        self._diffuse = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "brdf_lut_residual":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_position, new_normal, new_brdf_params, new_diffuse, new_opacity, new_scaling, new_rotation, reset_params=True):
        d = {"xyz": new_xyz,
        "position": new_position,
        "normal": new_normal,
        "brdf_params": new_brdf_params,
        "f_dc": new_diffuse, # keep the same name for compat
        "opacity": new_opacity,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        optimizable_tensors["xyz"].grad = torch.zeros_like(optimizable_tensors["xyz"])
        self._xyz = optimizable_tensors["xyz"]
        optimizable_tensors["position"].grad = torch.zeros_like(optimizable_tensors["position"])
        self._position = optimizable_tensors["position"]
        optimizable_tensors["normal"].grad = torch.zeros_like(optimizable_tensors["normal"])
        self._normal = optimizable_tensors["normal"]
        optimizable_tensors["brdf_params"].grad = torch.zeros_like(optimizable_tensors["brdf_params"])
        self._brdf_params = optimizable_tensors["brdf_params"]
        optimizable_tensors["f_dc"].grad = torch.zeros_like(optimizable_tensors["f_dc"])
        self._diffuse = optimizable_tensors["f_dc"] # keep the same name for compat
        optimizable_tensors["opacity"].grad = torch.zeros_like(optimizable_tensors["opacity"])
        self._opacity = optimizable_tensors["opacity"]
        optimizable_tensors["scaling"].grad = torch.zeros_like(optimizable_tensors["scaling"])
        self._scaling = optimizable_tensors["scaling"]
        optimizable_tensors["rotation"].grad = torch.zeros_like(optimizable_tensors["rotation"])
        self._rotation = optimizable_tensors["rotation"]

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_position = self._position[selected_pts_mask].repeat(N,1)
        new_normal = self._normal[selected_pts_mask].repeat(N,1)
        new_brdf_params = self._brdf_params[selected_pts_mask].repeat(N,1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_diffuse = self._diffuse[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # new_xyz, new_position, new_normal, new_brdf_params, new_diffuse, new_opacities, new_scaling, new_rotation,
        self.densification_postfix(new_xyz, new_position, new_normal, new_brdf_params, new_diffuse,  new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_position = self._position[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_brdf_params = self._brdf_params[selected_pts_mask]
        new_diffuse = self._diffuse[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_position, new_normal, new_brdf_params, new_diffuse, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        self._xyz.grad = torch.zeros_like(self._xyz)
        self._position.grad = torch.zeros_like(self._position)
        self._normal.grad = torch.zeros_like(self._normal)
        self._brdf_params.grad = torch.zeros_like(self._brdf_params)
        self._opacity.grad = torch.zeros_like(self._opacity)
        self._diffuse.grad = torch.zeros_like(self._diffuse)
        self._scaling.grad = torch.zeros_like(self._scaling)
        self._rotation.grad = torch.zeros_like(self._rotation)

    def densify_and_prune_glossy(self, max_grad, min_opacity, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        
        # # if max_screen_size:
        #     # big_points_vs = self.max_radii2D > max_screen_size
        # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        # prune_mask = torch.logical_or(prune_mask, big_points_ws)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        self._xyz.grad = torch.zeros_like(self._xyz)
        self._position.grad = torch.zeros_like(self._position)
        self._normal.grad = torch.zeros_like(self._normal)
        self._brdf_params.grad = torch.zeros_like(self._brdf_params)
        self._opacity.grad = torch.zeros_like(self._opacity)
        self._diffuse.grad = torch.zeros_like(self._diffuse)
        self._scaling.grad = torch.zeros_like(self._scaling)
        self._rotation.grad = torch.zeros_like(self._rotation)

    def densify_with_basic_pruning(self, max_grad, min_opacity, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # if max_screen_size:
            # big_points_vs = self.max_radii2D > max_screen_size
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats_glossy(self, xyz, update_filter):
        if xyz.grad is not None:
            self.xyz_gradient_accum[update_filter] += torch.norm(xyz.grad, dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if viewspace_point_tensor.grad is not None:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


# ---------------- below: mcmc densification


    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_opacity.grad = torch.zeros_like(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))
        new_scaling.grad = torch.zeros_like(new_scaling)

        return self._xyz[idxs], self._diffuse[idxs], new_opacity, new_scaling, self._rotation[idxs], self._position[idxs], self._normal[idxs], self._brdf_params[idxs]


    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    

    def relocate_gs(self, dead_mask=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices], 
            self._diffuse[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
            self._position[dead_indices],
            self._normal[dead_indices],
            self._brdf_params[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx) 
        
    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._diffuse,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation,
            "position" : self._position,
            "normal" : self._normal,
            "brdf_params" : self._brdf_params
        }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups: #!!!!!!!!! missing lut stuff
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        optimizable_tensors["xyz"].grad = self._xyz.grad
        self._xyz = optimizable_tensors["xyz"]

        optimizable_tensors["f_dc"].grad = self._diffuse.grad
        self._diffuse = optimizable_tensors["f_dc"]
        
        optimizable_tensors["opacity"].grad = self._opacity.grad
        self._opacity = optimizable_tensors["opacity"]

        optimizable_tensors["scaling"].grad = self._scaling.grad
        self._scaling = optimizable_tensors["scaling"]
        
        optimizable_tensors["rotation"].grad = self._rotation.grad
        self._rotation = optimizable_tensors["rotation"] 

        optimizable_tensors["normal"].grad = self._normal.grad
        self._normal = optimizable_tensors["normal"]

        optimizable_tensors["brdf_params"].grad = self._brdf_params.grad
        self._brdf_params = optimizable_tensors["brdf_params"]

        return optimizable_tensors
    
    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        print("Current num gaussians:", current_num_points, "Target num gaussians:", target_num)

        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1) 
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz, 
            new_diffuse,
            new_opacity,
            new_scaling,
            new_rotation,
            new_position,
            new_normal,
            new_brdf_params
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(new_xyz=new_xyz, new_position=new_position, new_normal=new_normal, new_brdf_params=new_brdf_params, new_diffuse=new_diffuse, new_opacity=new_opacity, new_scaling=new_scaling, new_rotation=new_rotation)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

import math 

N_max = 51
binoms = torch.zeros((N_max, N_max)).float().cuda()
for n in range(N_max):
    for k in range(n+1):
        binoms[n, k] = math.comb(n, k)

def compute_relocation_cuda(opacity_old, scale_old, N):
    N.clamp_(min=1, max=N_max-1)
    return torch.ops.gausstracer.compute_relocation(opacity_old, scale_old, N.to(torch.int), binoms, torch.tensor(N_max).to(torch.int)) # returns new_opacity, new_scale