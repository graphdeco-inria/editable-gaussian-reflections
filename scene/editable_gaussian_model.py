import torch
from scene.gaussian_model import GaussianModel
from dataclasses import dataclass
import kornia
import math 
import copy 
import os

class EditableGaussianModel(GaussianModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready_for_editing = False

    def make_editable(self, edits, bounding_boxes, model_path):
        assert set(edits.keys()) == set(bounding_boxes.keys()), "Edits and bounding boxes must have the same keys"
        self.edits = edits 
        self.bounding_boxes = bounding_boxes
        self.created_objects = list(self.edits.keys())
        self.selections = {}
        
        with torch.no_grad():
            for key in self.edits.keys():
                saved_selection_path = model_path + f"/selections/{key}.pt"
                if os.path.exists(saved_selection_path):
                    self.selections[key] = torch.load(saved_selection_path)
                else:
                    bounding_box = self.bounding_boxes[key]
                    dist1 = self._xyz - (torch.tensor(bounding_box["min"], device="cuda"))
                    dist2 = self._xyz - (torch.tensor(bounding_box["max"], device="cuda"))
                    within_bbox = (dist1 >= 0).all(dim=-1) & (dist2 <= 0).all(dim=-1)
                    self.selections[key] = within_bbox.unsqueeze(1)
            self.selections["everything"] = torch.ones(self._xyz.shape[0], 1, device="cuda", dtype=torch.bool)

        self.ready_for_editing = True
        self.previous_edits = None

        self.last_roughness = None
        self.last_diffuse = None
        self.last_f0 = None
        self.last_xyz = None 
        self.last_scaling = None 
        self.last_rotation = None

        self.is_dirty = True
        self.last_edits = None
        self.last_scaling_modifier = 1.0

    # ----------------------------------------------------------------

    def dirty_check(self, scaling_modifier):
        if self.last_edits is None or self.edits != self.last_edits or self.last_scaling_modifier != scaling_modifier:
            self.last_edits = copy.deepcopy(self.edits)
            self.last_scaling_modifier = scaling_modifier
            self.is_dirty = True
        else:
            self.is_dirty = False 
 
    @property
    def get_roughness(self):
        roughness = super().get_roughness.clone()

        if not self.ready_for_editing:
            return roughness
        
        if not self.is_dirty:
            return self.roughness

        for key, edit in self.edits.items():
            if edit.use_roughness_override:
                base_roughness = roughness * 0 + edit.roughness_override**2
            else:
                base_roughness = roughness
            modified_roughness = (edit.roughness_mult * (base_roughness +  math.copysign(edit.roughness_shift, edit.roughness_shift**2))).clamp(0, 1)
            roughness = torch.where(self.selections[key], modified_roughness, roughness)

        self.roughness = roughness
        return roughness

    @property
    def get_diffuse(self):
        diffuse = super().get_diffuse.clone()

        if not self.ready_for_editing:
            return diffuse
        
        if not self.is_dirty:
            return self.diffuse
            
        for key, edit in self.edits.items():
            base_diffuse = torch.lerp(diffuse, torch.tensor(edit.diffuse_override)[:3].cuda(), edit.diffuse_override[-1])
            hsv = kornia.color.rgb_to_hsv(base_diffuse.T[None, :, :, None])[0, :, :, 0].T
            hsv[:, 0] = (hsv[:, 0] + math.pi * edit.diffuse_hue_shift) % (2 * math.pi)
            hsv[:, 1] = (edit.diffuse_saturation_mult * (hsv[:, 1] + edit.diffuse_saturation_shift)).clamp(0, 1)
            hsv[:, 2] = (edit.diffuse_value_mult * (hsv[:, 2] + edit.diffuse_value_shift)).clamp(0)
            modified_diffuse = kornia.color.hsv_to_rgb(hsv.T[None, :, :, None])[0, :, :, 0].T
            diffuse = torch.where(self.selections[key], modified_diffuse, diffuse)
        
        self.diffuse = diffuse
        return diffuse
    
    @property
    def get_f0(self):
        f0 = super().get_f0.clone()
        
        if not self.ready_for_editing:
            return f0
        
        if not self.is_dirty:
            return self.f0

        for key, edit in self.edits.items():
            base_f0 = torch.lerp(f0, torch.tensor(edit.glossy_override)[:3].cuda(), edit.glossy_override[-1])
            hsv = kornia.color.rgb_to_hsv(base_f0.T[None, :, :, None])[0, :, :, 0].T
            hsv[:, 0] = (hsv[:, 0] + math.pi * edit.glossy_hue_shift) % (2 * math.pi)
            hsv[:, 1] = (edit.glossy_saturation_mult * (hsv[:, 1] + edit.glossy_saturation_shift)).clamp(0, 1)
            hsv[:, 2] = (edit.glossy_value_mult * (hsv[:, 2] + edit.glossy_value_shift)).clamp(0)
            modified_f0 = kornia.color.hsv_to_rgb(hsv.T[None, :, :, None])[0, :, :, 0].T
            f0 = torch.where(self.selections[key], modified_f0, f0)

        self.f0 = f0
        return f0

    # ----------------------------------------------------------------

    @property
    def get_xyz(self):
        xyz = super().get_xyz.clone()
        
        if not self.ready_for_editing:
            return xyz
        
        if not self.is_dirty:
            return self.xyz
        
        for key, edit in self.edits.items():
            xyz[self.selections[key].squeeze(1)] += torch.tensor([edit.translate_x, edit.translate_y, edit.translate_z], device=xyz.device)
            bounding_box = self.bounding_boxes[key]
            bbox_center = torch.tensor([(bounding_box["min"][i] + bounding_box["max"][i]) / 2 for i in range(3)], device=xyz.device)
            object_center = bbox_center + torch.tensor([edit.translate_x, edit.translate_y, edit.translate_z], device=xyz.device)
            xyz[self.selections[key].squeeze(1)] = (
                (xyz[self.selections[key].squeeze(1)] - object_center) * edit.scale + object_center
            )
            rotation_angles = torch.tensor([edit.rotate_x, edit.rotate_y, edit.rotate_z], device=xyz.device)
            rotation_angles = torch.deg2rad(rotation_angles) 
            rotation_matrix = kornia.geometry.axis_angle_to_rotation_matrix(rotation_angles[None])[0]

            xyz[self.selections[key].squeeze(1)] = (
                torch.matmul(
                    xyz[self.selections[key].squeeze(1)] - object_center, 
                    rotation_matrix.T
                ) + object_center
            )
        
        self.xyz = xyz
        return xyz 

    @property
    def _get_scaling(self): # no activation
        scaling = torch.exp(self._scaling)
        
        if not self.ready_for_editing:
            return torch.log(scaling)
        
        if not self.is_dirty:
            return self.scaling
        
        for key, edit in self.edits.items():
            scaling[self.selections[key].squeeze(1)] *= edit.scale # torch.tensor([edit.scale_x, edit.scale_y, edit.scale_z], device=scaling.device)

        self.scaling = torch.log(scaling)
        return self.scaling

    @property
    def _get_rotation(self): # no activation
        rotation = self._rotation.clone()

        if not self.ready_for_editing:
            return rotation
        
        if not self.is_dirty:
            return self.rotation 
        
        for key, edit in self.edits.items():
            rotation_angles = torch.deg2rad(torch.tensor([edit.rotate_x, edit.rotate_y, edit.rotate_z], device=rotation.device)).unsqueeze(0)
            rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(rotation[self.selections[key].squeeze(1)])
            rotation_matrix = kornia.geometry.axis_angle_to_rotation_matrix(rotation_angles)[0] @ rotation_matrix
            rotation[self.selections[key].squeeze(1)] = kornia.geometry.conversions.rotation_matrix_to_quaternion(rotation_matrix)
        
        self.rotation = rotation
        return rotation

    # ----------------------------------------------------------------

    @torch.no_grad()
    def duplicate_object(self, obj_name: str, offset: float):
        target_selection = self.selections[obj_name].squeeze(1).cuda()
        edit = self.edits[obj_name]

        delta_xyz = torch.tensor([edit.translate_x, edit.translate_y, edit.translate_z], device="cuda")

        new_xyz = self._xyz[target_selection].clone() + offset + delta_xyz
        new_position = self._position[target_selection].clone()
        new_rotation = self._rotation[target_selection].clone()
        new_scaling = self._scaling[target_selection].clone()
        new_opacity = self._opacity[target_selection].clone()
        new_diffuse = self._diffuse[target_selection].clone()
        new_roughness = self._roughness[target_selection].clone()
        new_f0 = self._f0[target_selection].clone()
        new_normal = self._normal[target_selection].clone()
        new_lod_mean = self._lod_mean[target_selection].clone()
        new_lod_scale = self._lod_scale[target_selection].clone()
        new_round_counter = self._round_counter[target_selection].clone()

        self._xyz = torch.cat((self._xyz, new_xyz), dim=0)
        self._position = torch.cat((self._position, new_position), dim=0)
        self._rotation = torch.cat((self._rotation, new_rotation), dim=0)
        self._scaling = torch.cat((self._scaling, new_scaling), dim=0)
        self._opacity = torch.cat((self._opacity, new_opacity), dim=0)
        self._diffuse = torch.cat((self._diffuse, new_diffuse), dim=0)
        self._roughness = torch.cat((self._roughness, new_roughness), dim=0)
        self._f0 = torch.cat((self._f0, new_f0), dim=0)
        self._normal = torch.cat((self._normal, new_normal), dim=0)
        self._lod_mean = torch.cat((self._lod_mean, new_lod_mean), dim=0)
        self._lod_scale = torch.cat((self._lod_scale, new_lod_scale), dim=0)
        self._round_counter = torch.cat((self._round_counter, new_round_counter), dim=0)

        self.selections[obj_name + "_copy"] = torch.zeros_like(self.selections[obj_name], dtype=torch.bool)
        xtra = target_selection[target_selection].unsqueeze(1)
        for key, selection in self.selections.items():
            new_selection = torch.cat((selection, xtra if key in ["Everything", obj_name + "_copy"] else ~xtra), dim=0)
            self.selections[key] = new_selection

        self.created_objects.append(obj_name + "_copy")

    @torch.no_grad()
    def remove_object(self, obj_name: str):
        target_selection = self.selections[obj_name].squeeze(1).cuda()
        self._opacity[target_selection] *= 0.0
        self._opacity[target_selection] -= 100000000.0 # its a sigmoid