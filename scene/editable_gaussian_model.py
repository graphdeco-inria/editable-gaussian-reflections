import torch
from scene.gaussian_model import GaussianModel
from dataclasses import dataclass
import kornia
import math 

class EditableGaussianModel(GaussianModel):
    def __init__(self, edits: dict, bounding_boxes: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert set(edits.keys()) == set(bounding_boxes.keys()), "Edits and bounding boxes must have the same keys"
        self.edits = edits
        self.bounding_boxes = bounding_boxes

    def construct_selections(self):
        self.selections = {}
        with torch.no_grad():
            for key in self.edits.keys():
                bounding_box = self.bounding_boxes[key]
                dist1 = self._xyz - (torch.tensor(bounding_box["min"], device="cuda"))
                dist2 = self._xyz - (torch.tensor(bounding_box["max"], device="cuda"))
                within_bbox = (dist1 >= 0).all(dim=-1) & (dist2 <= 0).all(dim=-1)
                self.selections[key] = within_bbox.unsqueeze(1)

    # ----------------------------------------------------------------

    
    @property
    def get_roughness(self):
        roughness = super().get_roughness.clone()

        for key, edit in self.edits.items():
            base_roughness = torch.lerp(roughness, torch.tensor(1.0).cuda(), edit.roughness_override)
            modified_roughness = (edit.roughness_mult * (base_roughness + edit.roughness_shift)).clamp(0, 1)
            roughness = torch.where(self.selections[key], modified_roughness, roughness)

        return roughness

    @property
    def get_diffuse(self):
        diffuse = super().get_diffuse.clone()
            
        for key, edit in self.edits.items():
            base_diffuse = torch.lerp(diffuse, torch.tensor(edit.diffuse_override)[:3].cuda(), edit.diffuse_override[-1])
            hsv = kornia.color.rgb_to_hsv(base_diffuse.T[None, :, :, None])[0, :, :, 0].T
            hsv[:, 0] = (hsv[:, 0] + math.pi * edit.diffuse_hue_shift) % (2 * math.pi)
            hsv[:, 1] = (edit.diffuse_saturation_mult * (hsv[:, 1] + edit.diffuse_saturation_shift)).clamp(0, 1)
            hsv[:, 2] = (edit.diffuse_value_mult * (hsv[:, 2] + edit.diffuse_value_shift)).clamp(0)
            modified_diffuse = kornia.color.hsv_to_rgb(hsv.T[None, :, :, None])[0, :, :, 0].T
            diffuse = torch.where(self.selections[key], modified_diffuse, diffuse)
        
        return diffuse
    
    @property
    def get_f0(self):
        f0 = super().get_f0.clone()

        for key, edit in self.edits.items():
            base_f0 = torch.lerp(f0, torch.tensor(edit.glossy_override)[:3].cuda(), edit.glossy_override[-1])
            hsv = kornia.color.rgb_to_hsv(base_f0.T[None, :, :, None])[0, :, :, 0].T
            hsv[:, 0] = (hsv[:, 0] + math.pi * edit.glossy_hue_shift) % (2 * math.pi)
            hsv[:, 1] = (edit.glossy_saturation_mult * (hsv[:, 1] + edit.glossy_saturation_shift)).clamp(0, 1)
            hsv[:, 2] = (edit.glossy_value_mult * (hsv[:, 2] + edit.glossy_value_shift)).clamp(0)
            modified_f0 = kornia.color.hsv_to_rgb(hsv.T[None, :, :, None])[0, :, :, 0].T
            f0 = torch.where(self.selections[key], modified_f0, f0)

        return f0

    # ----------------------------------------------------------------

    @property
    def get_xyz(self):
        xyz = super().get_xyz.clone()
        for key, edit in self.edits.items():
            xyz[self.selections[key].squeeze(1)] += torch.tensor([edit.translate_x, edit.translate_y, edit.translate_z], device=xyz.device)
        return xyz 

    @property
    def get_scaling(self):
        scaling = super().get_scaling.clone()
        for key, edit in self.edits.items():
            scaling[self.selections[key].squeeze(1)] *= torch.tensor([edit.scale_x, edit.scale_y, edit.scale_z], device=scaling.device)
        return scaling

    @property
    def get_rotation(self):
        return super().get_rotation # todo


    # ----------------------------------------------------------------

    def clone_selection(self, selection_name: str):
        selection = self.edits[selection_name].selection

        new_xyz = self._xyz[selection].clone()
        new_position = self._position[selection].clone()
        new_rotation = self._rotation[selection].clone()
        new_scaling = self._scaling[selection].clone()
        new_opacity = self._opacity[selection].clone()
        new_diffuse = self._diffuse[selection].clone()
        new_roughness = self._roughness[selection].clone()
        new_f0 = self._f0[selection].clone()
        new_normal = self._normal[selection].clone()
        new_lod_mean = self._lod_mean[selection].clone()
        new_lod_scale = self._lod_scale[selection].clone()
        new_round_counter = self._round_counter[selection].clone()

        self.densification_postfix(
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
            new_round_counter
        )
