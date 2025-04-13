from scene.gaussian_model import GaussianModel
from dataclasses import dataclass





class EditableGaussianModel(GaussianModel):
    def __init__(self, edits: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edits = edits

    # ----------------------------------------------------------------

    @property
    def get_roughness(self):
        roughness = super().get_roughness.clone()

        for edit in self.edits.values():
            base_roughness = torch.lerp(roughness, torch.tensor(1.0).cuda(), edits.roughness_override)
            modified_roughness = (edits.roughness_mult * (base_roughness + edits.roughness_shift)).clamp(0, 1)
            roughness = torch.where(edits.selection, modified_roughness, roughness)

        return roughness

    @property
    def get_diffuse(self):
        diffuse = super().get_diffuse.clone()
            
        for edit in self.edits.values():
            base_diffuse = torch.lerp(diffuse, torch.tensor(edits.diffuse_override)[:3].cuda(), edits.diffuse_override[-1])
            hsv = kornia.color.rgb_to_hsv(base_diffuse.T[None, :, :, None])[0, :, :, 0].T
            hsv[:, 0] = (hsv[:, 0] + math.pi * edits.diffuse_hue_shift) % (2 * math.pi)
            hsv[:, 1] = (edits.diffuse_saturation_mult * (hsv[:, 1] + edits.diffuse_saturation_shift)).clamp(0, 1)
            hsv[:, 2] = (edits.diffuse_value_mult * (hsv[:, 2] + edits.diffuse_value_shift)).clamp(0)
            modified_diffuse = kornia.color.hsv_to_rgb(hsv.T[None, :, :, None])[0, :, :, 0].T
            diffuse = torch.where(edits.selection, modified_diffuse, diffuse)
        
        return diffuse
    
    @property
    def get_f0(self):
        f0 = super().get_f0.clone()

        for edit in self.edits.values():
            base_f0 = torch.lerp(f0, torch.tensor(edit.glossy_override)[:3].cuda(), edit.glossy_override[-1])
            hsv = kornia.color.rgb_to_hsv(base_f0.T[None, :, :, None])[0, :, :, 0].T
            hsv[:, 0] = (hsv[:, 0] + math.pi * edit.glossy_hue_shift) % (2 * math.pi)
            hsv[:, 1] = (edit.glossy_saturation_mult * (hsv[:, 1] + edit.glossy_saturation_shift)).clamp(0, 1)
            hsv[:, 2] = (edit.glossy_value_mult * (hsv[:, 2] + edit.glossy_value_shift)).clamp(0)
            modified_f0 = kornia.color.hsv_to_rgb(hsv.T[None, :, :, None])[0, :, :, 0].T
            f0 = torch.where(edit.selection, modified_f0, f0)

        return f0

    # ----------------------------------------------------------------

    @property
    def get_xyz(self):
        xyz = super().get_xyz.clone()
        for edit in self.edits.values():
            xyz[edit.selection] += torch.tensor([edit.translation_x, edit.translation_y, edit.translation_z], device=xyz.device)
        return xyz

    @property
    def get_scaling(self):
        scaling = super().get_scaling.clone()
        for edit in self.edits.values():
            scaling[edit.selection] *= torch.tensor([edit.scale_x, edit.scale_y, edit.scale_z], device=scaling.device)
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
