import os
from OpenGL.GL import *
from threading import Lock
from argparse import ArgumentParser
from imgui_bundle import imgui_ctx, imgui
from viewer import Viewer
from viewer.types import ViewerMode
from viewer.widgets.image import TorchImage
from viewer.widgets.cameras.fps import FPSCamera
from viewer.widgets.monitor import PerformanceMonitor
from viewer.widgets.ellipsoid_viewer import EllipsoidViewer
from scene.tonemapping import *
import json 
from argparse import ArgumentParser, Namespace
from imgui_bundle import imgui_ctx, imgui, imguizmo
import math 
from PIL import Image
from viewer.widgets.ellipsoid_viewer import EllipsoidViewer
from dataclasses import dataclass
import dataclasses

gizmo = imguizmo.im_guizmo
Matrix3 = gizmo.Matrix3
Matrix6 = gizmo.Matrix6
Matrix16 = gizmo.Matrix16

class Dummy(object):
    pass

DUPLICATION_OFFSET = 0.08

@dataclass(eq=True)
class Edit:
    roughness_shift: float = 0.0
    roughness_mult: float = 1.0

    diffuse_override: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.0)
    diffuse_hue_shift: float = 0.0
    diffuse_saturation_shift: float = 0.0
    diffuse_saturation_mult: float = 1.0
    diffuse_value_shift: float = 0.0
    diffuse_value_mult: float = 1.0
    
    use_roughness_override: bool = False
    roughness_override: float = 0.0

    glossy_override: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.0)
    glossy_hue_shift: float = 0.0
    glossy_saturation_shift: float = 0.0
    glossy_saturation_mult: float = 1.0
    glossy_value_shift: float = 0.0
    glossy_value_mult: float = 1.0

    translate_x: float = 0.0
    translate_y: float = 0.0
    translate_z: float = 0.0

    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0

    rotate_x: float = 0.0
    rotate_y: float = 0.0 
    rotate_z: float = 0.0
    

class GaussianViewer(Viewer):
    def __init__(self, mode: ViewerMode, raytracer: "GaussianRaytracer"):
        super().__init__(mode)
        self.window_title = "Gaussian Viewer"
        self.gaussian_lock = Lock()
        self.raytracer = raytracer
        if self.raytracer is not None:
            self.ray_count = self.raytracer.config.MAX_BOUNCES + 1
            self.accumulated_rgb = torch.zeros_like(raytracer.cuda_module.output_rgb[:-2])
            self.current_sample_count = 0
        else:
            self.ray_count = 4
        self.init_pose = None
        self.train_transforms = None 
        self.test_transforms = None
        self.current_train_cam = -1 
    
        self.current_test_cam = -1

        self.blender_to_opengl = np.array([
            [1,  0,  0,  0],
            [0,  -1,  0,  0],
            [0, 0,  -1,  0],
            [0,  0,  0,  1]
        ], dtype=float)

        self.selected_object_transform = Matrix16([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.selection_mode_counter = 0
        self.last_rendered_selection_mask_id = -1
        self.sum_rgb_passes = False
        self.denoise = True
        self.accumulate_samples = True

    def import_server_modules(self):
        global torch
        import torch

        global GaussianModel
        from scene import GaussianModel

        global PipelineParams, ModelParams
        from arguments import PipelineParams, ModelParams

        global MiniCam
        from scene.cameras import MiniCam

        global render
        from gaussian_renderer import render
        
        global GaussianRaytracer
        from gaussian_renderer import GaussianRaytracer

    @classmethod
    def from_ply(cls, model_path, iter, mode: ViewerMode):
        global EditableGaussianModel
        from scene import EditableGaussianModel
        from gaussian_renderer import GaussianRaytracer

        # Read configuration
        with open(os.path.join(model_path, "model_params")) as f:
            model_params = eval(f.read())
       
        dataset = Dummy()
        dataset.white_background = False # params["white_background"] == "True"
        dataset.sh_degree = 0
        #dataset.train_test_exp = params["train_test_exp"] == "True"

        pipe = Dummy()
        pipe.debug = "debug" in model_params
        pipe.antialiasing = "antialiasing" in model_params
        pipe.compute_cov3D_python = "compute_cov3D_python" in model_params
        pipe.convert_SHs_python = "convert_SHs_python" in model_params
        
        gaussians = EditableGaussianModel(model_params)
        ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iter}", "point_cloud.ply")
        gaussians.load_ply(ply_path)
        raytracer = GaussianRaytracer(gaussians, int(model_params.resolution*1.5), model_params.resolution) #!!! hard-coded aspect ratio

        viewer = cls(mode, raytracer)
        viewer.separate_sh = False
        viewer.gaussians = gaussians
        viewer.dataset = dataset
        viewer.pipe = pipe
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        viewer.background = background

        viewer.load_metadata(model_params)

        return viewer
    
    def load_metadata(self, model_params):
        source_path = model_params.source_path
        
        self.train_transforms = json.load(open(os.path.join(source_path, "transforms_train.json"), "r"))
        self.test_transforms = json.load(open(os.path.join(source_path, "transforms_test.json"), "r"))

        self.bounding_boxes = json.load(open(os.path.join(source_path, "bounding_boxes.json"), "r"))
        
        self.edits = { bbox_name: Edit() for bbox_name in self.bounding_boxes.keys() }
        
        self.selection_masks = {}

        self.gaussians.make_editable(self.edits, self.bounding_boxes)

        self.bounding_boxes["everything"] = {"min": [-1000, 1000, -1000], "max": [1000, 1000, 1000] }
        

    @classmethod
    def from_gaussians(cls, raytracer, dataset, pipe, gaussians, separate_sh, mode: ViewerMode):
        viewer = cls(mode, raytracer)
        viewer.dataset = dataset
        viewer.pipe = pipe
        viewer.gaussians = gaussians
        viewer.separate_sh = separate_sh
        viewer.background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        # viewer.camera_poses = json.load(open(os.path.join(model_path, "cameras.json"), "r")) # todo fix
        return viewer

    def create_widgets(self):
        self.camera = FPSCamera(self.mode, self.raytracer.image_width if self.raytracer is not None else 600, self.raytracer.image_height if self.raytracer is not None else 400, 47, 0.001, 100)
        self.point_view = TorchImage(self.mode)
        self.ellipsoid_viewer = EllipsoidViewer(self.mode)
        self.monitor = PerformanceMonitor(self.mode, ["Render"], add_other=False)

        # Render modes
        self.render_modes = ["RGB", "Normals", "Position", "F0", "Roughness", "Illumination", "Ellipsoids"]
        self.render_mode = 0
        
        self.ray_choices = ["All/Default"] + ["Ray " + str(i) for i in range(self.ray_count)] 
        self.ray_choice = 0

        self.selection_choice = 0
        self.selection_choices = ["none"]
        if self.raytracer is not None:
            self.selection_choices = ["none"] + [x for x in self.bounding_boxes.keys()]
        else:
            self.selection_choices = ["none"] 
        
        # Render settings
        self.exposure = 1.0
        self.scaling_modifier = 1.0

        self.tool = "pan" # pan, select, move, scale, or rotate
        self.hovering_over = None

        # Editing
        if self.mode == ViewerMode.CLIENT:
            self.edit = None
        else:
            self.edit = Edit()
            self.set_camera_pose(self.train_transforms, 0)

    def set_camera_pose(self, transforms, i: int):
        self.camera.update_pose(np.array(transforms["frames"][i]["transform_matrix"]) @ self.blender_to_opengl)
        self.camera.fov_x = transforms["camera_angle_x"]
        self.camera.fov_y = transforms["camera_angle_y"]

    def update_bbox_selection(self):
        if self.edits is not None and self.selection_choice != 0:
            self.edit = self.edits[self.selection_choices[self.selection_choice]]

    def duplicate_selection(self):
        # For this to work with the remote client, this doesn't actually produce the copy, 
        # just adds the object to the list which the server notices
        old_key = self.selection_choices[self.selection_choice] 
        new_key = old_key + "_copy"
        self.selection_choices.insert(self.selection_choices.index(old_key) + 1, new_key)
        self.edits[new_key] = Edit()
        old_edit = self.edits[old_key]
        self.bounding_boxes[new_key] = self.bounding_boxes[old_key]
        for j in ["min", "max"]:
            self.bounding_boxes[new_key][j][0] += DUPLICATION_OFFSET + old_edit.translate_x
            self.bounding_boxes[new_key][j][1] += DUPLICATION_OFFSET + old_edit.translate_y
            self.bounding_boxes[new_key][j][2] += DUPLICATION_OFFSET + old_edit.translate_z
        self.selection_choice = self.selection_choices.index(new_key)
        self.update_bbox_selection()

    def step(self):
        world_to_view = torch.from_numpy(self.camera.to_camera).cuda().transpose(0, 1)
        full_proj_transform = torch.from_numpy(self.camera.full_projection).cuda().transpose(0, 1)
        
        camera = MiniCam(self.camera.res_x, self.camera.res_y, self.camera.fov_y, self.camera.fov_x, self.camera.z_near, self.camera.z_far, world_to_view, full_proj_transform)

        if self.ellipsoid_viewer.num_gaussians is None:
           self.ellipsoid_viewer.upload(
               self.gaussians.get_xyz.detach().cpu().numpy(),
               self.gaussians.get_rotation.detach().cpu().numpy(),
               self.gaussians.get_scaling.detach().cpu().numpy(),
               self.gaussians.get_opacity.detach().cpu().numpy(),
               self.gaussians.get_diffuse.detach().cpu().numpy()
           )
        
        #if self.render_mode == 0:
        if self.render_modes[self.render_mode] == "Ellipsoids":
           self.ellipsoid_viewer.step(self.camera)
           render_time = glGetQueryObjectuiv(self.ellipsoid_viewer.query, GL_QUERY_RESULT) / 1e6
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                with self.gaussian_lock:
                    self.gaussians.dirty_check(self.scaling_modifier)
                    self.camera.dirty_check()

                    if self.tool == "select" and self.last_rendered_selection_mask_id != self.selection_mode_counter:
                        # Render masks for point and click selection
                        self.gaussians.is_dirty = True
                        self.raytracer.cuda_module.accumulate.copy_(False)
                        accum_rgb_backup = self.raytracer.cuda_module.accumulated_rgb.clone()
                        accum_sample_count_backup = self.raytracer.cuda_module.accumulated_sample_count.clone()
                        self.raytracer.cuda_module.accumulated_rgb.zero_()
                        self.raytracer.cuda_module.accumulated_sample_count.zero_()
                        for obj_name in self.bounding_boxes.keys():
                            if obj_name == "everything":
                                continue
                            rgb_backup = self.gaussians._diffuse.clone()
                            rgb = self.gaussians._diffuse
                            rgb *= 0 
                            rgb[self.gaussians.selections[obj_name].squeeze(1)] += 1
                            package = render(camera, self.raytracer, self.pipe, self.background, blur_sigma=None, targets_available=False)
                            mask_render = package.rgb[0].mean(dim=0).cpu().numpy()
                            self.selection_masks[obj_name] = mask_render
                            self.gaussians._diffuse.copy_(rgb_backup)
                        self.last_rendered_selection_mask_id = self.selection_mode_counter
                        self.raytracer.cuda_module.accumulated_rgb.copy_(accum_rgb_backup)
                        self.raytracer.cuda_module.accumulated_sample_count.copy_(accum_sample_count_backup)

                    for key in self.edits.keys():
                        # Duplicates are produced here
                        if key not in self.gaussians.created_objects:
                            self.gaussians.duplicate_selected(key.replace("_copy", "", 1), DUPLICATION_OFFSET)
                            self.selection_choices.append(key)
                            self.raytracer.rebuild_bvh()
                    
                    self.update_bbox_selection() 
                    
                    if self.gaussians.is_dirty or self.camera.is_dirty or not self.accumulate_samples:
                        self.raytracer.cuda_module.accumulated_rgb.zero_()
                        self.raytracer.cuda_module.accumulated_sample_count.zero_()

                    self.raytracer.cuda_module.accumulate.copy_(self.accumulate_samples)
                    self.raytracer.cuda_module.denoise.copy_(self.denoise)
                    self.raytracer.cuda_module.global_scale_factor.copy_(self.scaling_modifier)
                    package = render(camera, self.raytracer, self.pipe, self.background, blur_sigma=None, targets_available=False, force_update_bvh=self.gaussians.is_dirty)
                    self.raytracer.cuda_module.global_scale_factor.copy_(1.0)

                    mode_name = self.render_modes[self.render_mode]
                    nth_ray = self.ray_choice - 1
                    if mode_name == "RGB":
                        if nth_ray == -1:
                            net_image = package.rgb[-1] 
                        elif self.sum_rgb_passes:
                            net_image = tonemap(untonemap(package.rgb[:nth_ray + 1]).sum(dim=0))
                        else:
                            net_image = package.rgb[nth_ray]
                    elif mode_name == "Diffuse":
                        net_image = package.rgb[max(nth_ray, 0)]
                    elif mode_name == "F0":
                        net_image = package.F0[max(nth_ray, 0)]
                    elif mode_name == "Normals":
                        net_image = package.normal[max(nth_ray, 0)] / 2 + 0.5
                    elif mode_name == "Position":
                        net_image = package.position[max(nth_ray, 0)]
                    elif mode_name == "Illumination":
                        net_image = self.raytracer.cuda_module.output_incident_radiance[max(nth_ray, 0)].moveaxis(-1, 0)
                    elif mode_name == "Roughness":
                        net_image = package.roughness[max(nth_ray, 0)]

                if mode_name == "RGB":
                    net_image = tonemap(untonemap(net_image.permute(1, 2, 0))*self.exposure) 
                else:
                    net_image = net_image.permute(1, 2, 0)*self.exposure
            if self.hovering_over is not None:
                overlay = torch.tensor(self.selection_masks[self.hovering_over]).cuda().unsqueeze(-1).repeat(1, 1, 3)
                net_image[:, :, 0] += overlay[:, :, 0] * 0.15
                net_image[:, :, 1] += overlay[:, :, 1] * 0.10
            end.record()
            end.synchronize()
            self.point_view.step(net_image)
            render_time = start.elapsed_time(end)

        self.monitor.step([render_time])

    def show_gui(self):
        gizmo.begin_frame()

        with imgui_ctx.begin(f"Point View Settings"):
            did_disable = False
            if self.tool == "select":
                imgui.begin_disabled()
                did_disable = True

            _, self.render_mode = imgui.list_box("Render Mode", self.render_mode, self.render_modes)
            _, self.ray_choice = imgui.list_box("Displayed Rays", self.ray_choice, self.ray_choices)
            _, self.sum_rgb_passes = imgui.checkbox("Cumulative Total RGB", self.sum_rgb_passes)
            

            imgui.separator_text("Render Settings")

            if self.render_modes[self.render_mode] == "Ellipsoids":
                _, self.ellipsoid_viewer.scaling_modifier = imgui.drag_float("Scaling Factor", self.ellipsoid_viewer.scaling_modifier, v_min=0, v_max=10, v_speed=0.01)
                
                _, self.ellipsoid_viewer.render_floaters = imgui.checkbox("Render Floaters", self.ellipsoid_viewer.render_floaters)
                _, self.ellipsoid_viewer.limit = imgui.drag_float("Alpha Threshold", self.ellipsoid_viewer.limit, v_min=0, v_max=1, v_speed=0.01)
            else:
                _, self.scaling_modifier = imgui.drag_float("Scaling Factor", self.scaling_modifier, v_min=0, v_max=10, v_speed=0.01)
                
                _, self.exposure = imgui.drag_float("Exposure", self.exposure, v_min=0, v_max=6, v_speed=0.01)

            _, self.accumulate_samples = imgui.checkbox("Accumulate Samples", self.accumulate_samples)
            imgui.same_line()
            _, self.denoise = imgui.checkbox("Denoise", self.denoise)

            imgui.separator_text("Camera Settings")

            if self.train_transforms is not None:
                using_train_cam = self.current_train_cam != -1
                if not using_train_cam:
                    imgui.push_style_color(imgui.Col_.frame_bg, (0.0, 0.0, 0.0, 0.0)) 
                    imgui.push_style_color(imgui.Col_.frame_bg_hovered, (0.0, 0.0, 0.0, 0.0))
                    imgui.push_style_color(imgui.Col_.frame_bg_active, (0.0, 0.0, 0.0, 0.0))
                train_cam_changed, self.current_train_cam = imgui.input_int("Set Train View", self.current_train_cam, step=1, step_fast=10)
                if not train_cam_changed and imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    train_cam_changed = True
                    self.current_train_cam = 0
                if not using_train_cam:
                    imgui.pop_style_color(3)
                self.current_train_cam = max(-1, min(len(self.train_transforms["frames"]) - 1, self.current_train_cam))

                using_test_cam = self.current_test_cam != -1
                if not using_test_cam:
                    imgui.push_style_color(imgui.Col_.frame_bg, (0.0, 0.0, 0.0, 0.0)) 
                    imgui.push_style_color(imgui.Col_.frame_bg_hovered, (0.0, 0.0, 0.0, 0.0))
                    imgui.push_style_color(imgui.Col_.frame_bg_active, (0.0, 0.0, 0.0, 0.0))
                test_cam_changed, self.current_test_cam = imgui.input_int("Set Test View", self.current_test_cam, step=1, step_fast=10)
                if not test_cam_changed and imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    test_cam_changed = True
                    self.current_test_cam = 0
                if not using_test_cam:
                    imgui.pop_style_color(3)
                self.current_test_cam = max(-1, min(len(self.test_transforms["frames"]) - 1, self.current_test_cam))
                
                if train_cam_changed:
                    self.camera.update_pose(np.array(self.train_transforms["frames"][self.current_train_cam]["transform_matrix"]) @ self.blender_to_opengl)
                    self.current_test_cam = -1
                elif test_cam_changed:
                    self.camera.update_pose(np.array(self.test_transforms["frames"][self.current_test_cam]["transform_matrix"]) @ self.blender_to_opengl)
                    self.current_train_cam = -1
                
            self.camera.show_gui()

            if did_disable:
                imgui.end_disabled()


        with imgui_ctx.begin(f"Editing Settings"):
            did_disable = False
            if self.tool == "select":
                imgui.begin_disabled()
                did_disable = True

            imgui.separator_text("Selection")

            clicked, self.selection_choice = imgui.combo("Object List", self.selection_choice, self.selection_choices)
            if clicked:
                self.update_bbox_selection()    
            
            if did_disable:
                imgui.end_disabled()
            # clicked = imgui.button("Point and Click", size=(240, 24))
            # if clicked:
            #     self.enter_selection_mode()

            if did_disable:
                imgui.begin_disabled()

            imgui.spacing() 
            imgui.spacing() 
            
            disabled_cause_no_selection = False
            if self.selection_choice == 0:
                imgui.begin_disabled()
                disabled_cause_no_selection = True

            clicked = imgui.button("Duplicate Selection", size=(240, 24))
            if clicked:
                self.duplicate_selection()
                

            clicked = imgui.button("Reset Selection", size=(240, 24))
            if clicked and self.edits is not None and self.selection_choice != 0:
                for key, default_value in dataclasses.asdict(Edit()).items():
                    setattr(self.edits[self.selection_choices[self.selection_choice]], key, default_value)
            clicked = imgui.button("Reset Everything", size=(240, 24))
            if clicked and self.edits is not None:
                for edit in self.edits.values():
                    for key, default_value in dataclasses.asdict(Edit()).items():
                        setattr(edit, key, default_value)
            
            imgui.separator_text("BRDF Editing")

            imgui.set_cursor_pos_x((imgui.get_content_region_avail().x - imgui.calc_text_size("Roughness").x) * 0.35)
            imgui.text("Roughness")
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.595)
            _, self.edit.roughness_override = imgui.drag_float("##Roughness Override", self.edit.roughness_override, v_min=0, v_max=1, v_speed=0.01/2)
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.roughness_override = 0.0
            imgui.pop_item_width()
            imgui.same_line()
            _, self.edit.use_roughness_override = imgui.checkbox("##Use Roughness Override", self.edit.use_roughness_override)
            imgui.same_line()
            imgui.text("Override")
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.329)
            _, self.edit.roughness_shift = imgui.drag_float("##Roughness Shift", self.edit.roughness_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.roughness_shift = 0.0
            imgui.same_line()
            _, self.edit.roughness_mult = imgui.drag_float("##Roughness Mult", self.edit.roughness_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.roughness_mult = 1.0
            imgui.same_line()
            imgui.text("Adjust")
            imgui.pop_item_width()

            imgui.spacing() 
            imgui.spacing() 

            imgui.spacing() 
            
            imgui.set_cursor_pos_x((imgui.get_content_region_avail().x - imgui.calc_text_size("Diffuse").x) * 0.35)
            imgui.text("Diffuse")
            _, self.edit.diffuse_override = imgui.color_edit4("##Diffuse Override", self.edit.diffuse_override, flags=imgui.ColorEditFlags_.no_options | imgui.ColorEditFlags_.alpha_preview)
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.diffuse_override = (0.0, 0.0, 0.0, 0.0)
            imgui.same_line()
            imgui.text("Override")
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.68)
            _, self.edit.diffuse_hue_shift = imgui.drag_float("##Diffuse Hue Shift", self.edit.diffuse_hue_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.diffuse_hue_shift = 0.0
            imgui.same_line()
            imgui.text("Hue")
            imgui.pop_item_width()
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.329)
            _, self.edit.diffuse_saturation_shift = imgui.drag_float("##Diffuse Saturation Shift", self.edit.diffuse_saturation_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.diffuse_saturation_shift = 0.0
            imgui.same_line()
            _, self.edit.diffuse_saturation_mult = imgui.drag_float("##Diffuse Saturation Mult", self.edit.diffuse_saturation_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.diffuse_saturation_mult = 1.0
            imgui.same_line()
            imgui.text("Saturation")
            _, self.edit.diffuse_value_shift = imgui.drag_float("##Diffuse Value Shift", self.edit.diffuse_value_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.diffuse_value_shift = 0.0
            imgui.same_line()
            _, self.edit.diffuse_value_mult = imgui.drag_float("##Diffuse Value Mult", self.edit.diffuse_value_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.diffuse_value_mult = 1.0
            imgui.same_line()
            imgui.text("Value")
            imgui.pop_item_width()

            imgui.spacing() 
            imgui.spacing() 

            imgui.set_cursor_pos_x((imgui.get_content_region_avail().x - imgui.calc_text_size("Specular").x) * 0.35)
            imgui.text("Specular")
            _, self.edit.glossy_override = imgui.color_edit4("##Specular Override", self.edit.glossy_override, flags=imgui.ColorEditFlags_.no_options | imgui.ColorEditFlags_.alpha_preview)
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.glossy_override = (0.0, 0.0, 0.0, 0.0)
            imgui.same_line()
            imgui.text("Override")
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.68)
            _, self.edit.glossy_hue_shift = imgui.drag_float("##Specular Hue Shift", self.edit.glossy_hue_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.glossy_hue_shift = 0.0
            imgui.same_line()
            imgui.text("Hue")
            imgui.pop_item_width()
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.329)
            _, self.edit.glossy_saturation_shift = imgui.drag_float("##Specular Saturation Shift", self.edit.glossy_saturation_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.glossy_saturation_shift = 0.0
            imgui.same_line()
            _, self.edit.glossy_saturation_mult = imgui.drag_float("##Specular Saturation Mult", self.edit.glossy_saturation_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.glossy_saturation_mult = 1.0
            imgui.same_line()
            imgui.text("Saturation")
            _, self.edit.glossy_value_shift = imgui.drag_float("##Specular Value Shift", self.edit.glossy_value_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.glossy_value_shift = 0.0
            imgui.same_line()
            _, self.edit.glossy_value_mult = imgui.drag_float("##Specular Value Mult", self.edit.glossy_value_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.glossy_value_mult = 1.0
            imgui.same_line()
            imgui.text("Value")
            imgui.pop_item_width()

            imgui.spacing() 
            imgui.spacing() 

            imgui.separator_text("Geometric Editing")

            imgui.spacing() 

            imgui.push_item_width(imgui.get_content_region_avail().x * 0.21)
            _, self.edit.translate_x = imgui.drag_float("##Translate X", self.edit.translate_x, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.translate_x = 0.0
            imgui.same_line()
            _, self.edit.translate_y = imgui.drag_float("##Translate Y", self.edit.translate_y, v_min=-1, v_max=1, v_speed=0.01, format="+%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.translate_y = 0.0
            imgui.same_line()
            _, self.edit.translate_z = imgui.drag_float("##Translate Z", self.edit.translate_z, v_min=-1, v_max=1, v_speed=0.01, format="+%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.translate_z = 0.0
            imgui.same_line()
            imgui.text("Translation")
            imgui.pop_item_width()

            imgui.push_item_width(imgui.get_content_region_avail().x * 0.21)
            _, self.edit.rotate_x = imgui.drag_float("##Rotate X", self.edit.rotate_x, v_min=-180, v_max=180, v_speed=1.0, format="%+.1f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.rotate_x = 0.0
            imgui.same_line()
            _, self.edit.rotate_y = imgui.drag_float("##Rotate Y", self.edit.rotate_y, v_min=-180, v_max=180, v_speed=1.0, format="%+.1f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.rotate_y = 0.0
            imgui.same_line()
            _, self.edit.rotate_z = imgui.drag_float("##Rotate Z", self.edit.rotate_z, v_min=-180, v_max=180, v_speed=1.0, format="%+.1f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.rotate_z = 0.0
            imgui.same_line()
            imgui.text("Rotation")
            imgui.pop_item_width()

            imgui.push_item_width(imgui.get_content_region_avail().x * 0.21)
            _, self.edit.scale_x = imgui.drag_float("##Scale X", self.edit.scale_x, v_min=0.01, v_max=10.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.scale_x = 1.0
            imgui.same_line()
            _, self.edit.scale_y = imgui.drag_float("##Scale Y", self.edit.scale_y, v_min=0.01, v_max=10.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.scale_y = 1.0
            imgui.same_line()
            _, self.edit.scale_z = imgui.drag_float("##Scale Z", self.edit.scale_z, v_min=0.01, v_max=10.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.scale_z = 1.0
            imgui.same_line()
            imgui.text("Scale")
            imgui.pop_item_width()

            imgui.spacing() 
        
            if disabled_cause_no_selection:
                imgui.end_disabled()

            if did_disable:
                imgui.end_disabled()

        with imgui_ctx.begin("Point View"):
            if self.render_modes[self.render_mode] == "Ellipsoids":
               self.ellipsoid_viewer.show_gui()
            else:
                image_top_left_corner = imgui.get_cursor_screen_pos()
                self.point_view.show_gui()

                if self.selection_choice == 0:
                    cam_changed_from_mouse = imgui.is_item_hovered() and self.camera.process_mouse_input()
                else:
                    cam_changed_from_mouse = False
            
                cam_changed_from_keyboard = (imgui.is_item_focused() or imgui.is_item_hovered()) and self.camera.process_keyboard_input()
                if cam_changed_from_mouse or cam_changed_from_keyboard:
                    self.current_train_cam = -1
                    self.current_test_cam = -1
                
                toolbar_width = 75
                toolbar_height = self.camera.res_y
                toolbar_x = image_top_left_corner.x + self.camera.res_x + 10
                toolbar_y = image_top_left_corner.y

                imgui.set_cursor_screen_pos((toolbar_x, toolbar_y))
                imgui.begin_child("Toolbar", (toolbar_width, toolbar_height))

                init_tool = self.tool
                if init_tool == "pan":
                    imgui.push_style_color(imgui.Col_.button, (0.2, 0.5, 0.2, 1.0))  # Highlight color
                if imgui.button("Pan", size=(toolbar_width - 10, 40)):
                    self.tool = "pan"
                if init_tool == "pan":
                    imgui.pop_style_color()

                if init_tool == "select":
                    imgui.push_style_color(imgui.Col_.button, (0.2, 0.5, 0.2, 1.0))  # Highlight color
                if imgui.button("Select", size=(toolbar_width - 10, 40)):
                    self.enter_selection_mode()
                if init_tool == "select":
                    imgui.pop_style_color()

                if init_tool == "move":
                    imgui.push_style_color(imgui.Col_.button, (0.2, 0.5, 0.2, 1.0))  # Highlight color
                if imgui.button("Move", size=(toolbar_width - 10, 40)):
                    self.tool = "move"
                if init_tool == "move":
                    imgui.pop_style_color()

                if init_tool == "scale":
                    imgui.push_style_color(imgui.Col_.button, (0.2, 0.5, 0.2, 1.0))  # Highlight color
                if imgui.button("Scale", size=(toolbar_width - 10, 40)):
                    self.tool = "scale"
                if init_tool == "scale":
                    imgui.pop_style_color()

                if init_tool == "rotate":
                    imgui.push_style_color(imgui.Col_.button, (0.2, 0.5, 0.2, 1.0))  # Highlight color
                if imgui.button("Rotate", size=(toolbar_width - 10, 40)):
                    self.tool = "rotate"
                if init_tool == "rotate":
                    imgui.pop_style_color()

                imgui.end_child()

                if self.tool == "select":
                    mouse_pos = imgui.get_mouse_pos()
                    window_pos = imgui.get_window_pos()
                    j, i = int(mouse_pos[0] - window_pos.x), int(mouse_pos[1] - window_pos.y - 30)
                    IMAGE_HEIGHT = 400 
                    IMAGE_WIDTH = 600 
                    i = min(max(0, i), IMAGE_HEIGHT - 1)
                    j = min(max(0, j), IMAGE_WIDTH - 1)
                    if False:
                        print(f"Mouse at pixel position: ({i}, {j})")
                    self.hovering_over = None
                    for bbox_name, mask in self.selection_masks.items():
                        if mask[i, j]:
                            self.hovering_over = bbox_name
                    if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                        for bbox_name, mask in self.selection_masks.items():
                            if mask[i, j]:
                                self.selection_choice = self.selection_choices.index(bbox_name)
                                self.update_bbox_selection()
                        self.tool = "move"
                    elif imgui.is_mouse_clicked(imgui.MouseButton_.right):
                        self.tool = "pan"
                else:
                    self.hovering_over = None

                if self.tool in ["move", "scale", "rotate"] and self.selection_choice != len(self.selection_choices) - 1:
                    if self.selection_choice == 0:
                        self.tool = "pan"
                    else:
                        gizmo.set_drawlist()

                        gizmo.set_rect(image_top_left_corner.x, image_top_left_corner.y, self.camera.res_x, self.camera.res_y)

                        to_camera = self.camera.to_camera
                        to_camera[1] *= -1

                        view_mat = Matrix16((to_camera.T).flatten().tolist())
                        proj_mat = Matrix16((self.camera.projection.T).flatten().tolist())
                        bbox = self.bounding_boxes[self.selection_choices[self.selection_choice]]
                        original_x = (bbox["min"][0] + bbox["max"][0]) / 2
                        original_y = (bbox["min"][1] + bbox["max"][1]) / 2
                        original_z = (bbox["min"][2] + bbox["max"][2]) / 2

                        rotation_x_matrix = np.array([
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, math.cos(math.radians(self.edit.rotate_x)), -math.sin(math.radians(self.edit.rotate_x)), 0.0],
                            [0.0, math.sin(math.radians(self.edit.rotate_x)), math.cos(math.radians(self.edit.rotate_x)), 0.0],
                            [0.0, 0.0, 0.0, 1.0]
                        ])
                        rotation_y_matrix = np.array([
                            [math.cos(math.radians(self.edit.rotate_y)), 0.0, math.sin(math.radians(self.edit.rotate_y)), 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [-math.sin(math.radians(self.edit.rotate_y)), 0.0, math.cos(math.radians(self.edit.rotate_y)), 0.0],
                            [0.0, 0.0, 0.0, 1.0]
                        ])
                        rotation_z_matrix = np.array([
                            [math.cos(math.radians(self.edit.rotate_z)), -math.sin(math.radians(self.edit.rotate_z)), 0.0, 0.0],
                            [math.sin(math.radians(self.edit.rotate_z)), math.cos(math.radians(self.edit.rotate_z)), 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]
                        ])

                        rotation_matrix = rotation_x_matrix @ rotation_y_matrix @ rotation_z_matrix

                        pose = Matrix16([
                            self.edit.scale_x * rotation_matrix[0, 0], self.edit.scale_x * rotation_matrix[0, 1], self.edit.scale_x * rotation_matrix[0, 2], 0.0,
                            self.edit.scale_y * rotation_matrix[1, 0], self.edit.scale_y * rotation_matrix[1, 1], self.edit.scale_y * rotation_matrix[1, 2], 0.0,
                            self.edit.scale_z * rotation_matrix[2, 0], self.edit.scale_z * rotation_matrix[2, 1], self.edit.scale_z * rotation_matrix[2, 2], 0.0,
                            original_x + self.edit.translate_x, original_y + self.edit.translate_y, original_z + self.edit.translate_z, 1.0
                        ])

                        gizmo_mode = dict(move=gizmo.OPERATION.translate, scale=gizmo.OPERATION.scale, rotate=gizmo.OPERATION.rotate)[self.tool]
                        gizmo.manipulate(view_mat, proj_mat, gizmo_mode, gizmo.MODE.local, pose, None, None, None, None)
                        
                        scale_x, scale_y, scale_z = np.linalg.norm(pose.values[0:3]), np.linalg.norm(pose.values[4:7]), np.linalg.norm(pose.values[8:11])

                        rotation_matrix = np.array(pose.values).reshape(4, 4)[:3, :3] / [scale_x, scale_y, scale_z]

                        translate_x, translate_y, translate_z = pose.values[12], pose.values[13], pose.values[14]

                        rotate_x, rotate_y, rotate_z = np.degrees(np.arctan2(
                            [rotation_matrix[2, 1], -rotation_matrix[2, 0], rotation_matrix[1, 0]],
                            [rotation_matrix[2, 2], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2), rotation_matrix[0, 0]]
                        ))

                        self.edit.translate_x = float(translate_x - original_x)
                        self.edit.translate_y = float(translate_y - original_y)
                        self.edit.translate_z = float(translate_z - original_z)
                        self.edit.scale_x = float(scale_x)
                        self.edit.scale_y = float(scale_y)
                        self.edit.scale_z = float(scale_z)
                        self.edit.rotate_x = float(rotate_x)
                        self.edit.rotate_y = float(rotate_y)
                        self.edit.rotate_z = float(rotate_z)

                        # self.edit.translate_x = float(pose.values[12] - original_x)
                        # self.edit.translate_y = float(pose.values[13] - original_y)
                        # self.edit.translate_z = float(pose.values[14] - original_z)
                        # self.edit.scale_x = float(pose.values[0])
                        # self.edit.scale_y = float(pose.values[5])
                        # self.edit.scale_z = float(pose.values[10])

            
        
        with imgui_ctx.begin("Performance"):
            self.monitor.show_gui()

        if self.tool == "select":
            mouse_pos = imgui.get_mouse_pos()
            draw_list = imgui.get_foreground_draw_list()
            color = imgui.color_convert_float4_to_u32((1.0, 1.0, 0.0, 0.7))  
            draw_list.add_circle_filled((mouse_pos[0], mouse_pos[1]), 3.0, color)
    
    def enter_selection_mode(self):
        self.tool = "select"
        self.selection_choice = 0
        self.selection_mode_counter += 1

    def client_send(self):
        return None, {
            "scaling_modifier": self.scaling_modifier,
            "render_mode": self.render_mode,
            "exposure": self.exposure,
            "ray_choice": self.ray_choice,
            "selection_choice": self.selection_choice,
            "hovering_over": self.hovering_over,
            "edits": { key: dataclasses.asdict(edit) for key, edit in self.edits.items() } if self.edits is not None else None,
            "tool": self.tool,
            "selection_mode_counter": self.selection_mode_counter,
            "sum_rgb_passes": self.sum_rgb_passes
        }
    
    def server_recv(self, _, text):
        self.scaling_modifier = text["scaling_modifier"]
        self.render_mode = text["render_mode"]
        self.ray_choice = text["ray_choice"]
        self.selection_choice = text["selection_choice"]
        self.exposure = text["exposure"]
        self.hovering_over = text["hovering_over"]
        self.tool = text["tool"]
        self.selection_mode_counter = text["selection_mode_counter"]
        self.sum_rgb_passes = text["sum_rgb_passes"]
        
        if text["edits"] is not None:
            for key, edit in text["edits"].items():
                self.edits[key] = Edit(**edit)

    def server_send(self):
        if self.first_send:
            data = {
                "ray_count": self.ray_count,
                "selection_choices": self.selection_choices,
                "train_transforms": self.train_transforms,
                "test_transforms": self.test_transforms,
                "bounding_boxes": self.bounding_boxes,
                "selection_masks": {
                    k: v.tolist() for k, v in self.selection_masks.items()
                },
            }
        else:
            data = {}
        return None, data
    
    def client_recv(self, _, text):
        if "ray_count" in text and self.ray_count != text["ray_count"]:
            self.ray_count = text["ray_count"]
            self.ray_choices = ["All/Default"] + ["Ray " + str(i) for i in range(self.ray_count)]
        if "train_transforms" in text:
            self.train_transforms = text["train_transforms"]
            self.test_transforms = text["test_transforms"]
            self.set_camera_pose(self.train_transforms, 0)
        if "selection_choices" in text:
            self.selection_choices = text["selection_choices"]
        if "bounding_boxes" in text:
            self.bounding_boxes = text["bounding_boxes"]
            self.edits = { bbox_name: Edit() for bbox_name in self.bounding_boxes.keys() }
        if "selection_masks" in text:
            self.selection_masks = {
                k: np.array(v) for k, v in text["selection_masks"].items()
            }
        
    
if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)
    local = subparsers.add_parser("local")
    local.add_argument("model_path")
    local.add_argument("iter", type=int, default=30000)
    client = subparsers.add_parser("client")
    client.add_argument("--ip", default="localhost")
    client.add_argument("--port", type=int, default=8000)
    server = subparsers.add_parser("server")
    server.add_argument("model_path")
    server.add_argument("iter", type=int, default=30000)
    server.add_argument("--ip", default="localhost")
    server.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    match args.mode:
        case "local":
            mode = ViewerMode.LOCAL
        case "client":
            mode = ViewerMode.CLIENT
        case "server":
            mode = ViewerMode.SERVER

    if mode is ViewerMode.CLIENT:
        viewer = GaussianViewer(mode, None)
    else:
        viewer = GaussianViewer.from_ply(args.model_path, args.iter, mode)

    if args.mode in ["client", "server"]:
        viewer.run(args.ip, args.port)
    else:
        viewer.run()

    #  xvfb-run -s "-screen 0 1400x900x24" python gaussianviewer.py server output/tmp2 7000 --ip 0.0.0.0