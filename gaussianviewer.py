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

gizmo = imguizmo.im_guizmo
Matrix3 = gizmo.Matrix3
Matrix6 = gizmo.Matrix6
Matrix16 = gizmo.Matrix16

class Dummy(object):
    pass

@dataclass
class Edit:
    roughness_shift: float = 0.0
    roughness_mult: float = 1.0

    diffuse_override: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    diffuse_hue_shift: float = 0.0
    diffuse_saturation_shift: float = 0.0
    diffuse_saturation_mult: float = 1.0
    diffuse_value_shift: float = 0.0
    diffuse_value_mult: float = 1.0
    
    roughness_override: float = 0.0

    glossy_override: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    glossy_hue_shift: float = 0.0
    glossy_saturation_shift: float = 0.0
    glossy_saturation_mult: float = 1.0
    glossy_value_shift: float = 0.0
    glossy_value_mult: float = 1.0

    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0

    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0

class GaussianViewer(Viewer):
    def __init__(self, mode: ViewerMode, raytracer: "GaussianRaytracer"):
        super().__init__(mode)
        self.window_title = "Gaussian Viewer"
        self.gaussian_lock = Lock()
        self.raytracer = raytracer
        if self.raytracer is not None:
            self.ray_count = self.raytracer.config.MAX_BOUNCES + 1
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

    def import_server_modules(self):
        global torch
        import torch

        global GaussianModel
        from scene import GaussianModel

        global EditableGaussianModel
        from scene import EditableGaussianModel

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
        global GaussianModel
        from scene import GaussianModel
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


        gaussians = GaussianModel(model_params)
        ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iter}", "point_cloud.ply")
        gaussians.load_ply(ply_path)
        raytracer = GaussianRaytracer(gaussians, int(model_params.resolution*1.5), model_params.resolution)

        viewer = cls(mode, raytracer)
        viewer.separate_sh = False
        viewer.gaussians = gaussians
        viewer.dataset = dataset
        viewer.pipe = pipe

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        viewer.background = background

        source_path = model_params.source_path
        viewer.train_transforms = json.load(open(os.path.join(source_path, "transforms_train.json"), "r"))
        viewer.test_transforms = json.load(open(os.path.join(source_path, "transforms_test.json"), "r"))

        viewer.bounding_boxes = json.load(open(os.path.join(source_path, "bounding_boxes.json"), "r"))
        viewer.selection_masks = {}
        for bbox_name in viewer.bounding_boxes.keys():
            mask = (np.array(Image.open(os.path.join(source_path, f"selection_masks/{bbox_name}.png")).convert("RGB")).mean(axis=2) > 0.5).astype(bool)
            viewer.selection_masks[bbox_name] = mask
        return viewer
    
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
        self.camera = FPSCamera(self.mode, self.raytracer.image_width if self.raytracer is not None else 800, self.raytracer.image_height if self.raytracer is not None else 600, 47, 0.001, 100)
        self.point_view = TorchImage(self.mode)
        self.ellipsoid_viewer = EllipsoidViewer(self.mode)
        self.monitor = PerformanceMonitor(self.mode, ["Render"], add_other=False)

        # Render modes
        self.render_modes = ["RGB", "Normals", "Position", "F0", "Roughness", "Illumination", "Ellipsoids"]
        self.render_mode = 0
        
        self.ray_choices = ["All/Default"] + ["Ray " + str(i) for i in range(self.ray_count)] 
        self.ray_choice = 0

        self.selection_choice = 0
        self.selection_choices = ["None"]
        if self.raytracer is not None:
            self.selection_choices = ["None"] + [x for x in self.bounding_boxes.keys()]
        else:
            self.selection_choices = ["None"]
        
        # Render settings
        self.exposure = 1.0
        self.scaling_modifier = 1.0
        
        

        self.in_selection_mode = False

        # Editing
        self.reset_brdf_edit_settings()
        if self.raytracer is not None:
            self.selection = torch.zeros_like(self.raytracer.pc._roughness).bool()

    def update_bbox_selection(self):
        if self.raytracer is not None and self.selection_choice != 0:
            bounding_box = self.bounding_boxes[self.selection_choices[self.selection_choice]]
            dist1 = self.raytracer.pc._xyz - (torch.tensor(bounding_box["min"], device="cuda"))
            dist2 = self.raytracer.pc._xyz - (torch.tensor(bounding_box["max"], device="cuda"))
            within_bbox = (dist1 >= 0).all(dim=-1) & (dist2 <= 0).all(dim=-1)
            self.selection = within_bbox.unsqueeze(1)

    def reset_brdf_edit_settings(self):
        self.edit = Edit()

    def step(self):
        camera = self.camera
        world_to_view = torch.from_numpy(camera.to_camera).cuda().transpose(0, 1)
        full_proj_transform = torch.from_numpy(camera.full_projection).cuda().transpose(0, 1)
        
        camera = MiniCam(camera.res_x, camera.res_y, self.train_transforms["camera_angle_y"], self.train_transforms["camera_angle_x"], camera.z_near, camera.z_far, world_to_view, full_proj_transform)

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
                    self.raytracer.cuda_module.global_scale_factor.copy_(self.scaling_modifier)
                    if self.scaling_modifier != 1.0:
                        self.raytracer.cuda_module.update_bvh()

                    self.update_bbox_selection() 

                    package = render(camera, self.raytracer, self.pipe, self.background, blur_sigma=None, targets_available=False, edits=dict(
                        roughness_shift=self.edit.roughness_shift,
                        roughness_mult=self.edit.roughness_mult,
                        diffuse_hue_shift=self.edit.diffuse_hue_shift,
                        diffuse_saturation_shift=self.edit.diffuse_saturation_shift,
                        diffuse_saturation_mult=self.edit.diffuse_saturation_mult,
                        diffuse_value_shift=self.edit.diffuse_value_shift,
                        diffuse_value_mult=self.edit.diffuse_value_mult,
                        glossy_hue_shift=self.edit.glossy_hue_shift,
                        glossy_saturation_shift=self.edit.glossy_saturation_shift,
                        glossy_saturation_mult=self.edit.glossy_saturation_mult,
                        glossy_value_shift=self.edit.glossy_value_shift,
                        glossy_value_mult=self.edit.glossy_value_mult,
                        roughness_override=self.edit.roughness_override,
                        diffuse_override=self.edit.diffuse_override,
                        glossy_override=self.edit.glossy_override,
                        selection=self.selection,
                    ))
                    
                    self.raytracer.cuda_module.global_scale_factor.copy_(1.0)

                    mode_name = self.render_modes[self.render_mode]
                    nth_ray = self.ray_choice - 1
                    if mode_name == "RGB":
                        if nth_ray == -1:
                            net_image = package.rgb[-1]
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
            end.record()
            end.synchronize()
            self.point_view.step(net_image)
            render_time = start.elapsed_time(end)

        self.monitor.step([render_time])

    def show_gui(self):
        gizmo.begin_frame()

        with imgui_ctx.begin(f"Point View Settings"):
            did_disable = False
            if self.in_selection_mode:
                imgui.begin_disabled()
                did_disable = True

            _, self.render_mode = imgui.list_box("Render Mode", self.render_mode, self.render_modes)
            _, self.ray_choice = imgui.list_box("Displayed Rays", self.ray_choice, self.ray_choices)

            imgui.separator_text("Render Settings")

            if self.render_modes[self.render_mode] == "Ellipsoids":
                _, self.ellipsoid_viewer.scaling_modifier = imgui.drag_float("Scaling Factor", self.ellipsoid_viewer.scaling_modifier, v_min=0, v_max=10, v_speed=0.01)
                
                _, self.ellipsoid_viewer.render_floaters = imgui.checkbox("Render Floaters", self.ellipsoid_viewer.render_floaters)
                _, self.ellipsoid_viewer.limit = imgui.drag_float("Alpha Threshold", self.ellipsoid_viewer.limit, v_min=0, v_max=1, v_speed=0.01)
            else:
                _, self.scaling_modifier = imgui.drag_float("Scaling Factor", self.scaling_modifier, v_min=0, v_max=10, v_speed=0.01)
                
                _, self.exposure = imgui.drag_float("Exposure", self.exposure, v_min=0, v_max=6, v_speed=0.01)

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

            imgui.separator_text("Selection")

            clicked, self.selection_choice = imgui.combo("Object List", self.selection_choice, self.selection_choices)
            if clicked:
                self.update_bbox_selection()    
                self.reset_brdf_edit_settings()
            
            if did_disable:
                imgui.end_disabled()
            clicked = imgui.button("Point and Click", size=(240, 24))
            if clicked:
                self.in_selection_mode = not self.in_selection_mode
                self.selection_choice = 0
            if did_disable:
                imgui.begin_disabled()

            imgui.separator_text("BRDF Editing")

            imgui.set_cursor_pos_x((imgui.get_content_region_avail().x - imgui.calc_text_size("Roughness").x) * 0.35)
            imgui.text("Roughness")
            _, self.edit.roughness_override = imgui.drag_float("##Roughness Override", self.edit.roughness_override, v_min=0, v_max=1, v_speed=0.01/2)
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.edit.roughness_override = 0.0
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
            _, self.edit.diffuse_override = imgui.color_edit4("##Diffuse Override", self.edit.diffuse_override, flags=imgui.ColorEditFlags_.no_options)
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
            _, self.edit.glossy_override = imgui.color_edit4("##Specular Override", self.edit.glossy_override, flags=imgui.ColorEditFlags_.no_options)
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

            clicked = imgui.button("Reset Selection BRDF", size=(240, 24))
            clicked = imgui.button("Reset All BRDFs", size=(240, 24))
            imgui.spacing() 
            imgui.spacing() 

            imgui.separator_text("Geometric Editing")

            imgui.spacing() 

            x = 0
            y = 0
            z = 0
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.21)
            _, x = imgui.drag_float("##Translate X", x, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                x = 0.0
            imgui.same_line()
            _, y = imgui.drag_float("##Translate Y", y, v_min=-1, v_max=1, v_speed=0.01, format="+%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                y = 0.0
            imgui.same_line()
            _, z = imgui.drag_float("##Translate Z", z, v_min=-1, v_max=1, v_speed=0.01, format="+%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                z = 0.0
            imgui.same_line()
            imgui.text("Translation")
            imgui.pop_item_width()

            imgui.spacing() 

            # imgui.checkbox("Show Gizmo", False)    
            imgui.button("Duplicate", size=(240, 24))
            clicked = imgui.button("Reset Selection Pose", size=(240, 24))
            clicked = imgui.button("Reset All Poses", size=(240, 24))
        
            if did_disable:
                imgui.end_disabled()

        with imgui_ctx.begin("Point View"):
            if self.render_modes[self.render_mode] == "Ellipsoids":
               self.ellipsoid_viewer.show_gui()
            else:
                self.point_view.show_gui()
                if self.in_selection_mode:
                    if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                        mouse_pos = imgui.get_mouse_pos()
                        window_pos = imgui.get_window_pos()
                        j, i = int(mouse_pos[0] - window_pos.x), int(mouse_pos[1] - window_pos.y - 30)
                        IMAGE_HEIGHT = 400 
                        IMAGE_WIDTH = 600 
                        i = min(max(0, i), IMAGE_HEIGHT - 1)
                        j = min(max(0, j), IMAGE_WIDTH - 1)
                        if False:
                            print(f"Mouse clicked at pixel position: ({i}, {j})")
                        for bbox_name, mask in self.selection_masks.items():
                            Image.fromarray((mask * 255).astype(np.uint8)).save(f"{bbox_name}_vis.png")
                            print(bbox_name, mask[i, j], np.sum(mask))
                            if mask[i, j]:
                                self.selection_choice = self.selection_choices.index(bbox_name)
                                self.update_bbox_selection()
                                self.reset_brdf_edit_settings()
                                break
                        self.in_selection_mode = False
                    elif imgui.is_mouse_clicked(imgui.MouseButton_.right):
                        self.in_selection_mode = False
                if self.selection_choice != 0 and False:

                    view = Matrix16([0.258821, -0, 0.965925, -0,
                    0.511862, 0.848048, -0.137154, 2.38419e-07,
                    -0.819151, 0.529919, 0.219493, -8,
                    0, 0, 0, 1])
                    
                    projection = Matrix16([4.1653, 0, 0, 0, 
                    0, 4.1653, 0, 0, 
                    0, 0, -1.002, -0.2002, 
                    0, 0, -1, 0])
                    
                    print((imgui.get_window_pos().x, imgui.get_window_pos().y, imgui.get_window_width(), imgui.get_window_height(), self.camera.res_x, self.camera.res_y))

                    gizmo.set_drawlist()
                    gizmo.set_rect(imgui.get_window_pos().x, imgui.get_window_pos().y, 600, 400) # ! tmp
                    def glm_to_mat16(mat: glm.mat4x4):
                        return Matrix16(mat[0].to_list() + mat[1].to_list() + mat[2].to_list() + mat[3].to_list())

                    cameraView =  Matrix16([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                    cameraProjection = Matrix16() 
                    fov = 27.0
                    camYAngle = 165.0 / 180.0 * 3.14159
                    camXAngle = 32.0 / 180.0 * 3.14159
                    camDistance = 8.0
                    eye = glm.vec3(
                        math.cos(camYAngle) * math.cos(camXAngle) * camDistance,
                        math.sin(camXAngle) * camDistance,
                        math.sin(camYAngle) * math.cos(camXAngle) * camDistance,
                    )
                    at = glm.vec3(0.0, 0.0, 0.0)
                    up = glm.vec3(0.0, 1.0, 0.0)
                    cameraView = glm_to_mat16(glm.lookAt(eye, at, up))

                    aspect_ratio = 1.0
                    cameraProjection_glm = glm.perspective(glm.radians(fov), aspect_ratio, 0.1, 100.0)
                    cameraProjection = glm_to_mat16(cameraProjection_glm)

                    view_mat = Matrix16(self.camera.to_camera.flatten().tolist())
                    # proj_mat = Matrix16(self.camera.full_projection.flatten().tolist())
                    gizmo.manipulate(cameraView, cameraProjection, gizmo.OPERATION.translate, gizmo.MODE.local, self.selected_object_transform, None, None, None, None)

            cam_changed_from_mouse = imgui.is_item_hovered() and self.camera.process_mouse_input()
            cam_changed_from_keyboard = (imgui.is_item_focused() or imgui.is_item_hovered()) and self.camera.process_keyboard_input()
            if cam_changed_from_mouse or cam_changed_from_keyboard:
                self.current_train_cam = -1
                self.current_test_cam = -1
        
        with imgui_ctx.begin("Performance"):
            self.monitor.show_gui()

        if self.in_selection_mode:
            mouse_pos = imgui.get_mouse_pos()
            draw_list = imgui.get_foreground_draw_list()
            color = imgui.color_convert_float4_to_u32((1.0, 1.0, 0.0, 0.7))  
            draw_list.add_circle_filled((mouse_pos[0], mouse_pos[1]), 3.0, color)
    
    def client_send(self):
        return None, {
            "scaling_modifier": self.scaling_modifier,
            "render_mode": self.render_mode,
            "exposure": self.exposure,
            "ray_choice": self.ray_choice,
            "selection_choice": self.selection_choice,


            # current edit 
            "roughness_shift": self.edit.roughness_shift,
            "roughness_mult": self.edit.roughness_mult,
            "diffuse_hue_shift": self.edit.diffuse_hue_shift,
            "diffuse_saturation_shift": self.edit.diffuse_saturation_shift,
            "diffuse_saturation_mult": self.edit.diffuse_saturation_mult,
            "diffuse_value_shift": self.edit.diffuse_value_shift,
            "diffuse_value_mult": self.edit.diffuse_value_mult,
            "glossy_hue_shift": self.edit.glossy_hue_shift,
            "glossy_saturation_shift": self.edit.glossy_saturation_shift,
            "glossy_saturation_mult": self.edit.glossy_saturation_mult,
            "glossy_value_shift": self.edit.glossy_value_shift,
            "glossy_value_mult": self.edit.glossy_value_mult,
            "roughness_override": self.edit.roughness_override,
            "diffuse_override": self.edit.diffuse_override,
            "glossy_override": self.edit.glossy_override
        }
    
    def server_recv(self, _, text):
        self.scaling_modifier = text["scaling_modifier"]
        self.render_mode = text["render_mode"]
        self.ray_choice = text["ray_choice"]
        self.selection_choice = text["selection_choice"]
        self.exposure = text["exposure"]
        
        # current edit 
        self.edit.roughness_shift = text["roughness_shift"]
        self.edit.roughness_mult = text["roughness_mult"]
        self.edit.diffuse_hue_shift = text["diffuse_hue_shift"]
        self.edit.diffuse_saturation_shift = text["diffuse_saturation_shift"]
        self.edit.diffuse_saturation_mult = text["diffuse_saturation_mult"]
        self.edit.diffuse_value_shift = text["diffuse_value_shift"]
        self.edit.diffuse_value_mult = text["diffuse_value_mult"]
        self.edit.glossy_hue_shift = text["glossy_hue_shift"]
        self.edit.glossy_saturation_shift = text["glossy_saturation_shift"]
        self.edit.glossy_saturation_mult = text["glossy_saturation_mult"]
        self.edit.glossy_value_shift = text["glossy_value_shift"]
        self.edit.glossy_value_mult = text["glossy_value_mult"]
        self.edit.roughness_override = text["roughness_override"]
        self.edit.diffuse_override = text["diffuse_override"]
        self.edit.glossy_override = text["glossy_override"]

    def server_send(self):
        if self.first_send:
            data = {
                # "cameras": self.cameras,
                "ray_count": self.ray_count,
                "selection_choices": self.selection_choices,
                "train_transforms": self.train_transforms,
                "test_transforms": self.test_transforms,
                "bounding_boxes": self.bounding_boxes,
                "selection_masks": {
                    k: v.tolist() for k, v in self.selection_masks.items()
                }
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
            self.camera.update_pose(np.array(text["train_transforms"]["frames"][0]["transform_matrix"]) @ self.blender_to_opengl)
        if "selection_choices" in text:
            self.selection_choices = text["selection_choices"]
        if "bounding_boxes" in text:
            self.bounding_boxes = text["bounding_boxes"]
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