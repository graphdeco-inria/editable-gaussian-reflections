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

gizmo = imguizmo.im_guizmo

Matrix3 = gizmo.Matrix3
Matrix6 = gizmo.Matrix6
Matrix16 = gizmo.Matrix16

from viewer.widgets.ellipsoid_viewer import EllipsoidViewer

class Dummy(object):
    pass


class Dummy(object):
    pass

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

        viewer.train_transforms = json.load(open(os.path.join(model_path, "transforms_train.json"), "r"))
        viewer.test_transforms = json.load(open(os.path.join(model_path, "transforms_test.json"), "r"))
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

        self.brdf_selection_choice = 0
        self.brdf_selection_options = ["None", "Cup", "Table", "Shelf", "All"]
        
        # Render settings
        self.exposure = 1.0
        self.scaling_modifier = 1.0
        
        # Editing
        self.reflectivity_shift = 0.0
        self.reflectivity_mult = 1.0
        self.roughness_shift = 0.0
        self.roughness_mult = 1.0

        self.diffuse_hue_shift = 0.0
        self.diffuse_saturation_shift = 0.0
        self.diffuse_saturation_mult = 1.0
        self.diffuse_value_shift = 0.0
        self.diffuse_value_mult = 1.0
        
        self.glossy_hue_shift = 0.0
        self.glossy_saturation_shift = 0.0
        self.glossy_saturation_mult = 1.0
        self.glossy_value_shift = 0.0
        self.glossy_value_mult = 1.0

        self.in_selection_mode = False


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
                    package = render(camera, self.raytracer, self.pipe, self.background, blur_sigma=None, targets_available=False, edits=dict(
                        reflectivity_shift=self.reflectivity_shift,
                        reflectivity_mult=self.reflectivity_mult,
                        roughness_shift=self.roughness_shift,
                        roughness_mult=self.roughness_mult,
                        diffuse_hue_shift=self.diffuse_hue_shift,
                        diffuse_saturation_shift=self.diffuse_saturation_shift,
                        diffuse_saturation_mult=self.diffuse_saturation_mult,
                        diffuse_value_shift=self.diffuse_value_shift,
                        diffuse_value_mult=self.diffuse_value_mult,
                        glossy_hue_shift=self.glossy_hue_shift,
                        glossy_saturation_shift=self.glossy_saturation_shift,
                        glossy_saturation_mult=self.glossy_saturation_mult,
                        glossy_value_shift=self.glossy_value_shift,
                        glossy_value_mult=self.glossy_value_mult,
                        # selection= put the selection mask here
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
            
            clicked, selected_index = imgui.combo("Object List", self.brdf_selection_choice, self.brdf_selection_options)
            
            if did_disable:
                imgui.end_disabled()
            clicked = imgui.button("                 Selection Tool                 " if not self.in_selection_mode else "                       Cancel                       ")
            if clicked:
                self.in_selection_mode = not self.in_selection_mode
            if did_disable:
                imgui.begin_disabled()

            imgui.separator_text("BRDF Editing")

            imgui.set_cursor_pos_x((imgui.get_content_region_avail().x - imgui.calc_text_size("Surface").x) * 0.35)
            imgui.text("Surface")
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.33333)
            _, self.roughness_shift = imgui.drag_float("##Roughness Shift", self.roughness_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.roughness_shift = 0.0
            imgui.same_line()
            _, self.roughness_mult = imgui.drag_float("##Roughness Mult", self.roughness_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.roughness_mult = 1.0
            imgui.same_line()
            imgui.text("Roughness")
            _, self.reflectivity_shift = imgui.drag_float("##Reflectivity Shift", self.reflectivity_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.reflectivity_shift = 0.0
            imgui.same_line()
            _, self.reflectivity_mult = imgui.drag_float("##Reflectivity Mult", self.reflectivity_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.reflectivity_mult = 1.0
            imgui.same_line()
            imgui.text("Reflectivity")
            imgui.pop_item_width()

            imgui.spacing() 
            imgui.spacing() 
        
            imgui.set_cursor_pos_x((imgui.get_content_region_avail().x - imgui.calc_text_size("Diffuse").x) * 0.35)
            imgui.text("Diffuse")
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.69)
            _, self.diffuse_hue_shift = imgui.drag_float("##Diffuse Hue Shift", self.diffuse_hue_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.diffuse_hue_shift = 0.0
            imgui.same_line()
            imgui.text("Hue")
            imgui.pop_item_width()
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.33333)
            _, self.diffuse_saturation_shift = imgui.drag_float("##Diffuse Saturation Shift", self.diffuse_saturation_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.diffuse_saturation_shift = 0.0
            imgui.same_line()
            _, self.diffuse_saturation_mult = imgui.drag_float("##Diffuse Saturation Mult", self.diffuse_saturation_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.diffuse_saturation_mult = 1.0
            imgui.same_line()
            imgui.text("Saturation")
            _, self.diffuse_value_shift = imgui.drag_float("##Diffuse Value Shift", self.diffuse_value_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.diffuse_value_shift = 0.0
            imgui.same_line()
            _, self.diffuse_value_mult = imgui.drag_float("##Diffuse Value Mult", self.diffuse_value_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.diffuse_value_mult = 1.0
            imgui.same_line()
            imgui.text("Value")

            imgui.spacing() 
            imgui.spacing() 

            imgui.set_cursor_pos_x((imgui.get_content_region_avail().x - imgui.calc_text_size("Specular").x) * 0.35)
            imgui.text("Specular")
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.69)
            _, self.glossy_hue_shift = imgui.drag_float("##Specular Hue Shift", self.glossy_hue_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.glossy_hue_shift = 0.0
            imgui.same_line()
            imgui.text("Hue")
            imgui.pop_item_width()
            imgui.push_item_width(imgui.get_content_region_avail().x * 0.33333)
            _, self.glossy_saturation_shift = imgui.drag_float("##Specular Saturation Shift", self.glossy_saturation_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.glossy_saturation_shift = 0.0
            imgui.same_line()
            _, self.glossy_saturation_mult = imgui.drag_float("##Specular Saturation Mult", self.glossy_saturation_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.glossy_saturation_mult = 1.0
            imgui.same_line()
            imgui.text("Saturation")
            _, self.glossy_value_shift = imgui.drag_float("##Specular Value Shift", self.glossy_value_shift, v_min=-1, v_max=1, v_speed=0.01, format="%+.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.glossy_value_shift = 0.0
            imgui.same_line()
            _, self.glossy_value_mult = imgui.drag_float("##Specular Value Mult", self.glossy_value_mult, v_min=0, v_max=3.0, v_speed=0.01, format="x%.2f")
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                self.glossy_value_mult = 1.0
            imgui.same_line()
            imgui.text("Value")

            imgui.spacing() 
            imgui.spacing() 

            imgui.separator_text("Geometric Editing")

            imgui.checkbox("Show Gizmo", False)    
            imgui.button("Duplicate")
        
            if did_disable:
                imgui.end_disabled()

        with imgui_ctx.begin("Point View"):
            gizmo.set_drawlist()
            pos = imgui.get_cursor_screen_pos()
            gizmo.set_drawlist()
            
            gizmo.set_rect(pos.x, pos.y, self.camera.res_x, self.camera.res_y)

            # if imgui.is_item_hovered() and not gizmo.is_using():
            #     cam_changed_from_mouse = self.camera.process_mouse_input()
            # else:
            #     cam_changed_from_mouse = False
            # if imgui.is_item_focused() or imgui.is_item_hovered():
            #     cam_changed_from_keyboard = self.camera.process_keyboard_input()
            # else:
            #     cam_changed_from_keyboard = False
            
            if self.render_modes[self.render_mode] == "Ellipsoids":
               self.ellipsoid_viewer.show_gui()
            else:
               self.point_view.show_gui()

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
            "ray_choice": self.ray_choice
        }
    
    def server_recv(self, _, text):
        self.scaling_modifier = text["scaling_modifier"]
        self.render_mode = text["render_mode"]
        self.ray_choice = text["ray_choice"]
        self.exposure = text["exposure"]

    def server_send(self):
        if self.first_send:
            data = {
                # "cameras": self.cameras,
                "ray_count": self.ray_count,
                "train_transforms": self.train_transforms,
                "test_transforms": self.test_transforms
            }
        else:
            data = {}
        return None, data
    
    def client_recv(self, _, text):
        if "ray_count" in text and self.ray_count != text["ray_count"]:
            self.ray_count = text["ray_count"]
            self.ray_choices = ["All/Default"] + ["Ray " + str(i) for i in range(self.ray_count)]
        if "train_transforms" in text and self.init_pose is None:
            self.train_transforms = text["train_transforms"]
            self.test_transforms = text["test_transforms"]
            self.camera.update_pose(np.array(text["train_transforms"]["frames"][0]["transform_matrix"]) @ self.blender_to_opengl)
    
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