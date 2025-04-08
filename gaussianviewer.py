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

        viewer.camera_poses = json.load(open(os.path.join(model_path, "cameras.json"), "r"))
        viewer.init_pose = np.hstack([ np.array(viewer.camera_poses[0]["rotation"]), np.array(viewer.camera_poses[0]["position"])[:, None] ])
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
        self.camera = FPSCamera(self.mode, 1297, 840, 47, 0.001, 100)
        self.point_view = TorchImage(self.mode)
        self.ellipsoid_viewer = EllipsoidViewer(self.mode)
        self.monitor = PerformanceMonitor(self.mode, ["Render"], add_other=False)

        # Render modes
        self.render_modes = ["RGB", "Normals", "Position", "F0", "Roughness", "Illumination", "Ellipsoids"]
        self.render_mode = 0
        
        self.ray_choices = ["All/Default"] + ["Ray " + str(i) for i in range(self.ray_count)] 
        self.ray_choice = 0
        
        # Render settings
        self.exposure = 1.0
        self.scaling_modifier = 1.0

    def step(self):
        camera = self.camera
        world_to_view = torch.from_numpy(camera.to_camera).cuda().transpose(0, 1)
        full_proj_transform = torch.from_numpy(camera.full_projection).cuda().transpose(0, 1)
        camera = MiniCam(camera.res_x, camera.res_y, camera.fov_y, camera.fov_x, camera.z_near, camera.z_far, world_to_view, full_proj_transform)

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
                    package = render(camera, self.raytracer, self.pipe, self.background, blur_sigma=None)
                    
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

                net_image = tonemap(untonemap(net_image.permute(1, 2, 0))*self.exposure) # todo only expose rgb
            end.record()
            end.synchronize()
            self.point_view.step(net_image)
            render_time = start.elapsed_time(end)

        self.monitor.step([render_time])

    
    def show_gui(self):
        with imgui_ctx.begin(f"Point View Settings"):
            _, self.render_mode = imgui.list_box("Render Mode", self.render_mode, self.render_modes)
            _, self.ray_choice = imgui.list_box("Rays", self.ray_choice, self.ray_choices)

            imgui.separator_text("Render Settings")
            if self.render_modes[self.render_mode] == "Ellipsoids":
                _, self.ellipsoid_viewer.scaling_modifier = imgui.drag_float("Scaling Factor", self.ellipsoid_viewer.scaling_modifier, v_min=0, v_max=10, v_speed=0.01)
                
                _, self.ellipsoid_viewer.render_floaters = imgui.checkbox("Render Floaters", self.ellipsoid_viewer.render_floaters)
                _, self.ellipsoid_viewer.limit = imgui.drag_float("Alpha Threshold", self.ellipsoid_viewer.limit, v_min=0, v_max=1, v_speed=0.01)
            else:
                _, self.scaling_modifier = imgui.drag_float("Scaling Factor", self.scaling_modifier, v_min=0, v_max=10, v_speed=0.01)
                _, self.exposure = imgui.drag_float("Exposure", self.exposure, v_min=0, v_max=3, v_speed=0.01)

            imgui.separator_text("Camera Settings")
            self.camera.show_gui()

        with imgui_ctx.begin("Point View"):
            if self.render_modes[self.render_mode] == "Ellipsoids":
               self.ellipsoid_viewer.show_gui()
            else:
               self.point_view.show_gui()

            if imgui.is_item_hovered():
                self.camera.process_mouse_input()
            
            if imgui.is_item_focused() or imgui.is_item_hovered():
                self.camera.process_keyboard_input()
        
        with imgui_ctx.begin("Performance"):
            self.monitor.show_gui()
    
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
        return None, {
            # "cameras": self.cameras,
            "ray_count": self.ray_count,
            "init_cam": self.camera_poses[0],
            
        }
    
    def client_recv(self, _, text):
        if self.ray_count != text["ray_count"]:
            self.ray_count = text["ray_count"]
            self.ray_choices = ["All/Default"] + ["Ray " + str(i) for i in range(self.ray_count)]
        if self.init_pose is None:
            self.init_pose = text["init_cam"]
            self.camera.update_pose(np.hstack([ 
            
            # np.array(self.init_pose["position"])[:, None] 
                np.array(self.init_pose["rotation"]), 
                np.array([ 0.00221941, -0.49231448,  0.29137429])[:, None] # todo clean this up
            
            ]))

            # [ 0.00221941 -0.49231448  0.29137429]
    
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

    viewer.run(args.ip, args.port)

    #  xvfb-run -s "-screen 0 1400x900x24" python gaussianviewer.py server output/tmp2 7000 --ip 0.0.0.0