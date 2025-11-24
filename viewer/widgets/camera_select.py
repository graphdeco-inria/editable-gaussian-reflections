import json
from . import Widget
from copy import deepcopy
from typing import List, Tuple
from viewer.cameras import Camera, FPSCamera
from imgui_bundle import imgui, portable_file_dialogs as pfd

class CameraSelect(Widget):
    def setup(self):
        self.cameras: List[Camera] = []
        self.cur_cam = 0
    
    def show_gui(self, camera: Camera) -> Tuple[str,Camera]:
        result = ""
        updated_camera = None
        imgui.text(f"# Cameras: {len(self.cameras)}")
        imgui.same_line()
        if imgui.button("Add Camera"):
            self.cameras.append(deepcopy(camera))
        if self.cameras:
            imgui.same_line()
            if imgui.button("Remove Camera"):
                self.cameras.pop(self.cur_cam)
            changed, self.cur_cam = imgui.slider_int(
                "Active Camera", self.cur_cam,
                0, len(self.cameras)-1, flags=imgui.SliderFlags_.always_clamp
            )
            if changed:
                updated_camera = deepcopy(self.cameras[self.cur_cam])
        if imgui.button("Load Cameras"):
            load_path = pfd.open_file("Load Cameras", filters=["JSON Files (*.json)", "*.json"]).result()
            if load_path:
                with open(load_path[0], "r") as f:
                    cameras = json.load(f)
                    # TODO: Fix FPS camera to generic camera
                    self.cameras = [FPSCamera.from_json(cam) for cam in cameras]
            result = f"Loaded {len(self.cameras)} cameras"
        imgui.same_line()
        if imgui.button("Export Cameras"):
            export_path = pfd.save_file("Export Cameras").result()
            if export_path:
                if not export_path.endswith(".json"):
                    export_path = export_path + ".json"
                with open(export_path, "w") as f:
                    json.dump([cam.to_json() for cam in self.cameras], f)
                result = f"Exported {len(self.cameras)} cameras"

        return result, updated_camera