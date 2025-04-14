import glfw
import numpy as np
from . import Camera
from ...types import ViewerMode
from imgui_bundle import imgui

# TODO: Coalesce all camera types into a single class
class FPSCamera(Camera):
    def __init__(
            self, mode: ViewerMode,
            res_x: int=1280, res_y: int=720, fov_y: float=30.0,
            z_near: float=0.001, z_far: float=100.0,
            to_world: np.ndarray=None
    ):
        super().__init__(mode, res_x, res_y, fov_y, z_near, z_far, to_world)
        self.speed = 2
        self.mouse_speed = 2
        self.radians_per_pixel = np.pi / 150
        self.invert_mouse = False
        self.current_type = "FPS"
        
        self.dirty = False
        self.last_state = self.to_json()

    def dirty_check(self):
        if self.to_json() != self.last_state:
            self.last_state = self.to_json()
            self.is_dirty = True
        else:
            self.is_dirty = False 

    def setup(self):
        self.movement_keys = {
            "w": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_W))],
            "a": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_A))],
            "s": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_S))],
            "d": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_D))],
            "q": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_Q))],
            "e": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_E))],
            "j": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_J))],
            "k": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_K))],
            "l": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_L))],
            "i": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_I))],
            "o": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_O))],
            "u": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_U))],
        }
    
    def process_mouse_input(self) -> bool:
        if imgui.is_mouse_dragging(0):
            delta = imgui.get_mouse_drag_delta()
            delta.y *= -1 if self.invert_mouse else 1
            delta.x *= -1 if self.invert_mouse else 1
            angle_right = -delta.y * self.radians_per_pixel * self.delta_time * self.mouse_speed
            angle_up = -delta.x * self.radians_per_pixel * self.delta_time * self.mouse_speed
            self.apply_rotation(0, angle_right, angle_up)
            imgui.reset_mouse_drag_delta()
            return True

        return False
    
    def process_keyboard_input(self):
        update = False

        if imgui.is_key_down(self.movement_keys["w"]):
            self.origin += self.speed * self.forward * self.delta_time
            update = True
        if imgui.is_key_down(self.movement_keys["a"]):
            self.origin -= self.speed * self.right * self.delta_time
            update = True
        if imgui.is_key_down(self.movement_keys["q"]):
            self.origin -= self.speed * self.up * self.delta_time
            update = True
        if imgui.is_key_down(self.movement_keys["s"]):
            self.origin -= self.speed * self.forward * self.delta_time
            update = True
        if imgui.is_key_down(self.movement_keys["d"]):
            self.origin += self.speed * self.right * self.delta_time
            update = True
        if imgui.is_key_down(self.movement_keys["e"]):
            self.origin += self.speed * self.up * self.delta_time
            update = True
        
        angle_forward = 0.0
        angle_right = 0.0
        angle_up = 0.0
        if imgui.is_key_down(self.movement_keys["i"]):
            angle_right += 50 * self.radians_per_pixel * self.delta_time
        if imgui.is_key_down(self.movement_keys["k"]):
            angle_right -= 50 * self.radians_per_pixel * self.delta_time
        if imgui.is_key_down(self.movement_keys["j"]):
            angle_up += 50 * self.radians_per_pixel * self.delta_time
        if imgui.is_key_down(self.movement_keys["l"]):
            angle_up -= 50 * self.radians_per_pixel * self.delta_time
        if imgui.is_key_down(self.movement_keys["o"]):
            angle_forward += 50 * self.radians_per_pixel * self.delta_time
        if imgui.is_key_down(self.movement_keys["u"]):
            angle_forward -= 50 * self.radians_per_pixel * self.delta_time

        if angle_forward or angle_right or angle_up:
            self.apply_rotation(angle_forward, angle_right, angle_up)
            update = True

        return update
    
    def show_gui(self):
        super().show_gui()
        _, self.speed = imgui.slider_float("Speed", self.speed, 0.1, 10)
        _, self.invert_mouse = imgui.checkbox("Invert Mouse", self.invert_mouse)
