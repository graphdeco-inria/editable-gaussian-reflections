from . import Widget
from .image import Image
from .cameras import Camera
from ..types import ViewerMode
from imgui_bundle import imgui, imgui_ctx

class Viewport3D(Widget):
    def __init__(self, mode: ViewerMode, title: str, image: Image, camera: Camera):
        super().__init__(mode)
        self.title = title
        self.img = image
        self.camera = camera
        self.camera_types = ["FPS", "Orbit", "Trackball"]
    
    def setup(self):
        self.img.setup()
        self.camera.setup()
    
    def destroy(self):
        self.img.destroy()
        self.camera.destroy()
    
    def step(self, img):
        self.img.step(img)
    
    def show_gui(self):
        # TODO: Use dock builder to build a layout
        with imgui_ctx.begin(f"{self.title} Camera"):
            imgui.combo("Camera Type", 0, self.camera_types)
            imgui.separator_text("Camera Settings")
            self.camera.show_gui()
        self.img.show_gui()
        
        if imgui.is_item_hovered():
            self.camera.process_mouse_input()
        
        if imgui.is_item_focused() or imgui.is_item_hovered():
            self.camera.process_keyboard_input()