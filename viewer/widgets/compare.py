import torch
import numpy as np
from imgui_bundle import imgui, ImVec2
from . import Widget
from .image import TorchImage, NumpyImage

class Compare(Widget):
    def __init__(self, img: TorchImage|NumpyImage, res_x: int, res_y: int, headless=False):
        self.img = img
        self.res_x = res_x
        self.res_y = res_y
        self.pos_x = 0.5
        self.dragging = False
        if isinstance(img, TorchImage):
            self.lib = torch
        else:
            self.lib = np
        self.img1 = None
        self.img2 = None
        super().__init__(headless)

    def setup(self):
        self.img.setup()
    
    def destroy(self):
        self.img.destroy()

    def step(self, img1: torch.Tensor|np.ndarray=None, img2: torch.Tensor|np.ndarray=None):
        if self.lib == torch:
            if img1 is not None:
                self.img1 = img1.detach().clone()
            if img2 is not None:
                self.img2 = img2.detach().clone()
        else:
            if img1 is not None:
                self.img1 = img1.copy()
            if img2 is not None:
                self.img2 = img2.copy()

        _, w, __ = self.img1.shape
        result = self.lib.zeros_like(self.img1)
        divider_pos = int(w * self.pos_x)
        result[:,:divider_pos] = self.img1[:,:divider_pos]
        result[:,divider_pos:] = self.img2[:,divider_pos:]
        self.img.step(result)

    def show_gui(self):
        res_x = self.res_x
        res_y = self.res_y

        draw_list = imgui.get_window_draw_list()
        window_pos = imgui.get_cursor_screen_pos()
        pos = res_x * self.pos_x
        begin = ImVec2(pos, 0) + window_pos
        end = ImVec2(pos, res_y) + window_pos

        self.img.show_gui(res_x=self.res_x, res_y=self.res_y)
        safety = 15
        thickness = 2
        imgui.set_cursor_screen_pos(ImVec2(begin.x-safety, begin.y))
        imgui.invisible_button("Slider", (thickness+safety, res_y))
        col = imgui.get_color_u32(imgui.Col_.plot_lines)
        if imgui.is_item_hovered() or self.dragging:
            col = imgui.get_color_u32(imgui.Col_.plot_lines_hovered)
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
            if imgui.is_mouse_down(imgui.MouseButton_.left):
                new_pos = imgui.get_mouse_pos().x-window_pos.x
                self.pos_x = new_pos / res_x
                self.pos_x = max(min(self.pos_x, 1), 0)
                self.dragging = True
            else:
                self.dragging = False

        draw_list.add_line(begin, end, col, thickness)
