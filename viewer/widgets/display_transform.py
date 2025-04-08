import torch
import numpy as np
from . import Widget
from imgui_bundle import imgui

class DisplayTransform(Widget):
    offset = 0
    exposure = 0
    gamma = 2.2

    def step(self, img: torch.Tensor|np.ndarray):
        img = img * (2**self.exposure)
        img += self.offset
        img = img.clamp(0, 1)
        img = img ** (1/self.gamma)

        return img

    def show_gui(self):
        _, self.exposure = imgui.drag_float(
            "Exposure", self.exposure,
            v_speed=0.1, v_min=-5, v_max=5, format="%.2f"
        )
        _, self.offset = imgui.drag_float(
            "Offset", self.offset,
            v_speed=0.1, v_min=-1, v_max=1, format="%.2f"
        )
        _, self.gamma = imgui.drag_float(
            "Gamma", self.gamma,
            v_speed=0.1, v_min=0.01, v_max=5, format="%.2f"
        )