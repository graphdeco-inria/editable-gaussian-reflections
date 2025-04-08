from . import Widget
from ..types import Texture2D
from imgui_bundle import imgui, imgui_ctx, ImVec2

class PixelInspector(Widget):
    def __init__(self, window_size=300, headless=False):
        self.active = False
        self.inspect_size = 30
        self.window_size = ImVec2(window_size, window_size)
        self.old_mouse_pos = None
        self.active_mouse_pos = None
        super().__init__(headless)

    def show_gui(self, texture: Texture2D) -> tuple[ImVec2,bool]:
        draw_list = imgui.get_window_draw_list()

        # Draw cross at mouse
        mouse_pos = imgui.get_mouse_pos()
        col = imgui.get_color_u32(imgui.Col_.plot_lines_hovered)
        thickness = 2
        vert_start = ImVec2(mouse_pos.x, mouse_pos.y-10)
        vert_end = ImVec2(mouse_pos.x, mouse_pos.y+10)
        draw_list.add_line(vert_start, vert_end, col, thickness)
        hor_start = ImVec2(mouse_pos.x - 10, mouse_pos.y)
        hor_end = ImVec2(mouse_pos.x + 10, mouse_pos.y)
        draw_list.add_line(hor_start, hor_end, col, thickness)
        
        # Draw rectangle denoting area of zoom
        inspect_size = ImVec2(self.inspect_size // 2, self.inspect_size // 2)
        draw_list.add_rect(mouse_pos - inspect_size, mouse_pos + inspect_size, col, thickness=thickness)

        with imgui_ctx.begin_tooltip():
            texture_size = ImVec2(texture.res_x, texture.res_y)
            uv0 = (mouse_pos - inspect_size) / texture_size
            uv1 = (mouse_pos + inspect_size) / texture_size
            imgui.image(texture.id, self.window_size, uv0, uv1)
            imgui.text("Click: Accept; Scroll +/-: Zoom; Esc: Exit")

        # Change inspect size
        self.inspect_size -= imgui.get_io().mouse_wheel * 5
        self.inspect_size = max(self.inspect_size, 0)

        # Update nouse position
        self.active_mouse_pos = mouse_pos

        # If update was rejected, revert back to old mouse position
        accept = None
        if imgui.is_key_pressed(imgui.Key.escape):
            self.active = False
            self.active_mouse_pos = self.old_mouse_pos
            accept = False

        # If update was accepted, change old mouse position to current
        if imgui.is_mouse_clicked(imgui.MouseButton_.left):
            self.active = False
            accept = True
            self.old_mouse_pos = mouse_pos

        return accept