import string
import random
from enum import Enum
from imgui_bundle import imgui
from viewer.widgets import Widget

class RadioPicker(Widget):
    def __init__(self, default: Enum):
        self.value = default
        self.states = dict.fromkeys(type(default), False)
        self.states[default] = True
        # Generate a random suffix to avoid collisions.
        # Technically a collision is still possible but highly unlikely.
        self.rand = "##" + "".join(random.choices(string.ascii_letters + string.digits, k=8))

    def show_gui(self) -> bool:
        for option, _ in self.states.items():
            if imgui.radio_button(option.name.capitalize() + self.rand, self.states[option]):
                if option != self.value:
                    self.states[option] = True
                    self.states[self.value] = False
                    self.value = option

                    return True

        return False