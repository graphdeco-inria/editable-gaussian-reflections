from OpenGL import *
from enum import Flag

class ViewerMode(Flag):
    LOCAL = 0
    SERVER = 1
    CLIENT = 2

# Aliases for easy access
LOCAL = ViewerMode.LOCAL
CLIENT = ViewerMode.CLIENT
SERVER = ViewerMode.SERVER
LOCAL_SERVER = ViewerMode.LOCAL | ViewerMode.SERVER
LOCAL_CLIENT = ViewerMode.LOCAL | ViewerMode.CLIENT

class Texture2D:
    """
    This is just a struct like class which holds the state of the texture which
    for now is only the resolution and OpenGL ID but more fields my be added as
    needed. It doesn't provide any methods to actually perform operations on the 
    exture, use the OpenGL API for that. It is the responsibility of the
    application to update the state of the texture when and if needed.
    """
    res_x = -1
    res_y = -1
    id = -1