import ctypes
import numpy as np
from . import Widget
from OpenGL.GL import *
from imgui_bundle import imgui
from ..types import Texture2D
from viewer.cameras import Camera
from OpenGL.GL.shaders import compileShader, compileProgram

_vert_shader = """
#version 330 core
layout(location = 0) in vec3 position;   // Vertex position
layout(location = 1) in vec3 color; // Vertex color

out vec3 fragColor; // Pass color to fragment shader

uniform mat4 projection; // Projection matrix
uniform mat4 view;       // View matrix

void main() {
    gl_Position = projection * view * vec4(position, 1.0);
    fragColor = color; // Pass color to fragment shader
}
"""

_frag_shader = """
#version 330 core
in vec3 fragColor;

out vec4 result;

void main() {
    // Render points as circles
    if (length(gl_PointCoord - 0.5) <= 0.5) {
        vec3 output = clamp(fragColor, 0, 1);
        output = pow(output, vec3(1 / 2.2));
        result = vec4(output, 1.0);
    } else {
        result = vec4(0.0);
    }
}
"""

class PointRenderer(Widget):
    def __init__(self, positions: np.ndarray, colors: np.ndarray, headless=False):
        self._positions = positions
        self._colors = colors
        self.point_size = 5
        super().__init__(headless)

    def setup(self):
        vertex_data = np.concatenate([self._positions, self._colors], axis=1)
        vertex_data = np.ascontiguousarray(vertex_data)
        self._vertex_data = vertex_data

        # Create and bind VAO / VBO
        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)
        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        # Upload data to buffer
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        # Configure vertex attribute for position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_data[0].nbytes, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Configure vertex attribute for color
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_data[0].nbytes, ctypes.c_void_p(3 * vertex_data.itemsize))
        glEnableVertexAttribArray(1)

        # Unbind VAO / VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # Compile shaders        
        self._shader = compileProgram(
            compileShader(_vert_shader, GL_VERTEX_SHADER),
            compileShader(_frag_shader, GL_FRAGMENT_SHADER)
        )

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)

        self._color_texture = Texture2D()
        self._color_texture.id = glGenTextures(1)
        self._depth_texture = Texture2D() # Technically its a RBO
        self._depth_texture.id = glGenRenderbuffers(1)
        self._fbo = None # FBO will be created in first call to step

    def destroy(self):
        glDeleteTextures(1, int(self._color_texture.id))
        glDeleteRenderbuffers(1, int(self._depth_texture.id))
        if self._fbo is not None:
            glDeleteFramebuffers(1, int(self._fbo))
        glDeleteProgram(self._shader)

    def _create_fbo(self, res_x: int, res_y: int):
        # Create framebuffer
        if self._fbo is not None:
            glDeleteFramebuffers(self._fbo)
        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # Create texture to render to
        self._color_texture.res_x = res_x
        self._color_texture.res_y = res_y
        glBindTexture(GL_TEXTURE_2D, self._color_texture.id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8,
            self._color_texture.res_x, self._color_texture.res_y,
            0, GL_RGBA, GL_UNSIGNED_BYTE, None
        )
        glBindTexture(GL_TEXTURE_2D, 0)
        # Attach texture to framebuffer
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            self._color_texture.id, 0
        )

        # Create depth RBO
        self._depth_texture.res_x = res_x
        self._depth_texture.res_y = res_y
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_texture.id)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,
            self._depth_texture.res_x, self._depth_texture.res_y
        )
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        # Attach it framebuffer
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
            self._depth_texture.id
        )

        # Verify framebuffer is complete
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "FBO not complete"

        # Unbind the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def step(self, camera: Camera, res_x: int, res_y: int) -> Texture2D:
        if res_x != self._color_texture.res_x or res_y != self._color_texture.res_y:
            # Recreate FBO
            self._create_fbo(res_x, res_y)

        glPointSize(self.point_size)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Use the shader program
        glUseProgram(self._shader)

        # Pass the matrices to the shader
        # Load the matrices as transpose because OpenGL expects them in column major
        glUniformMatrix4fv(glGetUniformLocation(self._shader, "projection"),
                           1, GL_TRUE, camera.projection(res_y/res_x))
        glUniformMatrix4fv(glGetUniformLocation(self._shader, "view"),
                           1, GL_TRUE, camera.to_camera)

        # Bind the VAO and draw the points
        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self.num_points)
        glBindVertexArray(0)

        # Unbind program
        glUseProgram(0)
        glPointSize(1)
    
    def show_gui(self, draw_list: imgui.ImDrawList=None):
        res_x = self._color_texture.res_x
        res_y = self._color_texture.res_y
        if draw_list is not None:
            # Figure out
            draw_list.add_image(self._color_texture.id, (0, 0), (res_x, res_y))
        else:
            imgui.image(self._color_texture.id, (res_x, res_y))


    @property
    def num_points(self):
        return len(self._vertex_data)