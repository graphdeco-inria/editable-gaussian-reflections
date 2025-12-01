import numpy as np
from . import Widget
from OpenGL.GL import *
from ..types import Texture2D
from OpenGL.GL.shaders import compileShader, compileProgram
from imgui_bundle import imgui, imgui_ctx, ImVec2

_vert_shader = """
#version 330 core

layout(location = 0) in vec3 position;   // Vertex position

void main() { gl_Position = vec4(position, 1.0); }
"""

_frag_shader = """
#version 330 core

#define M_PI 3.14159265359f

out vec4 fragColor;
uniform float zoom_fac;
uniform vec2 offset;
uniform vec2 iResolution;
uniform sampler2D sph_function;

// Assumes y-up
vec2 cart2pol(in vec3 cart) {
    cart = normalize(cart);
    float theta = acos(cart.y);
    float phi = atan(cart.z, cart.x);
    phi += phi < 0 ? 2*M_PI : 0;

    return vec2(phi, theta);
}

bool intersect_sphere(in vec3 o, in vec3 d, out float t, out vec3 normal) {
    vec3 oc = o;
    float b = dot(oc, d);
    float c = dot(oc, oc) - 1.f;
    float disc = b * b - c;
    if (disc < 0.0) {
      return false;
    }
    disc = sqrt(disc);
    t = -b - disc;
    t = t > 0.f ? t : -b + disc;
    normal = normalize(o + d * t);
    return t > 0.f;
}

void main() {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 uv = (fragCoord - iResolution.xy / 2.f)/min(iResolution.x, iResolution.y);
    uv.y *= -1;

    // Output to screen
    vec3 d = normalize(vec3(-1, -1, -1));
    
    vec3 up = vec3(0, 1, 0);
    vec3 right = normalize(cross(d, up));
    up = normalize(cross(right, d));
    right *= zoom_fac;
    up *= zoom_fac;
    
    vec3 o = vec3(1);
    o += right*uv.x;
    o += up*uv.y;
    
    float t;
    vec3 n;
    if (intersect_sphere(o, d, t, n)) {
        vec2 pol = cart2pol(n);
        pol += offset;
        pol /= vec2(2*M_PI, M_PI);
        fragColor = texture(sph_function, pol);
    } else {
        fragColor = vec4(0);
    }
}
"""

class SphereViewer(Widget):
    def __init__(self, headless=False):
        self.zoom_fac = 2
        self.last_frame_time = 0
        self.theta = 0
        self.phi = 0
        super().__init__(headless)

    def setup(self):
        vertex_data = np.array([
            -1, -1, 0, -1, +1, 0,
            +1, +1, 0, +1, -1, 0
        ], dtype=np.float32)
        self._vertex_data = vertex_data
        self._indices = np.array([0, 1, 2, 0, 2, 3])

        # Create and bind VAO / VBO
        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)
        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        # Upload data to buffer
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        # Configure vertex attribute for position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Unbind VAO / VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # Compile shaders        
        self._shader = compileProgram(
            compileShader(_vert_shader, GL_VERTEX_SHADER),
            compileShader(_frag_shader, GL_FRAGMENT_SHADER)
        )

        self._color_texture = Texture2D()
        self._color_texture.id = glGenTextures(1)
        self._fbo = None # FBO will be created in first call to step

    def destroy(self):
        glDeleteTextures(1, int(self._color_texture.id))
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

        # Verify framebuffer is complete
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "FBO not complete"

        # Unbind the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def step(self, res_x: int, res_y: int, texture: Texture2D) -> Texture2D:
        if res_x != self._color_texture.res_x or res_y != self._color_texture.res_y:
            # Recreate FBO
            self._create_fbo(res_x, res_y)

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glClearColor(0,0,0,1)
        glClear(GL_COLOR_BUFFER_BIT)

        # Use the shader program
        glUseProgram(self._shader)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture.id)

        # Pass uniforms to the shader
        glUniform1f(glGetUniformLocation(self._shader, "zoom_fac"), self.zoom_fac)
        glUniform2f(glGetUniformLocation(self._shader, "offset"), self.phi, 0)
        glUniform2f(glGetUniformLocation(self._shader, "iResolution"), res_x, res_y)
        glUniform1i(glGetUniformLocation(self._shader, "sph_function"), 0)

        # Bind the VAO and draw the points
        glBindVertexArray(self._vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, self._indices);
        glBindVertexArray(0)

        # Unbind program and FBO
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def show_gui(self, draw_list: imgui.ImDrawList=None):
        res_x = self._color_texture.res_x
        res_y = self._color_texture.res_y
        curr_time = imgui.get_time()
        delta_time = curr_time - self.last_frame_time
        self.last_frame_time = curr_time
        if imgui.is_window_focused() or imgui.is_window_hovered():
            io = imgui.get_io()
            self.zoom_fac -= io.mouse_wheel * 10 * delta_time
            self.zoom_fac = max(0, self.zoom_fac)

            radians_per_pixel = np.pi / 50
            if imgui.is_mouse_dragging(0):
                delta = imgui.get_mouse_drag_delta()
                self.theta -= delta.y * radians_per_pixel * delta_time
                self.theta %= np.pi
                self.phi += delta.x * radians_per_pixel * delta_time
                self.phi %= np.pi * 2
                imgui.reset_mouse_drag_delta()

        if draw_list is not None:
            # Figure out
            tex_ref = imgui.ImTextureRef(int(self._color_texture.id))
            draw_list.add_image(tex_ref, (0, 0), (res_x, res_y))
        else:
            tex_ref = imgui.ImTextureRef(int(self._color_texture.id))
            imgui.image(tex_ref, (res_x, res_y))
