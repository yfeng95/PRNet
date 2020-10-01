import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

lib = ctypes.CDLL('./utils/fast_render/build/libfast_render.so')

c_render_texture_loop = lib.render_texture_loop
c_render_texture_loop.argtypes = [
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # tri_depth
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # tri_tex
    ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_int, flags='C_CONTIGUOUS'), # triangles
    ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # vertices
    ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # depth_buffer
    ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), # image
    ctypes.c_int,
    ctypes.c_int,
]


def render_texture_loop(tri_depth, tri_tex, triangles, vertices, depth_buffer, image):
    if len(image.shape) == 2:
        image_channels = 1
    else:
        image_channels = image.shape[2]

    c_render_texture_loop(
            tri_depth, tri_depth.shape[0],
            tri_tex, tri_tex.shape[1], tri_tex.shape[0],
            triangles, triangles.shape[1], triangles.shape[0],
            vertices, vertices.shape[1], vertices.shape[0],
            depth_buffer, depth_buffer.shape[1], depth_buffer.shape[0],
            image, image.shape[1], image.shape[0], image_channels
        )
