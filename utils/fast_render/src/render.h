#pragma once

#include <string.h>
#include <stdio.h>
#include <math.h> // ceil
#include <algorithm> // min max

extern "C" {
    void render_texture_loop(
            const double *tri_depth, int tri_depth_height,
            const double *tri_tex, int tri_tex_width, int tri_tex_height,
            const int *triangles, int triangles_width, int triangles_height,
            const double *vertices, int vertices_width, int vertices_height,
            double *depth_buffer, int depth_buffer_width, int depth_buffer_height,
            double *image, int image_width, int image_height, int image_channels
        );
}
