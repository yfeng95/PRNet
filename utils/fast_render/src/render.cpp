#include "render.h"


bool PointInTriangle(double p_x, double p_y, double p0_x, double p0_y, double p1_x, double p1_y, double p2_x, double p2_y) {
    float A = 1/2 * (-p1_y * p2_x + p0_y * (-p1_x + p2_x) + p0_x * (p1_y - p2_y) + p1_x * p2_y);
    float sign = A < 0 ? -1 : 1;
    float s = (p0_y * p2_x - p0_x * p2_y + (p2_y - p0_y) * p_x + (p0_x - p2_x) * p_y) * sign;
    float t = (p0_x * p1_y - p0_y * p1_x + (p0_y - p1_y) * p_x + (p1_x - p0_x) * p_y) * sign;

    return s > 0 && t > 0 && (s + t) < 2 * A * sign;
}

void render_texture_loop(const double *tri_depth, int tri_depth_height,
        const double *tri_tex, int tri_tex_width, int tri_tex_height,
        const int *triangles, int triangles_width, int triangles_height,
        const double *vertices, int vertices_width, int vertices_height,
        double *depth_buffer, int depth_buffer_width, int depth_buffer_height,
        double *image, int image_width, int image_height)
{
    for (int i=0; i<triangles_width; i++) {
        // 3 vertex indices
        int tri1 = triangles[0*triangles_width+i];
        int tri2 = triangles[1*triangles_width+i];
        int tri3 = triangles[2*triangles_width+i];

        double v1_u = vertices[0*vertices_width+tri1];
        double v2_u = vertices[0*vertices_width+tri2];
        double v3_u = vertices[0*vertices_width+tri3];

        double v1_v = vertices[1*vertices_width+tri1];
        double v2_v = vertices[1*vertices_width+tri2];
        double v3_v = vertices[1*vertices_width+tri3];


        // the inner bounding box
        int umin = std::max(ceil(std::min(std::min(v1_u,v2_u),v3_u)), 0.0);
        int umax = std::min(floor(std::max(std::max(v1_u,v2_u),v3_u)), double(image_width-1));

        int vmin = std::max(ceil(std::min(std::min(v1_v,v2_v),v3_v)), 0.0);
        int vmax = std::min(floor(std::max(std::max(v1_v,v2_v),v3_v)), double(image_height-1));

        if (umax<umin || vmax<vmin)
            continue;

        for (int u = umin; u<umax+1; u++) {
            for (int v = vmin; v<vmax+1; v++) {
                if (tri_depth[i] > depth_buffer[v*depth_buffer_width + u]) {
                    if (PointInTriangle(u, v, v1_u, v2_u, v3_u, v1_v, v2_v, v3_v)) {
                        depth_buffer[v*depth_buffer_width + u] = tri_depth[i];
                        for (int aux=0; aux<tri_tex_height; aux++) {
                            image[v*image_width + u] = tri_tex[aux*tri_tex_width + i];
                        }
                    }
                }
            }
        }
    }
}

