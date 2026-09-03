#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(frag_color.x, frag_color.y, frag_color.z, 0.92);
}
