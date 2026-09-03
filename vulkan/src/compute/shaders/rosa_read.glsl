#version 450
layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint in_off;
    uint out_off;
    uint weight_off;
    uint bias_off;
    uint gate_off;
    uint residual_off;
    uint rows;
    uint cols;
    uint k;
    uint flags;
    uint token_off;
    uint extra0;
    uint extra1;
    uint extra2;
    float eps;
} pc;

layout(std430, set = 0, binding = 0) buffer Weights { float w[]; };
layout(std430, set = 0, binding = 1) buffer Work { float wk[]; };
layout(std430, set = 0, binding = 2) buffer State { float st[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.cols) {
        return;
    }
    // in_off -> memory (state), weight_off -> direction, gate_off -> read_gate (work)
    uint mem_base = pc.in_off;
    uint dir_base = pc.weight_off;
    uint rg_base = pc.gate_off;
    float norm2 = 0.0;
    for (uint j = 0; j < pc.cols; j++) {
        float d = w[dir_base + j];
        norm2 += d * d;
    }
    float inv_norm = 1.0 / max(sqrt(norm2), 1e-12);
    float proj = 0.0;
    for (uint j = 0; j < pc.cols; j++) {
        proj += st[mem_base + j] * w[dir_base + j] * inv_norm;
    }
    float mem = st[mem_base + i];
    float dir = w[dir_base + i] * inv_norm;
    // read = H(m) * sigmoid(read_gate); the Python reference and the fused
    // CUDA kernel both multiply by the post-sigmoid gate (the previous
    // version divided here, which does not match either reference).
    float read = (mem - 2.0 * proj * dir) * (1.0 / (1.0 + exp(-wk[rg_base + i])));
    wk[pc.out_off + i] = read;
}
