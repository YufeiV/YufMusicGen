#version 450
layout(local_size_x = 8, local_size_y = 8) in;

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
    uint col = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;
    if (col >= pc.cols || row >= pc.rows) {
        return;
    }
    uint in_base = pc.in_off + row * pc.k;
    float acc = 0.0;
    for (uint j = 0; j < pc.k; j++) {
        float x = wk[in_base + j];
        if ((pc.flags & (1u << 5u)) != 0u) {
            x *= wk[pc.gate_off + j];
        }
        if ((pc.flags & (1u << 6u)) != 0u) {
            // Transposed weight: y[col] = sum_j x[j] * W[j][col] (raw matmul
            // semantics, e.g. `x @ W1` in the low-rank TimeMix paths).
            acc += x * w[pc.weight_off + j * pc.cols + col];
        } else {
            acc += x * w[pc.weight_off + col * pc.k + j];
        }
    }
    if ((pc.flags & (1u << 0u)) != 0u) {
        acc += w[pc.bias_off + col];
    }
    if ((pc.flags & (1u << 1u)) != 0u) {
        acc = acc / (1.0 + exp(-acc));
    } else if ((pc.flags & (1u << 2u)) != 0u) {
        acc = tanh(acc);
    } else if ((pc.flags & (1u << 3u)) != 0u) {
        acc = 1.0 / (1.0 + exp(-acc));
    }
    uint out_idx = pc.out_off + row * pc.cols + col;
    if ((pc.flags & (1u << 4u)) != 0u) {
        acc += wk[pc.residual_off + col];
    }
    wk[out_idx] = acc;
}
