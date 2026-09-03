#version 450
layout(local_size_x = 1) in;

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
    float sum = 0.0;
    float sumsq = 0.0;
    for (uint i = 0; i < pc.cols; i++) {
        float x = wk[pc.in_off + i];
        sum += x;
        sumsq += x * x;
    }
    wk[pc.out_off] = sum;
    wk[pc.out_off + 1u] = sumsq;
}
