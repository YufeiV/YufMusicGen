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
    float x = wk[pc.in_off + i];
    float prev = wk[pc.gate_off + i];
    float delta = x - prev;
    // mix params are contiguous 6 x cols starting at weight_off
    float mr = w[pc.weight_off + 0 * pc.cols + i];
    float mw = w[pc.weight_off + 1 * pc.cols + i];
    float mk = w[pc.weight_off + 2 * pc.cols + i];
    float mv = w[pc.weight_off + 3 * pc.cols + i];
    float ma = w[pc.weight_off + 4 * pc.cols + i];
    float mg = w[pc.weight_off + 5 * pc.cols + i];
    wk[pc.out_off + 0 * pc.cols + i] = x + delta * mr;
    wk[pc.out_off + 1 * pc.cols + i] = x + delta * mw;
    wk[pc.out_off + 2 * pc.cols + i] = x + delta * mk;
    wk[pc.out_off + 3 * pc.cols + i] = x + delta * mv;
    wk[pc.out_off + 4 * pc.cols + i] = x + delta * ma;
    wk[pc.out_off + 5 * pc.cols + i] = x + delta * mg;
}

