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
    float sum = wk[pc.gate_off];
    float sumsq = wk[pc.residual_off];
    float mean = sum / float(pc.cols);
    float variance = max(sumsq / float(pc.cols) - mean * mean, 0.0);
    float inv = 1.0 / sqrt(variance + pc.eps);
    float x = wk[pc.in_off + i];
    float y = (x - mean) * inv * w[pc.weight_off + i] + w[pc.bias_off + i];
    wk[pc.out_off + i] = y;
    // Optional mirror write (stores the previous TimeMix input for the next
    // token; lives in the work buffer so `mix_inputs` can read it back).
    if (pc.extra0 != 0xFFFFFFFFu) {
        wk[pc.extra0 + i] = y;
    }
}
