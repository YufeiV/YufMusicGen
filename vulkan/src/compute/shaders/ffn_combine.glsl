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
    uint col = gl_GlobalInvocationID.x;
    if (col >= pc.cols) {
        return;
    }
    // in_off -> ffn_in, weight_off -> ffn_gate, bias_off -> W_out
    // gate_off -> residual (hidden), out_off -> hidden output
    float acc = 0.0;
    for (uint j = 0; j < pc.k; j++) {
        float a = wk[pc.in_off + j];
        float silu_a = a / (1.0 + exp(-a));
        float b = wk[pc.weight_off + j];
        acc += silu_a * b * w[pc.bias_off + col * pc.k + j];
    }
    acc += wk[pc.gate_off + col];
    wk[pc.out_off + col] = acc;
}

