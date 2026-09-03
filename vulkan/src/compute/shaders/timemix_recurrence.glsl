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
    uint gid = gl_GlobalInvocationID.x;
    uint total = pc.cols; // n_heads * head_size
    if (gid >= total) {
        return;
    }
    uint hs = pc.k; // head_size
    uint h = gid / hs;
    uint i = gid % hs;

    // Offsets inside `wk` for this layer's TimeMix block.
    // in_off        -> r
    // weight_off    -> k
    // bias_off      -> v
    // gate_off      -> w
    // residual_off  -> a
    // extra0        -> k_k
    // extra1        -> k_a
    // extra2        -> memory base
    uint r_base = pc.in_off;
    uint k_base = pc.weight_off;
    uint v_base = pc.bias_off;
    uint w_base = pc.gate_off;
    uint a_base = pc.residual_off;
    uint kk_base = pc.extra0;
    uint ka_base = pc.extra1;
    uint mem_base = pc.extra2;

    // Compute the normalized key head (per-head norm).
    float norm2 = 0.0;
    for (uint j = 0; j < hs; j++) {
        float kk = wk[k_base + h * hs + j] * w[kk_base + h * hs + j];
        norm2 += kk * kk;
    }
    float denom = max(sqrt(norm2), 1e-12);

    // state_a = sum_j neg_kk[j] * memory[i][j], where neg_kk is the negated
    // normalized key head (this is the `a` argument of the Python recurrence).
    float state_a = 0.0;
    for (uint j = 0; j < hs; j++) {
        uint idx = h * hs + j;
        float kk = wk[k_base + idx] * w[kk_base + idx];
        float khead = kk / denom;
        state_a += -khead * st[mem_base + (h * hs + i) * hs + j];
    }

    float acc = 0.0;
    uint mem_row = mem_base + (h * hs + i) * hs;
    for (uint j = 0; j < hs; j++) {
        uint idx = h * hs + j;
        float kk = wk[k_base + idx] * w[kk_base + idx];
        float khead = kk / denom;
        float a_val = wk[a_base + idx];
        float kadj = wk[k_base + idx] * (1.0 + (a_val - 1.0) * w[ka_base + idx]);
        float neg_kk = -khead;
        float kka = khead * a_val;
        float retention = exp(-0.6065306597 / (1.0 + exp(-wk[w_base + idx])));
        float old = st[mem_row + j];
        float next = old * retention + state_a * kka + wk[v_base + gid] * kadj;
        st[mem_row + j] = next;
        acc += next * wk[r_base + idx];
    }
    wk[pc.out_off + gid] = acc;
}
