// Fused Householder-orthogonal ROSA scan.
//
// The ROSA memory branch is a gated recurrent cell whose state transition is
// exactly orthogonal: each step reflects the state through a (learned,
// normalized) Householder direction before applying a per-element decay and a
// gated write.  A naive implementation iterates one token at a time from the
// host, which serializes the whole model on the Python loop.  Mirroring the
// RWKV-7 fused kernel in this repository, this file implements the whole
// recurrence (forward and backward) as a small CUDA kernel:
//
//     m_t  = decay * H(m_{t-1}) + write_t * cand_t
//     read_t = H(m_t)
//
// where H(x) = x - 2 (x . d) d is the Householder reflection around the unit
// direction d.  The forward pass checkpoints every intermediate state, so the
// backward pass reads them directly instead of inverting the recurrence (which
// is numerically unstable for long sequences).

#include <cuda_runtime.h>
#include <stdexcept>

using i64 = long long int;

// One thread block per batch element; `ROSA_DIM_` threads each own one element
// of the state.  The reduction over the state dimension (the Householder
// projection) is a plain O(D) loop over shared memory, which keeps this kernel
// simple and numerically identical to the reference recurrence for the small
// ROSA sizes this model uses.

template <int D>
__global__ void rosa_forward_kernel(
    int T,
    const float* __restrict__ cand_in,
    const float* __restrict__ write_in,
    const float* __restrict__ read_raw_in,
    const float* __restrict__ decay_in,
    const float* __restrict__ direction_in,
    const float* __restrict__ memory_in,
    float* __restrict__ read_out,
    float* __restrict__ memory_out,
    float* __restrict__ cand_saved,
    float* __restrict__ write_saved,
    float* __restrict__ read_saved,
    float* __restrict__ gate_saved,
    float* __restrict__ decay_saved,
    float* __restrict__ mem_saved) {
    const int batch = blockIdx.x;
    const int lane = threadIdx.x;

    __shared__ float direction[D];
    __shared__ float memory[D];
    __shared__ float cand[D];
    __shared__ float write[D];
    __shared__ float decay[D];

    direction[lane] = direction_in[lane];
    memory[lane] = memory_in[batch * D + lane];
    // decay is a single per-dimension vector shared by every batch element.
    decay[lane] = decay_in[lane];
    __syncthreads();
    mem_saved[batch * (T + 1) * D + lane] = memory[lane];

    for (int t = 0; t < T; ++t) {
        const int base = (batch * T + t) * D;

        cand[lane] = tanhf(cand_in[base + lane]);
        write[lane] = 1.0f / (1.0f + __expf(-write_in[base + lane]));
        const float gate_value = 1.0f / (1.0f + __expf(-read_raw_in[base + lane]));
        __syncthreads();

        float projection = 0.0f;
#pragma unroll
        for (int j = 0; j < D; ++j) {
            projection += memory[j] * direction[j];
        }

        float next_memory = decay[lane] * (memory[lane] - 2.0f * projection * direction[lane])
                            + write[lane] * cand[lane];
        __syncthreads();

        memory[lane] = next_memory;
        __syncthreads();

        projection = 0.0f;
#pragma unroll
        for (int j = 0; j < D; ++j) {
            projection += memory[j] * direction[j];
        }
        const float read_value = memory[lane] - 2.0f * projection * direction[lane];

        read_out[base + lane] = read_value;
        memory_out[batch * D + lane] = memory[lane];
        cand_saved[base + lane] = cand[lane];
        write_saved[base + lane] = write[lane];
        read_saved[base + lane] = read_value;
        gate_saved[base + lane] = gate_value;
        decay_saved[base + lane] = decay[lane];
        // `memory` now holds m_{t+1}; store it at index t+1.
        mem_saved[batch * (T + 1) * D + (t + 1) * D + lane] = memory[lane];
    }
}

template <int D>
__global__ void rosa_backward_kernel(
    int T,
    const float* __restrict__ cand_saved,
    const float* __restrict__ write_saved,
    const float* __restrict__ read_saved,
    const float* __restrict__ gate_saved,
    const float* __restrict__ decay_saved,
    const float* __restrict__ mem_saved,
    const float* __restrict__ direction_in,
    const float* __restrict__ final_memory,
    const float* __restrict__ d_read,
    float* __restrict__ d_cand,
    float* __restrict__ d_write,
    float* __restrict__ d_read_raw,
    float* __restrict__ d_decay,
    float* __restrict__ d_direction_partial,
    float* __restrict__ d_memory_out,
    float* __restrict__ workspace) {
    const int batch = blockIdx.x;
    const int lane = threadIdx.x;

    __shared__ float cand[D];
    __shared__ float write[D];
    __shared__ float read_val[D];
    __shared__ float gate_val[D];
    __shared__ float decay[D];
    __shared__ float direction[D];
    __shared__ float memory[D];
    __shared__ float dstate[D];
    __shared__ float direction_grad[D];

    direction[lane] = direction_in[lane];
    memory[lane] = final_memory[batch * D + lane];
    dstate[lane] = 0.0f;
    direction_grad[lane] = 0.0f;
    // Per-step arrays live in a per-batch global workspace so that long
    // sequences do not exhaust the shared-memory limit.  Each array has
    // B * T * D floats and the block for `batch` uses slice `batch * T * D`.
    float* ws = workspace + i64(batch) * 5 * T * D;
    float* g_read_all = ws;
    float* gate_val_all = g_read_all + T * D;
    float* p_all = gate_val_all + T * D;
    float* proj_next_all = p_all + T * D;
    float* dq_all = proj_next_all + T * D;
    __syncthreads();

    // First pass: read the checkpointed states and compute the per-step read
    // gradients.
    for (int t = 0; t < T; ++t) {
        const int base = (batch * T + t) * D;
        const int step_base = t * D;
        const float dq = d_read[base + lane];

        // read_t = H(m_{t+1}); the read gradient flowing into the memory
        // state is H^T(dq) = dq - 2(dq.d) d, while the direction gradient
        // uses proj(m_{t+1}).  The checkpointed state m_{t+1} is stored at
        // index t+1.
        memory[lane] = mem_saved[batch * (T + 1) * D + (t + 1) * D + lane];
        __syncthreads();
        float proj_next = 0.0f;
#pragma unroll
        for (int j = 0; j < D; ++j) {
            proj_next += memory[j] * direction[j];
        }
        proj_next_all[step_base + lane] = proj_next;
        // Store this lane's dq, then compute the true dot product
        // sum_j(dq[j] * direction[j]) from shared memory.
        dq_all[step_base + lane] = dq;
        __syncthreads();
        float proj_dq = 0.0f;
#pragma unroll
        for (int j = 0; j < D; ++j) {
            proj_dq += dq_all[step_base + j] * direction[j];
        }

        // m_{t+1} = decay * H(m_t) + write*cand, so the unrotated state is
        // H^T(dq) = dq - 2 (dq.d) d, applied element-wise.
        g_read_all[step_base + lane] = dq - 2.0f * proj_dq * direction[lane];
        gate_val_all[step_base + lane] = gate_saved[base + lane];
        // read_val = raw read; gate_val = sigmoid(read_gate_in).  The module
        // computes `raw_read * gate_val` and dq = dL/d(raw_read) = 2 r g^2 / N,
        // so the gradient owed to the pre-activation read-gate input is
        // dL/d(read_gate) = 2 r^2 g^2 (1-g) / N = dq * raw_read * g^2 * (1-g).
        d_read_raw[base + lane] =
            dq * read_val[lane] * gate_val_all[step_base + lane]
            * gate_val_all[step_base + lane] * (1.0f - gate_val_all[step_base + lane]);
    }

    // Second pass: recover m_t, accumulate p, and emit per-step gradients.
    for (int t = T - 1; t >= 0; --t) {
        const int base = (batch * T + t) * D;
        const int step_base = t * D;
        const float g_read = g_read_all[step_base + lane];
        const float gate_val = gate_val_all[step_base + lane];
        cand[lane] = cand_saved[base + lane];
        write[lane] = write_saved[base + lane];
        read_val[lane] = read_saved[base + lane];
        decay[lane] = decay_saved[base + lane];

        __syncthreads();

        dstate[lane] += g_read;  // dstate now holds p_{t+1} (includes g_read_t)

        // cand = tanh(cand_raw), write = sigmoid(write_raw): the backward
        // must include the activation derivatives.
        d_cand[base + lane] = write[lane] * dstate[lane] * (1.0f - cand[lane] * cand[lane]);
        d_write[base + lane] = cand[lane] * dstate[lane] * write[lane] * (1.0f - write[lane]);

        // Store the full p_{t+1} (includes g_read_t and all later
        // contributions) for the final gradient pass.
        p_all[step_base + lane] = dstate[lane];

        // Propagate p: p_{t} -> decay * H^T(p_{t}) for the previous step.
        float proj_dstate = 0.0f;
#pragma unroll
        for (int j = 0; j < D; ++j) {
            proj_dstate += dstate[j] * direction[j];
        }
#pragma unroll
        for (int j = 0; j < D; ++j) {
            dstate[j] = decay[j] * (dstate[j] - 2.0f * proj_dstate * direction[j]);
        }

        __syncthreads();
    }

    // Final pass: emit the decay and rotation direction gradients using the
    // stored full p_{t+1} and checkpointed memory states.  Decay:
    // m_{t+1} = decay * H(m_t) + write*cand, so d_decay_t = H(m_t) * p_{t+1}
    // with H(m_t) = m_t - 2 proj(m_t) d.  Direction: the rotation into step
    // t+1 contributes -2 decay [(p.d) m_t + (m_t.d) p].
    for (int t = T - 1; t >= 0; --t) {
        const int base = (batch * T + t) * D;
        const int step_base = t * D;
        memory[lane] = mem_saved[batch * (T + 1) * D + step_base + lane];
        __syncthreads();
        float proj_mt = 0.0f;
#pragma unroll
        for (int j = 0; j < D; ++j) {
            proj_mt += memory[j] * direction[j];
        }
        const float h_mt =
            memory[lane] - 2.0f * proj_mt * direction[lane];
        d_decay[base + lane] = h_mt * p_all[step_base + lane];

        // Read part of the direction gradient: the derivative of H_d(m) with
        // respect to d contracted with the raw read gradient dq is
        // -2[(dq.d) m + (m.d) dq].
        float g_dot_d = 0.0f;
#pragma unroll
        for (int j = 0; j < D; ++j) {
            g_dot_d += dq_all[step_base + j] * direction[j];
        }
        direction_grad[lane] -= 2.0f * g_dot_d
                                * mem_saved[batch * (T + 1) * D + (t + 1) * D + lane];
        direction_grad[lane] -= 2.0f * proj_next_all[step_base + lane]
                                * dq_all[step_base + lane];

        // Rotation part: the direction enters m_{t+1} through
        // decay * H_d(m_t); with p = p_{t+1} the contribution is
        // -2 decay [(p.d) m_t + (m_t.d) p].
        if (t > 0) {
            float p_dot_d = 0.0f;
#pragma unroll
            for (int j = 0; j < D; ++j) {
                p_dot_d += p_all[step_base + j] * direction[j];
            }
            direction_grad[lane] -= 2.0f * decay_saved[base + lane]
                                    * p_dot_d * memory[lane];
            direction_grad[lane] -= 2.0f * decay_saved[base + lane]
                                    * proj_mt
                                    * p_all[step_base + lane];
        }
        __syncthreads();
    }

    // `direction` is a single vector shared by every batch element; each
    // block accumulates its own partial and the host sums them.
    d_direction_partial[batch * D + lane] = direction_grad[lane];
    // After the second pass, `dstate` holds p_0 = dL/dm_0.
    d_memory_out[batch * D + lane] = dstate[lane];
}

void rosa_cuda_forward(
    int B, int T, int D,
    const float* cand, const float* write, const float* read_raw,
    const float* decay, const float* direction, const float* memory,
    float* read_out, float* memory_out,
    float* cand_saved, float* write_saved, float* read_saved,
    float* gate_saved, float* decay_saved, float* mem_saved) {
    switch (D) {
        case 16:
            rosa_forward_kernel<16><<<B, 16>>>(
                T, cand, write, read_raw, decay, direction, memory,
                read_out, memory_out, cand_saved, write_saved, read_saved,
                gate_saved, decay_saved, mem_saved);
            break;
        case 32:
            rosa_forward_kernel<32><<<B, 32>>>(
                T, cand, write, read_raw, decay, direction, memory,
                read_out, memory_out, cand_saved, write_saved, read_saved,
                gate_saved, decay_saved, mem_saved);
            break;
        case 64:
            rosa_forward_kernel<64><<<B, 64>>>(
                T, cand, write, read_raw, decay, direction, memory,
                read_out, memory_out, cand_saved, write_saved, read_saved,
                gate_saved, decay_saved, mem_saved);
            break;
        case 128:
            rosa_forward_kernel<128><<<B, 128>>>(
                T, cand, write, read_raw, decay, direction, memory,
                read_out, memory_out, cand_saved, write_saved, read_saved,
                gate_saved, decay_saved, mem_saved);
            break;
        case 256:
            rosa_forward_kernel<256><<<B, 256>>>(
                T, cand, write, read_raw, decay, direction, memory,
                read_out, memory_out, cand_saved, write_saved, read_saved,
                gate_saved, decay_saved, mem_saved);
            break;
        default:
            throw std::runtime_error("ROSA CUDA kernel supports D in {16, 32, 64, 128, 256}");
    }
}

void rosa_cuda_backward(
    int B, int T, int D,
    const float* cand_saved, const float* write_saved, const float* read_saved,
    const float* gate_saved, const float* decay_saved, const float* mem_saved,
    const float* direction, const float* final_memory,
    const float* d_read,
    float* d_cand, float* d_write, float* d_read_raw, float* d_decay,
    float* d_direction_partial, float* d_memory_out, float* workspace) {
    switch (D) {
        case 16:
            rosa_backward_kernel<16><<<B, 16>>>(
                T, cand_saved, write_saved, read_saved, gate_saved,
                decay_saved, mem_saved, direction, final_memory, d_read, d_cand, d_write,
                d_read_raw, d_decay, d_direction_partial, d_memory_out, workspace);
            break;
        case 32:
            rosa_backward_kernel<32><<<B, 32>>>(
                T, cand_saved, write_saved, read_saved, gate_saved,
                decay_saved, mem_saved, direction, final_memory, d_read, d_cand, d_write,
                d_read_raw, d_decay, d_direction_partial, d_memory_out, workspace);
            break;
        case 64:
            rosa_backward_kernel<64><<<B, 64>>>(
                T, cand_saved, write_saved, read_saved, gate_saved,
                decay_saved, mem_saved, direction, final_memory, d_read, d_cand, d_write,
                d_read_raw, d_decay, d_direction_partial, d_memory_out, workspace);
            break;
        case 128:
            rosa_backward_kernel<128><<<B, 128>>>(
                T, cand_saved, write_saved, read_saved, gate_saved,
                decay_saved, mem_saved, direction, final_memory, d_read, d_cand, d_write,
                d_read_raw, d_decay, d_direction_partial, d_memory_out, workspace);
            break;
        case 256:
            rosa_backward_kernel<256><<<B, 256>>>(
                T, cand_saved, write_saved, read_saved, gate_saved,
                decay_saved, mem_saved, direction, final_memory, d_read, d_cand, d_write,
                d_read_raw, d_decay, d_direction_partial, d_memory_out, workspace);
            break;
        default:
            throw std::runtime_error("ROSA CUDA kernel supports D in {16, 32, 64, 128, 256}");
    }
}
