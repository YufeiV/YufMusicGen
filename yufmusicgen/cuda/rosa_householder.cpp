// C++ wrapper for the fused ROSA Householder scan.  The Python side always
// converts to float32 (matching the reference scan, which intentionally runs
// the recurrent accumulation in FP32), so this binding is float32-only.

#include <torch/extension.h>

void rosa_cuda_forward(
    int B, int T, int D,
    const float* cand, const float* write, const float* read_raw,
    const float* decay, const float* direction, const float* memory,
    float* read_out, float* memory_out,
    float* cand_saved, float* write_saved, float* read_saved,
    float* gate_saved, float* decay_saved, float* mem_saved);

void rosa_cuda_backward(
    int B, int T, int D,
    const float* cand_saved, const float* write_saved, const float* read_saved,
    const float* gate_saved, const float* decay_saved, const float* mem_saved,
    const float* direction, const float* final_memory,
    const float* d_read,
    float* d_cand, float* d_write, float* d_read_raw, float* d_decay,
    float* d_direction_partial, float* d_memory_out, float* workspace);

void rosa_forward(
    torch::Tensor& cand,
    torch::Tensor& write,
    torch::Tensor& read_raw,
    torch::Tensor& decay,
    torch::Tensor& direction,
    torch::Tensor& memory,
    torch::Tensor& read_out,
    torch::Tensor& memory_out,
    torch::Tensor& cand_saved,
    torch::Tensor& write_saved,
    torch::Tensor& read_saved,
    torch::Tensor& gate_saved,
    torch::Tensor& decay_saved,
    torch::Tensor& mem_saved) {
    const int B = static_cast<int>(cand.sizes()[0]);
    const int T = static_cast<int>(cand.sizes()[1]);
    const int D = static_cast<int>(cand.sizes()[2]);
    rosa_cuda_forward(
        B, T, D,
        cand.data_ptr<float>(), write.data_ptr<float>(), read_raw.data_ptr<float>(),
        decay.data_ptr<float>(), direction.data_ptr<float>(), memory.data_ptr<float>(),
        read_out.data_ptr<float>(), memory_out.data_ptr<float>(),
        cand_saved.data_ptr<float>(), write_saved.data_ptr<float>(),
        read_saved.data_ptr<float>(), gate_saved.data_ptr<float>(),
        decay_saved.data_ptr<float>(), mem_saved.data_ptr<float>());
}

void rosa_backward(
    torch::Tensor& cand_saved,
    torch::Tensor& write_saved,
    torch::Tensor& read_saved,
    torch::Tensor& gate_saved,
    torch::Tensor& decay_saved,
    torch::Tensor& mem_saved,
    torch::Tensor& direction,
    torch::Tensor& final_memory,
    torch::Tensor& d_read,
    torch::Tensor& d_cand,
    torch::Tensor& d_write,
    torch::Tensor& d_read_raw,
    torch::Tensor& d_decay,
    torch::Tensor& d_direction_partial,
    torch::Tensor& d_memory_out,
    torch::Tensor& workspace) {
    const int B = static_cast<int>(cand_saved.sizes()[0]);
    const int T = static_cast<int>(cand_saved.sizes()[1]);
    const int D = static_cast<int>(cand_saved.sizes()[2]);
    rosa_cuda_backward(
        B, T, D,
        cand_saved.data_ptr<float>(), write_saved.data_ptr<float>(),
        read_saved.data_ptr<float>(), gate_saved.data_ptr<float>(),
        decay_saved.data_ptr<float>(), mem_saved.data_ptr<float>(),
        direction.data_ptr<float>(), final_memory.data_ptr<float>(),
        d_read.data_ptr<float>(),
        d_cand.data_ptr<float>(), d_write.data_ptr<float>(),
        d_read_raw.data_ptr<float>(), d_decay.data_ptr<float>(),
        d_direction_partial.data_ptr<float>(), d_memory_out.data_ptr<float>(),
        workspace.data_ptr<float>());
}

TORCH_LIBRARY(yufmusicgen_rosa, m) {
    m.def("forward", rosa_forward);
    m.def("backward", rosa_backward);
}
