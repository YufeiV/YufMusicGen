// Modified from BlinkDL/RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_clampw.cpp.
// YufMusicGen changes: project-local operator namespace and explicit tensor
// layout contract. Original source is Apache-2.0; see third_party/RWKV7_NOTICE.md.

#include <torch/extension.h>

#ifdef _FP32_
using scalar_t = float;
#else
#include <cuda_bf16.h>
using scalar_t = __nv_bfloat16;
#endif

void cuda_forward(int B, int T, int H, scalar_t* r, scalar_t* w, scalar_t* k,
                  scalar_t* v, scalar_t* a, scalar_t* b, scalar_t* y,
                  float* state, float* sa);

void forward(torch::Tensor& r, torch::Tensor& w, torch::Tensor& k, torch::Tensor& v,
             torch::Tensor& a, torch::Tensor& b, torch::Tensor& y,
             torch::Tensor& state, torch::Tensor& sa) {
    const int B = static_cast<int>(r.sizes()[0]);
    const int T = static_cast<int>(r.sizes()[1]);
    const int H = static_cast<int>(r.sizes()[2]);
    cuda_forward(B, T, H,
                 reinterpret_cast<scalar_t*>(r.data_ptr()),
                 reinterpret_cast<scalar_t*>(w.data_ptr()),
                 reinterpret_cast<scalar_t*>(k.data_ptr()),
                 reinterpret_cast<scalar_t*>(v.data_ptr()),
                 reinterpret_cast<scalar_t*>(a.data_ptr()),
                 reinterpret_cast<scalar_t*>(b.data_ptr()),
                 reinterpret_cast<scalar_t*>(y.data_ptr()),
                 state.data_ptr<float>(), sa.data_ptr<float>());
}

void cuda_backward(int B, int T, int H, scalar_t* r, scalar_t* w, scalar_t* k,
                   scalar_t* v, scalar_t* a, scalar_t* b, scalar_t* dy,
                   float* state, float* sa, scalar_t* dr, scalar_t* dw,
                   scalar_t* dk, scalar_t* dv, scalar_t* da, scalar_t* db);

void backward(torch::Tensor& r, torch::Tensor& w, torch::Tensor& k, torch::Tensor& v,
              torch::Tensor& a, torch::Tensor& b, torch::Tensor& dy,
              torch::Tensor& state, torch::Tensor& sa, torch::Tensor& dr,
              torch::Tensor& dw, torch::Tensor& dk, torch::Tensor& dv,
              torch::Tensor& da, torch::Tensor& db) {
    const int B = static_cast<int>(r.sizes()[0]);
    const int T = static_cast<int>(r.sizes()[1]);
    const int H = static_cast<int>(r.sizes()[2]);
    cuda_backward(B, T, H,
                  reinterpret_cast<scalar_t*>(r.data_ptr()),
                  reinterpret_cast<scalar_t*>(w.data_ptr()),
                  reinterpret_cast<scalar_t*>(k.data_ptr()),
                  reinterpret_cast<scalar_t*>(v.data_ptr()),
                  reinterpret_cast<scalar_t*>(a.data_ptr()),
                  reinterpret_cast<scalar_t*>(b.data_ptr()),
                  reinterpret_cast<scalar_t*>(dy.data_ptr()),
                  state.data_ptr<float>(), sa.data_ptr<float>(),
                  reinterpret_cast<scalar_t*>(dr.data_ptr()),
                  reinterpret_cast<scalar_t*>(dw.data_ptr()),
                  reinterpret_cast<scalar_t*>(dk.data_ptr()),
                  reinterpret_cast<scalar_t*>(dv.data_ptr()),
                  reinterpret_cast<scalar_t*>(da.data_ptr()),
                  reinterpret_cast<scalar_t*>(db.data_ptr()));
}

TORCH_LIBRARY(yufmusicgen_rwkv7, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
