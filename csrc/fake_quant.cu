#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#ifdef ENABLE_FP8_E5M2
#include "quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  typedef __hip_bfloat16 __nv_bfloat16;
#endif

namespace vllm {

template<typename Tout>
__global__ void fake_quant_fp8_e5m2_kernel(
  const Tout* __restrict__ src_cache,
  Tout* __restrict__ dst_cache,
  const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
#ifdef ENABLE_FP8_E5M2
    uint8_t tmp_val = fp8_e5m2_unscaled::vec_conversion<uint8_t,Tout>(src_cache[idx]);
    dst_cache[idx] = fp8_e5m2_unscaled::vec_conversion<Tout, uint8_t>(tmp_val);
#else
    assert(false);
#endif
  }
}

} // namespace vllm


#define CALL_FAKE_QUANT_FP8_E5M2(Tout)                                   \
  vllm::fake_quant_fp8_e5m2_kernel<Tout><<<grid, block, 0, stream>>>(    \
    reinterpret_cast<Tout*>(src_cache.data_ptr()),                       \
    reinterpret_cast<Tout*>(dst_cache.data_ptr()),                       \
    block_stride);

torch::Tensor fake_quant_fp8_e5m2(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache)
{
  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (src_cache.dtype() == at::ScalarType::Half) {
    CALL_FAKE_QUANT_FP8_E5M2(uint16_t);
  }else{
    TORCH_CHECK(false, "Unsupported data type:  ", src_cache.dtype());
  }
  return dst_cache;
}


