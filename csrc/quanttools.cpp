#include <torch/extension.h>
#include <iostream>
#include <cassert>

torch::Tensor fake_quant_fp8_e5m2(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_quant_fp8_e5m2", &fake_quant_fp8_e5m2, "fake_quant_fp8_e5m2");
}

