#include <torch/extension.h>
#include <iostream>
#include <cassert>
#include <string.h>

torch::Tensor fake_quant_fp8(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache,
  std::string dtype);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_quant_fp8", &fake_quant_fp8, "fake_quant_fp8_e5m2");
}

