import torch
import fake_fp8quant_tools
inputs = torch.randn(1, 3, 224, 224).cuda().half()
output =torch.zeros_like(inputs)
fake_fp8quant_tools.fake_quant_fp8_e5m2(inputs,output)
print(inputs)
print(output)