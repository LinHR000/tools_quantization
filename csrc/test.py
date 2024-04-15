import torch
import fake_fp8quant_tools
inputs = torch.randn(1, 3, 224, 224).cuda().half()
output =torch.zeros_like(inputs)
fake_fp8quant_tools.fake_quant_fp8(inputs,output,'fp8_e5m2')
print(inputs)
print(output)

fake_fp8quant_tools.fake_quant_fp8(inputs,output,'fp8_e4m3')
print(inputs)
print(output)