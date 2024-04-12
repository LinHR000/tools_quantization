from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os
import torch
model_path = '/mnt/data/linhaoran/models/Llama-2-13b'
quant_path = 'CUDA'
quant_config = { "zero_point": False, "q_group_size": 128, "w_bit": 4, "version": "marlin"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# kv_cache_dtype = 'fp8_e5m2'
kv_cache_dtype = 'None'
# Quantize
arch = model.config.architectures[0]
model.quantize(tokenizer, quant_config=quant_config,export_compatible=True,apply_clip=False,arch=arch,kv_cache_dtype=kv_cache_dtype)
# model.quantize(tokenizer, quant_config=quant_config,export_compatible=False)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')