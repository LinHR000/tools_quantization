import torch
from tqdm import tqdm

def get_models(model_path, arch,quant_dtype,sm='80'):
    if sm == '80':
        if "Llama-2" in arch:
            from models_te.modeling_llama import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.float16,device_map='cpu')
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.float16,device_map='cpu')
        return model
    elif sm =='89' or sm >= '90':
        if arch =='qwen2':
            if quant_dtype == 'fp8_e5m2':
                from models_te.modeling_qwen2_fp8_e5m2 import LlamaForCausalLM
                model = LlamaForCausalLM.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.float16,device_map='cpu')
                return model
            elif quant_dtype == 'fp8_e4m3':
                from models_te.modeling_qwen2_fp8_e4m3 import LlamaForCausalLM
                model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,trust_remote_code=True,torch_dtype=torch.float16,device_map='cpu')
                return model
            else:
                raise ValueError()
        elif "Llama-2" in arch:
            if quant_dtype == 'fp8_e5m2':
                from models_te.modeling_llama_fp8_e5m2 import LlamaForCausalLM
                model = LlamaForCausalLM.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.float16,device_map='cpu')
                return model
            elif quant_dtype == 'fp8_e4m3':
                from models_te.modeling_llama_fp8_e4m3 import LlamaForCausalLM
                model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,trust_remote_code=True,torch_dtype=torch.float16,device_map='cpu')
                return model
            else:
                raise ValueError()
        else:
            raise ValueError(f"not support arch:{arch}")

def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        # for l_idx in range(len(levels) - 1):
        #     if levels[l_idx].isdigit():
        #         mod_ = mod_[int(levels[l_idx])]
        #     else:
        #         mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def replace_modules(model,arch,quant_dtype,config,sm='80'):
    if sm == '89' or sm >= '90':
        if "Llama-2" in arch:
            from models_te.te_llama import TeLlamaMLP,TeLlamaAttention,TeLlamaFlashAttention2,TeLlamaSdpaAttention
            from models_te.modeling_llama import LlamaMLP,LlamaAttention,LlamaFlashAttention2,LlamaSdpaAttention, LlamaDecoderLayer
            for name,module in tqdm(model.named_modules(),desc="replacing fp8 modules"):
                if isinstance(module,LlamaDecoderLayer):
                    new_mlp_module = TeLlamaMLP(config, module.mlp, quant_dtype).half().cpu()
                    set_op_by_name(module,name+".mlp",new_mlp_module)
                    if isinstance(module.self_attn,LlamaSdpaAttention):
                        new_attn_module = TeLlamaSdpaAttention(config, module.self_attn.layer_idx,module.self_attn,quant_dtype=quant_dtype).half().cpu()
                    elif isinstance(module.self_attn,LlamaFlashAttention2):
                        new_attn_module = TeLlamaFlashAttention2(config, module.self_attn.layer_idx,module.self_attn,quant_dtype=quant_dtype).half().cpu()
                    elif isinstance(module.self_attn,LlamaAttention):
                        new_attn_module = TeLlamaAttention(config, module.self_attn.layer_idx,module.self_attn,quant_dtype=quant_dtype).half().cpu()
                    
                    set_op_by_name(module,name+".self_attn",new_attn_module)
        else:
            raise ValueError(f"not support arch {arch}")
        pass