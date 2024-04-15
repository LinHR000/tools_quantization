# 直接将fp16转化为fp8 e5m2
import torch
import torch.nn as nn
import fake_fp8quant_tools
from tqdm import tqdm
from logger import logger
from config.fp8_gptq_quant_controller import FP8GPTQQuantController  
from auto_gptq import BaseQuantizeConfig

# 引入smoothquant进行FP8量化

# W4AFp8量化流程
# 1、通过naive量化方式我们证明了，Fp16->FP8直接转换不会引入较大的误差
# 1.1、可以更加保护重要权重的选项
# 2、我们首先将W-FP16->W-INT4->W-FP8
#   2.1、先FP16->FP8,在FP8->int4
#   2.2、先FP16->int4,在int->FP8
# 3、A-FP16->FP8
# 4、引入层间微调和LM_head/norm微调

class FakeQuantFp8GPTQ(object):
    def __init__(self,args, model,dtype,device, quant_controllr:FP8GPTQQuantController,sm='80'):
        self.model = model
        self.dtype = dtype
        self.device = device
        assert dtype in ['fp8_e5m2','fp8_e4m3']
        assert 'cuda' in self.device, '量化需要再GPU进行，请指定GPU设备'
        self.sm = sm
        self.quant_controllr = quant_controllr
        self.quant_config = BaseQuantizeConfig(
            bits=args.abits,  # quantize model to 4-bit
            group_size=args.group_size,  # it is recommended to set the value to 128
            desc_act=False,  # desc_act and group size only works on triton
            is_marlin_format=False)


    def quantize(self,):
        model.quantize()
        if self.quant_config.pre_smooth:
            logger.info("使用AWQ对模型的重要权重进行保护")
            pass
        if self.quant_config.w_quant_order == 'fp16_to_int4_to_fp8':
            pass
        elif self.quant_config.w_quant_order == 'fp16_to_int4_to_fp8':
            pass
        else:
            raise ValueError(f"not support w_quant_order : {self.quant_config.w_quant_order}")

        pass
        