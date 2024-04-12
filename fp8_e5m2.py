# 直接将fp16转化为fp8 e5m2
import torch
import torch.nn as nn
import fake_fp8quant_tools
from tqdm import tqdm
from logger import logger

class FakeQuantFp8(object):
    def __init__(self,model,dtype,device):
        self.model = model
        self.dtype = dtype
        self.device = device
        assert dtype in ['fp8_e5m2','fp8_e4m3']
        assert 'cuda' in self.device, '量化需要再GPU进行，请指定GPU设备'

    def quantize(self):
        if self.dtype == 'fp8_e4m3':
            logger.warning("FP16 直接转换到fp8 e4m3可能引起极大的精度损失，请谨慎使用")
        for name,module in tqdm(self.model.named_modules()):
            if isinstance(module,nn.Linear) and 'lm_head' not in name:
                tmp = torch.zeros_like(module.weight.data).to(self.device)
                fake_fp8quant_tools.fake_quant_fp8_e5m2(module.weight.data.to(self.device), tmp)
                module.weight.data.copy_(tmp.cpu())
        return self.model
        