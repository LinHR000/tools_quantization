import os
import torch
import argparse
from transformers import AutoTokenizer,AutoConfig
from fp8_quantize import FakeQuantFp8
from eval_ppl import llama_eval
from datautils import get_loaders
from logger import logger
from model_utils import get_models,replace_modules
parser = argparse.ArgumentParser()
# =============================模型输入输出参数=============================================================================================================
parser.add_argument("--model_path", type=str, default='/mnt/data/linhaoran/models/Llama-2-13b', help="model name of model path") 
# parser.add_argument("--model_path", type=str, default='/mnt/project/skyllm/linhaoran/models/Llama-2-13b', help="model name of model path") 
parser.add_argument("--save_path", default="/mnt/data/linhaoran/models/Llama-2-13b-fp8-e5m2-navie", type=str, help="direction of logging file")
parser.add_argument("--quant_dtype", default="fp8_e5m2", type=str, help="direction of logging file")
parser.add_argument("--sm", default="89", type=str, help="direction of logging file")
parser.add_argument("--quant_mode", default="gptq", type=str, choices=['naive','gptq'], help="direction of logging file")
parser.add_argument("--calib_dataset",type=str,default="wikitext2",
    choices=["wikitext2", "ptb", "c4", "mix","pile"],
    help="Where to extract calibration data from.",
)
parser.add_argument("--eval_ppl",default=False, action="store_true")
parser.add_argument("--use_eval",default=False, action="store_true")
parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
parser.add_argument("--seqlen", type=int, default=2048, help="Number of calibration data samples.")
parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
parser.add_argument("--seed", type=int, default=42, help="batch size.")
parser.add_argument("--dev", type=str, default='cuda:0', help="batch size.")
# =============================模型训练超参参数=============================================================================================================
# 量化比特设置
parser.add_argument("--wbits", type=int, default=8)
parser.add_argument("--abits", type=int, default=8)
args = parser.parse_args()
args.eval_ppl = True
args.quant_dtype = 'fp8_e5m2'
args.save_path = args.model_path + "-" + args.quant_dtype + "-" + args.quant_mode


logger.info("#"*20+"量化参数设置"+"#"*20)
w_dtype = "int4" if args.wbits == 4 else args.quant_dtype
logger.info(f"weight     : {args.wbits} bits, quantize dtype : {w_dtype}")
logger.info(f"activation : {args.abits} bits, quantize dtype : {args.quant_dtype}")


# 获取模型名称
config = AutoConfig.from_pretrained(args.model_path,trust_remote_code=True)
model_name = config._name_or_path.split("/")[-1]
traindataset, testenc = None,None
model = get_models(args.model_path,model_name,args.quant_dtype,args.quant_mode)
if args.quant_mode == 'naive':
    quant_instance = FakeQuantFp8(model,args.quant_dtype,device=args.dev,sm=args.sm)
    model = quant_instance.quantize()
else:
    model.quantize()
replace_modules(model,model_name,args.quant_dtype,config,args.sm)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# tokenizer.save_pretrained(args.save_path)
# model.save_pretrained(args.save_path)
# logger.info("model has been saved to %s",args.save_path)

if args.eval_ppl:
    if testenc is None:
        traindataset_path = os.path.join('cache/', f'traindataset-{model_name}-{args.seqlen}.cache')
        testenc_path = os.path.join('cache/', f'testenc-{model_name}-{args.seqlen}.cache')
        if not os.path.isfile(traindataset_path):
            traindataset, testenc = get_loaders(
            'wikitext2',
            seed=args.seed,
            model=args.model_path,
            seqlen=args.seqlen)
            torch.save(traindataset, traindataset_path)
            torch.save(testenc, testenc_path)
        traindataset = torch.load(traindataset_path)
        testenc = torch.load(testenc_path)
    llama_eval(model, testenc, args.dev, seqlen=args.seqlen)