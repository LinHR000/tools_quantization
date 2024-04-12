import os
import sys
import random
import numpy as np
import torch
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

seqlen = 2048
model_path = '/mnt/data/linhaoran/models/Llama-2-13b'
seed = 2
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
if hasattr(config, 'quantization_config'):
    delattr(config, "quantization_config")
model  = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map='auto',torch_dtype=config.torch_dtype,trust_remote_code=True)
# for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
# for dataset in ["wikitext2", "c4"]:
for dataset in ["wikitext2"]:
    dataloader, testloader = get_loaders(
        dataset,
        seed=seed,
        model=model_path,
        seqlen=seqlen,
    )
        # torch.save(testloader, cache_testloader)
    if "c4" in dataset:
        testenc = testloader
    else:
        testenc = testloader.input_ids

    nsamples = testenc.numel() // seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    nlls = []
    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].cuda()
            output = model(batch)
            logits = output.logits.detach().clone()
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:].cuda()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float().detach().clone() * seqlen
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(ppl)
    
