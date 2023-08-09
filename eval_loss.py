#! -*- coding: utf-8 -*-
# 使用llama2-13b测试英文数据集loss
# transformers 4.31.0 测试通过

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensor_parallel import tensor_parallel
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss
# import ntk_patch  # test NTK-RoPE-mixed
import rerope_patch  # test ReRoPE
# import leaky_rerope_patch  # test Leaky ReRoPE


model_path = 'meta-llama/Llama-2-13b-hf'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval()
model = tensor_parallel(model)
device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_path)

L = 4096 * 4
loss = 0

with torch.no_grad():
    with open('samples_15k.jsonl') as fr:
        for i, l in enumerate(tqdm(fr, ncols=0)):
            text = json.loads(l)['text']
            input_ids = tokenizer([text], return_tensors='pt').to(device).input_ids
            input_ids = input_ids[:, -(L + 1):]
            logits = model(input_ids[:, :-1]).logits[:, -4096:].view(-1, 32000)
            labels = input_ids[:, -4096:].view(-1).to(logits.device)
            loss += CrossEntropyLoss()(logits, labels)
            print({'total': i + 1, 'loss': float(loss) / (i + 1)})
