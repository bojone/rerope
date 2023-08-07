#! -*- coding: utf-8 -*-
# transformers 4.31.0 测试通过

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TopPLogitsWarper, LogitsProcessorList
from transformers import TextStreamer
from tensor_parallel import tensor_parallel
# import ntk_patch  # test NTK-RoPE-mixed
import rerope_patch  # test ReRoPE
# import leaky_rerope_patch  # test Leaky ReRoPE

# 加载模型
model_path = 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model = tensor_parallel(model)
device = torch.device('cuda')

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.unk_token
streamer = TextStreamer(tokenizer)

# 示例问题集
question = """请仔细阅读材料，然后回答：
- 菲律宾国家电网公司，中国占股多少？
- 领英计划裁员多少人？
- 吉利德收购Pharmasset的价格是多少？
- 丙肝神药Sovaldi在哪一年上市？
- 中亚峰会将在哪里举行？由谁主持？
- 哪个演员由于侮辱人民军队而被立案调查？
- 哪个项目宣称“能过坦克”的水上道路？
- 如果你是默沙东的CEO，你的首要任务是什么？"""

# 示例Context
contexts = json.load(open('contexts.json')) + json.load(open('contexts.100.json'))[:10]
context = '\n\n'.join(contexts)
context = 'User: %s\n\n%s\n\nAssistant:' % (context, question)

# Top-P截断
processors = LogitsProcessorList()
processors.append(TopPLogitsWarper(0.95))


@torch.inference_mode()
def generate(max_tokens):
    """采样演示代码
    """
    inputs = tokenizer([context], padding='longest', return_tensors='pt').to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    print('input_ids', input_ids.shape)
    past_key_values = None

    for i in range(max_tokens):
        # 模型输出
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=True,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values

        # 构建采样
        # tau = 1是标准的随机采样，tau->0则是贪心搜索
        tau = 0.01
        logits = processors(input_ids, outputs.logits[:, -1])
        probas = torch.nn.functional.softmax(logits / tau, dim=-1)
        next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)
        streamer.put(next_tokens)
        if next_tokens[0] == tokenizer.eos_token_id:
            break

        input_ids = next_tokens.unsqueeze(-1).tile(1, 1)
        ones = torch.ones(1, 1, dtype=torch.long, device=device)
        attention_mask = torch.cat([attention_mask, ones], dim=-1)


if __name__ == '__main__':
    import time
    start = time.time()
    generate(1000)
    print(f'time usage: {time.time() - start}')
