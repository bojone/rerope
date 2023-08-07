#! -*- coding: utf-8 -*-
# NTK-RoPE-logn-mixed
# 链接：https://kexue.fm/archives/9706
# transformers 4.31.0 测试通过

import torch
from transformers.models.llama import modeling_llama
import numpy as np


def ntk_rope_mixed_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    old_init(self, dim, max_position_embeddings, base, device)
    k, b = 12, 0.75
    max_position_embeddings = training_length * k
    a = np.log(k) / (dim / 2)**b
    inv_freq = base**(-torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq *= (-a * torch.arange(1, dim // 2 + 1).float().to(device)**b).exp()
    self.register_buffer('inv_freq', inv_freq)
    self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.get_default_dtype())


def apply_rotary_pos_emb_and_logn_scale(q, k, cos, sin, position_ids):
    q_embed, k_embed = old_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    scale = ((position_ids + 1)[:, None, :, None].log() / np.log(training_length)).clip(1)
    return q_embed * scale.to(q_embed.dtype), k_embed


training_length = 4096
old_init = modeling_llama.LlamaRotaryEmbedding.__init__
old_apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
modeling_llama.LlamaRotaryEmbedding.__init__ = ntk_rope_mixed_init
modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb_and_logn_scale
