import torch
from torch import nn
import math
from model_args import DeepSeekV3ModelArgs
import ScaledDotProductAttentionWrapper

class Attention(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = (
            model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        )
        self.v_head_dim = model_args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias = False)

        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias = False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps = model_args.norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias = False)

        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias = False)

        self.kv_norm = nn.RMSNorm(self.kv_lora_rank  eps = model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias = False
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias = False)
        self.softmax_scale = self.qk_head_dim**-0.5

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0

        self.inner_attnetion = ScaledDotProductAttentionWrapper()


    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):

        batch_size, seq_len, _ = x.size()
        pass
        





        