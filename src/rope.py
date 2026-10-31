import math
import torch
from model.model_args import DeepSeekV3ModelArgs


def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seq_len = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    t = torch.arange(seq_len)

    freq = torch.outer(t, freqs)
    # convert complex expoentials to polar coordinates: e^(i*freq*pos) = cos(freq*pos) + i*sin(freq*pos)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # (max_seq_len, head_dim/2)
    return freqs_cis

def apply_rotary_emb(x:torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: [batch_size, seq_len, num_heads, head_dim]

    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2)) #=> [batch_size, seq_len, num_heads, head_dim/2]
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1)) #=> [1, seq_len, 1, head_dim/2]

    y = torch.view_as_real(x * freqs_cis).flatten(3).to(dtype) #=> [batch_size, seq_len, num_heads, head_dim]
    return y



   