import torch
import math
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from torch import Tensor
from torch.nn import functional as F


transformer_configs = {
    "Llama-2-7b-chat": dict(num_layers=32, num_heads=32, embed_dim=4096)
}

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class Config:
    vocab_size: int = 32000
    embed_dim: int = 64
    seq_len: int = 32
    num_heads: int = 16
    num_layers: int = 10
    test_batch_size: int = 64
    intermediate_size: Optional[int] = None
    rope_base: int = 10000
    block_size: int = 2048

    def __post_init__(self):
        if self.intermediate_size is None:
            hidden_dim = 4 * self.embed_dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        num_heads = config.num_heads
        num_layers = config.num_layers
        vocab_size = config.vocab_size
        print("vocab size is unknown but set to", vocab_size)

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        print("embedding layers created")
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(num_layers)])
        print("transformer layers created")
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.freqs_cis: Optional[Tensor] = None

    def setup_caches(self):
        self.freqs_cis = precompute_freqs_cis(
            seq_len=self.config.block_size, 
            n_elem=self.config.embed_dim // self.config.num_heads, 
            base=self.config.rope_base, 
            dtype=self.output.weight.dtype)
    
    def forward(self, x, input_pos):
        # x: int, [B, T, dim] // 
        print("Input", x.sum())
        freqs_cis = self.freqs_cis[input_pos] # [B, T, head_dim / 2, 2]
        x = self.tok_embeddings(x) # [B, seq_len, dim]
        print("embed tensor", x.sum())
        for i, block in enumerate(self.layers):
            x = block(x, input_pos, freqs_cis)
            if i < 1:
                print(f"[{i}] layer", x.sum())
        print(f"Last layer", x.sum())
        x = self.norm(x)
        out = self.output(x)
        print("Output", x.sum())
        return out

    @classmethod
    def from_name(cls, name):
        candidates = [k for k in transformer_configs if k in name]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous model name: {name}, {candidates}")
        if len(candidates) == 0:
            raise ValueError(f"Unknown model name: {name}")
        name = candidates[0]
        cfg = Config(**transformer_configs[name])
        return cls(cfg)



class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.embed_dim)
        self.attention = SelfAttention(cfg.num_heads, cfg.embed_dim)
        self.ln_2 = RMSNorm(cfg.embed_dim)
        self.ff = FeedForward(cfg)

    def forward(self, x, input_pos: Tensor, freqs_cis: Tensor):
        x = self.attention(self.ln_1(x), input_pos, freqs_cis) + x
        out = self.ff(self.ln_2(x)) + x
        return out


class SelfAttention(nn.Module):
    def __init__(self, num_heads, dimension) -> None:
        super().__init__()
        self.Wq = nn.Linear(dimension, dimension, bias=False)
        self.Wk = nn.Linear(dimension, dimension, bias=False)
        self.Wv = nn.Linear(dimension, dimension, bias=False)
        self.Wo = nn.Linear(dimension, dimension, bias=False)
        self.num_heads = num_heads
        self.dimension = dimension
    
    def forward(self, x, input_pos: Tensor, freqs_cis: Tensor): # x: [Batch, T, dims]
        B, T, dims = x.shape # T = seqlen
        q = self.Wq(x).view(B, T, self.dimension // self.num_heads, self.num_heads) # q: [B, T, d_h, num_heads]
        k = self.Wk(x).view(B, T, self.dimension // self.num_heads, self.num_heads) # k: [B, T, d_h, num_heads]
        v = self.Wv(x).view(B, T, self.dimension // self.num_heads, self.num_heads) # v: [B, T, d_h, num_heads]

        q = apply_rotary_emb(q, freqs_cis) # B, T, d_h, num_heads
        k = apply_rotary_emb(k, freqs_cis)

        B, T, d_h, num_heads = q.shape
        q = q.reshape(B, num_heads, T, d_h)
        k = k.reshape(B, num_heads, d_h, T)
        v = v.reshape(B, num_heads, T, d_h)

        score = q @ k # [B, num_heads, T, d_h] x [B, num_heads, d_h, T] -> [B, num_heads, T, T]
        score = score / math.sqrt(k.shape[2])
        soft = torch.softmax(score, dim=-1) 
        result = soft @ v # [B,num_heads, T, T] x [B, num_heads, T, d_h] -> [B, num_heads, T, d_h]
        result = result.transpose(1, 2).contiguous().view(B, T, dims) # B, T, num_heads * d_h
        
        result = self.Wo(result)
        return result


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up_proj = nn.Linear(cfg.embed_dim, cfg.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(cfg.embed_dim, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.embed_dim, bias=False)
        

    def forward(self, x):
        return self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x))



def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    B, T, head_dim, num_heads = x.shape
    xshaped = x.float().reshape(B, T, num_heads, head_dim // 2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    x_out2 = x_out2.reshape(B, T, head_dim, num_heads)
    return x_out2.type_as(x)


# if __name__ == "__main__":
#     # test_attention(Config())
#     test_model(Config())
#     # prompt = ""
#     # tokenizer = Tokenizer()
#     # model = Model()
#     # print(model(prompt))


# def test_attention(cfg):
#     attn = SelfAttention(num_heads=8, dimension=cfg.dims)
#     embedding = torch.rand(cfg.seq_len, cfg.dims)
#     result = attn(embedding)
#     print(result.shape, torch.sum(result))

# @torch.inference_mode()
# def test_model(cfg: Config):
#     model = Transformer(cfg)
#     my_input = torch.randint(0, cfg.vocab_size, (cfg.test_batch_size, cfg.seq_len,))
#     result = model(my_input)
#     print(result.shape, torch.sum(result))
