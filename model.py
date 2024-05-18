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

    def __post_init__(self):
        if self.intermediate_size is None:
            hidden_dim = 4 * self.embed_dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        # self.head_dim = self.dims // self.num_heads


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
        dim = config.embed_dim
        num_heads = config.num_heads
        num_layers = config.num_layers
        vocab_size = config.vocab_size
        print("vocab size is unknown but set to", vocab_size)

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        # self.pos_embed = nn.Embedding(vocab_size, dim)
        print("embedding layers created")
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(num_layers)])
        print("transformer layers created")
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
    
    def forward(self, x):
        # x: int, [B, seq_len]
        x = self.tok_embeddings(x) # [B, seq_len, dim]
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        return self.output(x)

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


class SelfAttention(nn.Module):
    def __init__(self, num_heads, dimension) -> None:
        super().__init__()
        self.Wq = nn.Linear(dimension, dimension, bias=False)
        self.Wk = nn.Linear(dimension, dimension, bias=False)
        self.Wv = nn.Linear(dimension, dimension, bias=False)
        self.Wo = nn.Linear(dimension, dimension, bias=False)
        self.scale = 1 / math.sqrt(dimension)
        self.num_heads = num_heads
        self.dimension = dimension
    
    def forward(self, x): # x: [Batch, T, dims]
        B, T, dims = x.shape # T = seqlen
        q = self.Wq(x).view(-1, self.dimension // self.num_heads, self.num_heads) # q: [B, T, d_h, num_heads]
        k = self.Wk(x).view(-1, self.dimension // self.num_heads, self.num_heads) # k: [B, T, d_h, num_heads]
        v = self.Wv(x).view(-1, self.dimension // self.num_heads, self.num_heads) # v: [B, T, d_h, num_heads]
        score = q @ k.transpose(1, 2) # [B, T, d_h, num_heads] x [B, d_h, T, num_heads] -> [B, T, T, num_heads]
        score = score * self.scale
        soft = torch.softmax(score, dim=-1) 
        result = soft @ v # [B, T, T, num_heads] x [B, T, d_h, num_heads] -> [B, T, d_h, num_heads]
        result = result.view(B, T, dims) # B, T, num_heads * d_h
        result = self.Wo(result)
        return result

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.embed_dim)
        self.attention = SelfAttention(cfg.num_heads, cfg.embed_dim)
        self.ln_2 = RMSNorm(cfg.embed_dim)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = self.attention(self.ln_1(x) + x)
        out = self.ff(self.ln_2(x) + x)
        return out

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up_proj = nn.Linear(cfg.embed_dim, cfg.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(cfg.embed_dim, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.embed_dim, bias=False)
        

    def forward(self, x):
        return self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x))


def test_attention(cfg):
    attn = SelfAttention(num_heads=8, dimension=cfg.dims)
    embedding = torch.rand(cfg.seq_len, cfg.dims)
    result = attn(embedding)
    print(result.shape, torch.sum(result))

@torch.inference_mode()
def test_model(cfg: Config):
    model = Transformer(cfg)
    my_input = torch.randint(0, cfg.vocab_size, (cfg.test_batch_size, cfg.seq_len,))
    result = model(my_input)
    print(result.shape, torch.sum(result))

class Tokenizer(nn.Module):
    pass



if __name__ == "__main__":
    # test_attention(Config())
    test_model(Config())
    # prompt = ""
    # tokenizer = Tokenizer()
    # model = Model()
    # print(model(prompt))