import torch
import math
import torch.nn as nn
from dataclasses import dataclass

from torch.nn import functional as F


transformer_configs = {
    "Llama-2-7b-chat": dict(num_layers=32, num_heads=32, dims=4096)
}

@dataclass
class Config:
    vocab_size: int = 1000
    dims: int = 64
    seq_len: int = 32
    num_heads: int = 16
    num_layers: int = 10
    test_batch_size: int = 64


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.dims
        num_heads = config.num_heads
        num_layers = config.num_layers
        vocab_size = config.vocab_size

        self.word_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)
    
    def forward(self, x):
        # x: int, [seq_len]
        x = self.word_embed(x) # [seq_len, dim]
        for block in self.blocks:
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
        self.Wq = nn.Linear(dimension, dimension)
        self.Wk = nn.Linear(dimension, dimension)
        self.Wv = nn.Linear(dimension, dimension)
        self.Wo = nn.Linear(dimension, dimension)
        self.scale = 1 / math.sqrt(dimension)
        self.num_heads = num_heads
        self.dimension = dimension
    
    def forward(self, x): # x: [Batch, seq_len, dims]
        B, seq_len, dims = x.shape
        q = self.Wq(x).view(-1, self.dimension // self.num_heads, self.num_heads) # q: [B, seq_len, d_h, num_heads]
        k = self.Wk(x).view(-1, self.dimension // self.num_heads, self.num_heads) # k: [B, seq_len, d_h, num_heads]
        v = self.Wv(x).view(-1, self.dimension // self.num_heads, self.num_heads) # v: [B, seq_len, d_h, num_heads]
        score = q @ k.transpose(1, 2) * self.scale # : [B, seq_len, seq_len, num_heads]
        soft = torch.softmax(score, dim=-1) 
        result = soft @ v # [B, seq_len, d_h, num_heads]
        result = result.view(B, seq_len, dims) # B, seq_len, num_heads * d_h
        result = self.Wo(result)
        return result

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(num_heads, embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x):
        x = self.attention(self.ln_1(x) + x)
        out = self.ff(self.ln_2(x) + x)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.up_proj = nn.Linear(embed_dim, 4*embed_dim)
        self.gate_proj = nn.Linear(embed_dim, 4*embed_dim)
        self.down_proj = nn.Linear(4*embed_dim, embed_dim)
        

    def forward(self, x):
        return self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x))


def test_attention(cfg):
    attn = SelfAttention(num_heads=8, dimension=cfg.dims)
    embedding = torch.rand(cfg.seq_len, cfg.dims)
    result = attn(embedding)
    print(result.shape, torch.sum(result))

@torch.inference_mode()
def test_model(cfg):
    model = Transformer(cfg.vocab_size, cfg.num_heads, cfg.dims, cfg.num_layers)
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