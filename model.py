import torch
import math
import torch.nn as nn

# Create the model

class Transformer(nn.Module):
    def __init__(self, vocab_size, num_heads, dim, num_layers):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)
    
    def forward(self, x):
        # x: int, [seq_len]
        x = self.word_embed(x) # [seq_len, dim]
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.output(x)


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
        self.ff = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.ln_1(self.attention(x) + x)
        out = self.ln_2(self.ff(x) + x)
        return out


from dataclasses import dataclass

@dataclass
class Config:
    vocab_size = 1000
    dims = 64
    seq_len = 32
    num_heads = 16
    num_layers = 10
    batch_size = 2


def test_attention(cfg):
    attn = SelfAttention(num_heads=8, dimension=cfg.dims)
    embedding = torch.rand(cfg.seq_len, cfg.dims)
    result = attn(embedding)
    print(result.shape, torch.sum(result))

def test_model(cfg):
    model = Transformer(cfg.vocab_size, cfg.num_heads, cfg.dims, cfg.num_layers)
    my_input = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len,))
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