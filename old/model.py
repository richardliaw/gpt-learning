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
class ModelArgs:
    vocab_size: int = 32000
    embed_dim: int = 64
    seq_len: int = 32
    num_heads: int = 16
    num_layers: int = 10
    test_batch_size: int = 64
    intermediate_size: Optional[int] = None
    rope_base: int = 10000
    block_size: int = 2048
    head_dim: int = 64

    def __post_init__(self):
        if self.intermediate_size is None:
            hidden_dim = 4 * self.embed_dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.embed_dim // self.num_heads

    @classmethod
    def from_name(cls, name):
        candidates = [k for k in transformer_configs if k in name]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous model name: {name}, {candidates}")
        if len(candidates) == 0:
            raise ValueError(f"Unknown model name: {name}")
        name = candidates[0]
        return cls(**transformer_configs[name])


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

    def setup_caches(self, max_seq_len):
        self.freqs_cis = precompute_freqs_cis(
            seq_len=self.config.block_size, 
            n_elem=self.config.embed_dim // self.config.num_heads, 
            base=self.config.rope_base, 
            dtype=self.output.weight.dtype)
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
    
    def forward(self, x, input_pos):
        # x: int, [B, T, dim] // 
        print("-------")
        print("Input", x.sum(), input_pos)
        freqs_cis = self.freqs_cis[input_pos] # [B, T, head_dim / 2, 2]
        x = self.tok_embeddings(x) # [B, seq_len, dim]
        print("embed tensor", x.sum())
        print("freq_cis", float(freqs_cis.sum()), float(freqs_cis.mean()))
        for i, block in enumerate(self.layers):
            x = block(x, input_pos, freqs_cis, mask=self.mask)
            print(f"[{i}] layer", float(x.sum()), float(x.mean()))
        x = self.norm(x)
        out = self.output(x)
        print("Output", out.sum())
        return out

    @classmethod
    def from_name(cls, name):
        cfg = ModelArgs.from_name(name)
        return cls(cfg)



class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.embed_dim)
        self.attention = SelfAttention(cfg.num_heads, cfg.embed_dim, cfg.head_dim)
        self.ln_2 = RMSNorm(cfg.embed_dim)
        self.ff = FeedForward(cfg)

    def forward(self, x, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor):
        x = self.attention(self.ln_1(x), input_pos, freqs_cis, mask) + x
        out = self.ff(self.ln_2(x)) + x
        return out


class SelfAttention(nn.Module):
    def __init__(self, num_heads, dimension, head_dim) -> None:
        super().__init__()
        self.Wq = nn.Linear(dimension, dimension, bias=False)
        self.Wk = nn.Linear(dimension, dimension, bias=False)
        self.Wv = nn.Linear(dimension, dimension, bias=False)
        self.Wo = nn.Linear(dimension, dimension, bias=False)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dimension = dimension
    
    def forward(self, x, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor): # x: [Batch, T, dims]
        B, T, dims = x.shape # T = seqlen
        q = self.Wq(x)
        print(q)
        import ipdb; ipdb.set_trace()
        q = q.reshape(B, T, self.num_heads, self.head_dim) # q: [B, T, num_heads, d_h]
        k = self.Wk(x).reshape(B, T, self.num_heads, self.head_dim) # k: [B, T, num_heads, d_h]
        v = self.Wv(x).reshape(B, T, self.num_heads, self.head_dim) # v: [B, T, num_heads, d_h]
        print(". q, k, v", list(map(lambda x: (float(x.mean()), float(x.sum())), (q, k, v))))
        

        # q = apply_rotary_emb(q, freqs_cis) # B, T, d_h, num_heads
        # k = apply_rotary_emb(k, freqs_cis)
        print(". after rotary q, k", list(map(lambda x: float(x.sum()), (q, k))))

        B, T, num_heads, d_h = q.shape
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        attn = q @ k.transpose(-2, -1) # [B, num_heads, T, d_h] x [B, num_heads, d_h, T] -> [B, num_heads, T, T]
        attn = attn / math.sqrt(k.shape[-1])
        print(". q@k/s", float(attn.mean()), float(attn.sum()))
        assert attn.shape == (B, num_heads, T, T)
        import ipdb; ipdb.set_trace()
        # Main problem right now: need to support both decode (T=1, Pos=N) and prefill (T=T, Pos=0,T)
        # 
        # [B, num_heads, T, T]
        # input_pos: [B, seq_len]
        attn = attn.masked_fill(mask[input_pos, :T]== 0, float('-inf')) # this is probably wrong
        attn = attn
        score = torch.softmax(attn, dim=-1) 

        print(". soft(q@k/s)", float(score.mean()), float(score.sum()))
        """
        ipdb> score.sum(dim=2)
        tensor([[[1.0078, 0.9922],
                [1.0703, 0.9297],
                [1.0703, 0.9297],
                [1.3438, 0.6562],
                [1.1797, 0.8203],
                [1.9219, 0.0811],
                [1.1562, 0.8398],
                [1.2188, 0.7852],
                [1.1719, 0.8242],
                [1.0312, 0.9648],
                [1.8750, 0.1196],
                [1.1172, 0.8828],
                [1.4062, 0.5977],

        ipdb> score.sum(dim=1)
        tensor([[[32.0000,  0.0000],
                [ 9.5625, 22.3750]]], device='cuda:0', dtype=torch.bfloat16,
            grad_fn=<SumBackward1>)
        """
        result = score @ v # [B,num_heads, T, T] x [B, num_heads, T, d_h] -> [B, num_heads, T, d_h]
        assert result.shape == (B, num_heads, T, d_h)
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
        # self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))



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
