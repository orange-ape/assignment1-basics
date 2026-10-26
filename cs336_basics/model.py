import torch
import math
from torch import nn
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        data = torch.empty((out_features, in_features), device=device, dtype=dtype)
        self.weight = nn.Parameter(data)
        nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std=std,
            a=-3.0*std,
            b=3.0*std
        )

    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        data = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        self.weight = nn.Parameter(data)
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )
    def forward(self, x):
        return self.weight[x]

def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype) 
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        data = torch.ones((d_model, ), device=device, dtype=dtype)
        self.weight = nn.Parameter(data)
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(
            torch.mean(x**2, dim=-1, keepdim=True) + self.eps
        )
        normalized = x / rms
        result = normalized * self.weight
        return result.to(in_dtype)

def softmax(x, dim):
    max_values = torch.amax(x, dim=dim, keepdim=True)
    shifted = x - max_values
    esum = torch.sum(torch.exp(shifted), dim=dim, keepdim=True)
    result = torch.exp(shifted) / esum
    return result

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attention_weights = softmax(scores, dim=-1)
    return attention_weights @ V

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        exponents = (
            torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        )
        frequencies = theta ** (-exponents)
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32
        )
        angles = torch.outer(positions, frequencies)
        self.register_buffer(
            "cos_cached",
            torch.cos(angles),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            torch.sin(angles),
            persistent=False,
        )
    def forward(self, x, token_positions):
        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        output_even = x_even * cos - x_odd * sin
        output_odd = x_even * sin + x_odd * cos
        return torch.stack(
            (output_even, output_odd),
            dim=-1
        ).flatten(-2)
