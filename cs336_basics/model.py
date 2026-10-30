import torch
import math
from torch import nn
from einops import rearrange
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
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model, 
        num_heads, 
        use_rope=False,
        max_seq_len=None, 
        theta=10000.0, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = rearrange(
            q,
            "... seq (h d) -> ... h seq d",
            h=self.num_heads,
        )
        k = rearrange(
            k,
            "... seq (h d) -> ... h seq d",
            h=self.num_heads,
        )
        v = rearrange(
            v,
            "... seq (h d) -> ... h seq d",
            h=self.num_heads,
        )

        if self.rope is not None:
            if token_positions is None:
                sequence_length = x.shape[-2]
                token_positions = torch.arange(
                    sequence_length,
                    device=x.device,
                )
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        sequence_length = x.shape[-2]
        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            )
        )
        attention_output = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=causal_mask,
        )

        attention_output = rearrange(
            attention_output,
            "... h seq d -> ... seq (h d)"
        )
        return self.output_proj(attention_output)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=10000.00, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
    def forward(self, x, token_positions=None):
        normalized_x = self.ln1(x)
        attention_output = self.attn(
            normalized_x, 
            token_positions=token_positions,
        )
        h = x + attention_output
        normalized_h = self.ln2(h)
        ffn_output = self.ffn(normalized_h)
        y = h + ffn_output
        return y
class TransformerLM(nn.Module):
    def __init__(self, 
        vocab_size,
        context_length, 
        d_model, 
        num_layers, 
        num_heads, 
        d_ff, 
        rope_theta,
        device=None,
        dtype=None,
        ):
        super().__init__()
        self.token_embeddings = Embedding(
            num_embeddings = vocab_size,
            embedding_dim = d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )
    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
def CrossEntropy(inputs, targets):
    max_values = torch.amax(inputs, dim=-1, keepdim=True)
    shifted = inputs - max_values
    exp_values = torch.exp(shifted)
    exp_sum = torch.sum(exp_values, dim=-1, keepdim=True)
    log_probs = shifted - torch.log(exp_sum)
    target_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    return -target_log_probs.mean()
def GradientClipping(parameters, max_l2_norm):
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_squared_norm = sum(
        (grad ** 2).sum()
        for grad in grads
    )
    total_norm = torch.sqrt(total_squared_norm)
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(clip_coef)
    return None

def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5 * (1 + math.cos(math.pi*((it - warmup_iters)/(cosine_cycle_iters-warmup_iters))))*(max_learning_rate-min_learning_rate)
    else:
        lr=min_learning_rate
    return lr
