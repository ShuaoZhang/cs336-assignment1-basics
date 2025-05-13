from jaxtyping import Float
import torch
import torch.nn as nn
from torch import Tensor
from einops import reduce, einsum, rearrange
import math


class MyRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        g: Float[Tensor, "d_model"] = torch.ones(d_model, device=device, dtype=dtype)
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
        result = x * self.g / rms

        return result.to(in_dtype)


class MySiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * x


class MySwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        # d_ff = d_model * 8 / 3

        w1: Float[Tensor, "d_ff d_model"] = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        w2: Float[Tensor, "d_model d_ff"] = torch.empty(d_model, d_ff, device=device, dtype=dtype)
        w3: Float[Tensor, "d_ff d_model"] = torch.empty(d_ff, d_model, device=device, dtype=dtype)

        # TODO: how to initialize?
        w1 = nn.init.trunc_normal_(w1)
        w2 = nn.init.trunc_normal_(w2)
        w3 = nn.init.trunc_normal_(w3)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)
        self.silu = MySiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w3x = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        w1x = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        t = self.silu(w1x) * w3x
        return einsum(t, self.w2, "... d_ff, d_model d_ff -> ... d_model")


class MyRoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        half_dk = d_k // 2
        theta_k: Float[Tensor, "half_dk"] = 1.0 / (theta ** torch.arange(2, d_k + 1, 2).to(device=device) / d_k)
        positions: Float[Tensor, "max_seq_len"] = torch.arange(0, max_seq_len).to(device=device)

        theta_ik = einsum(positions, theta_k, "max_seq_len, half_dk -> max_seq_len half_dk")

        self.register_buffer("cos_cache", torch.cos(theta_ik), persistent=False)
        self.register_buffer("sin_cache", torch.sin(theta_ik), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_shaped = rearrange(x, "... seq_len (d1 r) -> ... seq_len d1 r", r=2)
        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]

        x1, x2 = x_shaped[..., 0], x_shaped[..., 1]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        out = rearrange(torch.stack([x1_rot, x2_rot], dim=-1), "b s d1 r -> b s (d1 r)")
        return out


def scaled_dot_product_attention(Q, K, V, mask) -> Tensor:
    d_k = Q.shape[-1]

    t1 = einsum(Q, K, "... n dk, ... m dk -> ... n m") / math.sqrt(d_k)
    t2 = t1.masked_fill(~mask, float("-inf"))
    t3 = softmax(t2, -1)
    return einsum(t3, V, "... n m, ... m dv -> ... n dv")


def softmax(x: Tensor, i: int) -> Tensor:
    t1 = x - torch.max(x, dim=i, keepdim=True).values
    t2 = torch.exp(t1)
    return t2 / torch.sum(t2, dim=i, keepdim=True)
