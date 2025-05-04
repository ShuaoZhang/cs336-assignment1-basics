from jaxtyping import Float
import torch
import torch.nn as nn
from torch import Tensor
from einops import reduce, einsum


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
