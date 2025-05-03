from jaxtyping import Float
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
import math


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        w: Float[Tensor, "out_features in_features"] = torch.empty(
            out_features, in_features, device=device, dtype=dtype
        )
        std = math.sqrt(2 / (in_features + out_features))
        w = nn.init.trunc_normal_(w, std=std, a=-3*std, b=3*std)
        self.layer = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.layer, "... d_in, d_out d_in -> ... d_out")
