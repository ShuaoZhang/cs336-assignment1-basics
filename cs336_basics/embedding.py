from jaxtyping import Float
import torch
import torch.nn as nn
from torch import Tensor


class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        w: Float[Tensor, "embedding_dim num_embeddings"] = torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )
        w = nn.init.trunc_normal_(w, std=1, a=-3, b=3)
        self.layer = nn.Parameter(w)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.layer[token_ids]
