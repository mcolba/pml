"""Mean-pooling set encoder utilities used by the hierarchical VAEs."""

import torch
import torch.nn as nn


class MeanPoolSetEncoder(nn.Module):
    """Encode a variable-size set into a fixed-size summary via masked mean pooling."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pre_pool = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.post_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return a permutation-invariant summary of the input set."""
        h = self.pre_pool(x)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            h = h * mask_expanded
            counts = mask_expanded.sum(dim=1).clamp(min=1.0)
            h = h.sum(dim=1) / counts
        else:
            h = h.mean(dim=1)

        return self.post_pool(h)
