"""Mean-pooling set encoder utilities used by the hierarchical VAEs."""

from collections.abc import Sequence

import torch
from torch import nn

from src.vae.utils.nn import build_mlp


class MeanPoolSetEncoder(nn.Module):
    """Encode a set using per-member features, masked means, and valid counts.

    ``hidden_dim`` specifies the per-member ELU hidden layers (default [64, 64]);
    ``d_model`` is the pooled embedding width and ``output_dim`` the summary width.
    The nonlinear post-pooling network receives the mean embedding and the raw
    number of valid members. Empty sets return a zero summary.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: int = 5,
        hidden_dim: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        if input_dim <= 0 or d_model <= 0 or output_dim <= 0:
            raise ValueError("input_dim, d_model, and output_dim must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pre_pool = nn.Sequential(
            build_mlp(input_dim, hidden_dim),
            nn.Linear(hidden_dim[-1], d_model),
        )
        self.post_pool = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Summarize ``x`` (B, N, input_dim) with an optional mask (B, N).

        Return (B, output_dim); masks exclude padded members from both the mean
        and count. Masked input values may be nonfinite, and empty rows map to zero.
        """
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError("x must have shape (B, N, input_dim)")
        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError("mask must have shape (B, N) matching x")
            if mask.device != x.device:
                raise ValueError("mask and x must be on the same device")
            mask_expanded = mask.bool().unsqueeze(-1)
            x = torch.where(mask_expanded, x, torch.zeros_like(x))

        h = self.pre_pool(x)
        if mask is not None:
            h = torch.where(mask_expanded, h, torch.zeros_like(h))
            counts = mask_expanded.sum(dim=1).to(dtype=h.dtype)
        else:
            counts = h.new_full((x.shape[0], 1), x.shape[1])

        pooled = h.sum(dim=1) / counts.clamp(min=1)
        summary = self.post_pool(torch.cat([pooled, counts], dim=-1))
        return torch.where(counts > 0, summary, torch.zeros_like(summary))
