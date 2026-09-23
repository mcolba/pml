"""Shared building blocks for hierarchical VAE variants."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.vae.utils.set_transformer import MeanPoolSetEncoder


class ParentEncoder(nn.Module):
    """q(z1 | x1) — infer global latent from Index only."""

    def __init__(self, x1_dim: int, z1_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(x1_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z1_dim)
        self.fc_scale = nn.Linear(hidden_dim, z1_dim)

    def forward(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = x1.reshape(-1, self.fc1.in_features)
        hidden = F.softplus(self.fc1(x1))
        z1_loc = self.fc_loc(hidden)
        z1_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z1_loc, z1_scale


class ParentDecoder(nn.Module):
    """p(x1 | z1) — reconstruct Index from global latent."""

    def __init__(self, x1_dim: int, z1_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(z1_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, x1_dim)
        self.log_x_scale = nn.Parameter(torch.zeros(x1_dim))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.softplus(self.fc1(z))
        x_loc = self.fc_loc(hidden)
        return x_loc, self.log_x_scale.expand_as(x_loc)


class ChildPrior(nn.Module):
    """p(z2 | z1, c2) — learned conditional prior for child latent."""

    def __init__(self, z1_dim: int, c2_dim: int, z2_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(z1_dim + c2_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z2_dim)
        self.fc_scale = nn.Linear(hidden_dim, z2_dim)

    def forward(
        self, z1: torch.Tensor, c2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([z1, c2], dim=-1)
        hidden = F.softplus(self.fc1(inp))
        z2_loc = self.fc_loc(hidden)
        z2_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z2_loc, z2_scale


class ChildEncoder(nn.Module):
    """q(z2 | x2, z1, c2) — shared child encoder across all names."""

    def __init__(
        self, x2_dim: int, z1_dim: int, c2_dim: int, z2_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(x2_dim + z1_dim + c2_dim, hidden_dim)
        self.fc_loc = nn.Linear(hidden_dim, z2_dim)
        self.fc_scale = nn.Linear(hidden_dim, z2_dim)

    def forward(
        self, x2: torch.Tensor, z1: torch.Tensor, c2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x2, z1, c2], dim=-1)
        hidden = F.softplus(self.fc1(inp))
        z2_loc = self.fc_loc(hidden)
        z2_scale = F.softplus(self.fc_scale(hidden)) + 1e-4
        return z2_loc, z2_scale


@dataclass(frozen=True)
class ObservedChildren:
    """Observed child context: x2 (B, N, D2), c2 (B, N, C2), mask (B, N)."""

    x2: torch.Tensor
    c2: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.x2.ndim != 3 or self.c2.ndim != 3 or self.mask.ndim != 2:
            raise ValueError(
                "observed children require x2 and c2 of shape (B, N, D) "
                "and mask of shape (B, N)"
            )
        if (
            self.x2.shape[:2] != self.c2.shape[:2]
            or self.mask.shape != self.x2.shape[:2]
        ):
            raise ValueError(
                "x2_set, c2_set, and child_mask must match on batch and set axes"
            )
        if self.x2.device != self.c2.device or self.x2.device != self.mask.device:
            raise ValueError("observed child tensors must be on the same device")
        object.__setattr__(self, "mask", self.mask.bool())


class ParentEncoderInput(nn.Module):
    """Compose x1 with paired `(x2, c2)` context or a zero no-context summary."""

    def __init__(
        self,
        *,
        x1_dim: int,
        summary_encoder: nn.Module | None = None,
        summary_dim: int = 0,
        x2_dim: int | None = None,
        c2_dim: int | None = None,
    ):
        super().__init__()
        if summary_encoder is None and summary_dim != 0:
            raise ValueError(
                "summary_dim must be 0 when no summary_encoder is provided"
            )
        if summary_encoder is not None and summary_dim <= 0:
            raise ValueError(
                "summary_dim must be positive when summary_encoder is provided"
            )
        if (x2_dim is None) != (c2_dim is None):
            raise ValueError("x2_dim and c2_dim must be provided together")
        if summary_encoder is None and x2_dim is not None:
            raise ValueError("child feature dimensions require a summary_encoder")

        self.summary_encoder = summary_encoder
        self.summary_dim = summary_dim
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.c2_dim = c2_dim
        self.encoder_input_dim = x1_dim + self.summary_dim
        if x2_dim is not None and c2_dim is not None:
            self.bind_child_dims(x2_dim, c2_dim)

    def bind_child_dims(self, x2_dim: int, c2_dim: int) -> None:
        """Set and validate the paired child widths expected by the summary encoder."""
        if self.summary_encoder is None:
            return
        if self.x2_dim is not None and self.x2_dim != x2_dim:
            raise ValueError("x2_dim does not match the parent summary contract")
        if self.c2_dim is not None and self.c2_dim != c2_dim:
            raise ValueError("c2_dim does not match the parent summary contract")
        expected_input_dim = x2_dim + c2_dim
        declared_input_dim = getattr(
            self.summary_encoder, "input_dim", expected_input_dim
        )
        if declared_input_dim != expected_input_dim:
            raise ValueError("set_encoder input_dim must equal x2_dim + c2_dim")
        self.x2_dim = x2_dim
        self.c2_dim = c2_dim

    @property
    def uses_summary(self) -> bool:
        """Return whether the parent encoder consumes a child-set summary."""
        return self.summary_encoder is not None

    def forward(
        self,
        x1: torch.Tensor,
        x2_set: torch.Tensor | None = None,
        child_mask: torch.Tensor | None = None,
        c2_set: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append a paired child summary; context requires x2, c2, and mask together."""
        if x1.ndim != 2 or x1.shape[1] != self.x1_dim:
            raise ValueError("x1 must have shape (B, x1_dim)")

        parts = (x2_set, c2_set, child_mask)
        if all(part is None for part in parts):
            observed = None
        elif any(part is None for part in parts):
            raise ValueError("x2_set, c2_set, and child_mask must be provided together")
        else:
            observed = ObservedChildren(x2_set, c2_set, child_mask)

        if self.summary_encoder is None:
            if observed is not None:
                raise ValueError(
                    "this model was trained without observed-child context"
                )
            return x1

        if observed is None:
            summary = x1.new_zeros(x1.shape[0], self.summary_dim)
        else:
            if observed.x2.shape[0] != x1.shape[0] or observed.x2.device != x1.device:
                raise ValueError("observed children must match x1 on batch and device")
            if self.x2_dim is not None and observed.x2.shape[-1] != self.x2_dim:
                raise ValueError("x2_set must have x2_dim features")
            if self.c2_dim is not None and observed.c2.shape[-1] != self.c2_dim:
                raise ValueError("c2_set must have c2_dim features")
            summary = x1.new_zeros(x1.shape[0], self.summary_dim)
            active_rows = observed.mask.any(dim=1)
            if active_rows.any():
                summary_input = torch.cat([observed.x2, observed.c2], dim=-1)
                summary_input = torch.where(
                    observed.mask.unsqueeze(-1),
                    summary_input,
                    torch.zeros_like(summary_input),
                )
                active_summary = self.summary_encoder(
                    summary_input[active_rows], mask=observed.mask[active_rows]
                )
                if active_summary.shape != (int(active_rows.sum()), self.summary_dim):
                    raise ValueError(
                        "summary encoder must return shape (B, summary_dim)"
                    )
                summary = summary.index_copy(
                    0, active_rows.nonzero().flatten(), active_summary
                )
        return torch.cat([x1, summary], dim=-1)


def build_full_posterior_encoder_input(
    *,
    x1_dim: int,
    x2_dim: int,
    c2_dim: int,
    hidden_dim: int,
) -> ParentEncoderInput:
    """Create a mean-pool parent encoder input composer for paired `(x2, c2)` sets."""
    summary_input_dim = x2_dim + c2_dim
    return ParentEncoderInput(
        x1_dim=x1_dim,
        x2_dim=x2_dim,
        c2_dim=c2_dim,
        summary_encoder=MeanPoolSetEncoder(
            input_dim=summary_input_dim,
            d_model=hidden_dim,
            output_dim=hidden_dim,
        ),
        summary_dim=hidden_dim,
    )
