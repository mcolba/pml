"""
Set Transformer components for permutation-invariant set encoding.

Based on "Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks" (Lee et al., 2019).

Components
----------
MultiheadAttention : standard scaled dot-product multi-head attention
MAB  : Multihead Attention Block (attention + residual + FF + LayerNorm)
ISAB : Induced Set Attention Block (O(mn) instead of O(n²))
PMA  : Pooling by Multihead Attention (fixed-size output from variable set)
SetEncoder : end-to-end  LinearProj → ISAB → PMA → LinearProj
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """Scaled dot-product multi-head attention with optional key mask."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Parameters
        ----------
        Q : Tensor (B, n_q, d_model)
        K : Tensor (B, n_k, d_model)
        V : Tensor (B, n_k, d_model)
        mask : Tensor (B, n_k) bool — True where element exists
        """
        B, n_q, _ = Q.shape
        n_k = K.shape[1]

        q = self.W_q(Q).reshape(B, n_q, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(K).reshape(B, n_k, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(V).reshape(B, n_k, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            # mask: (B, n_k) → (B, 1, 1, n_k) to broadcast over heads & queries
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # If an entire row is masked the softmax produces nan → replace with 0
        attn = attn.nan_to_num(0.0)

        out = torch.matmul(attn, v)  # (B, num_heads, n_q, d_head)
        out = out.transpose(1, 2).reshape(B, n_q, self.d_model)
        return self.W_o(out)


class MAB(nn.Module):
    """Multihead Attention Block.

    MAB(X, Y) = LN(H + FF(H))  where  H = LN(X + MHA(X, Y, Y))
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mha = MultiheadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, X, Y, mask=None):
        """
        Parameters
        ----------
        X : Tensor (B, n_x, d_model) — queries
        Y : Tensor (B, n_y, d_model) — keys / values
        mask : Tensor (B, n_y) bool — mask for Y
        """
        H = self.ln1(X + self.mha(X, Y, Y, mask=mask))
        return self.ln2(H + self.ff(H))


class ISAB(nn.Module):
    """Induced Set Attention Block.

    Uses *m* learnable inducing points to reduce self-attention cost
    from O(n²) to O(mn).

    ISAB(X) = MAB(X, H)  where  H = MAB(I, X)
    """

    def __init__(self, d_model: int, num_heads: int, num_inducing: int):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, d_model))
        self.mab1 = MAB(d_model, num_heads)  # inducing points attend to input
        self.mab2 = MAB(d_model, num_heads)  # input attends to inducing points

    def forward(self, X, mask=None):
        """
        Parameters
        ----------
        X : Tensor (B, N, d_model)
        mask : Tensor (B, N) bool

        Returns
        -------
        Tensor (B, N, d_model)
        """
        B = X.shape[0]
        I = self.inducing_points.unsqueeze(0).expand(B, -1, -1)
        H = self.mab1(I, X, mask=mask)  # (B, m, d_model)
        return self.mab2(X, H)  # (B, N, d_model) — H has no masked slots


class PMA(nn.Module):
    """Pooling by Multihead Attention.

    PMA(Z) = MAB(S, Z)  where S are *k* learnable seed vectors.
    Produces a fixed-size output regardless of input set size.
    """

    def __init__(self, d_model: int, num_heads: int, num_seeds: int = 1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_seeds, d_model))
        self.mab = MAB(d_model, num_heads)

    def forward(self, Z, mask=None):
        """
        Parameters
        ----------
        Z : Tensor (B, N, d_model)
        mask : Tensor (B, N) bool

        Returns
        -------
        Tensor (B, num_seeds, d_model)
        """
        B = Z.shape[0]
        S = self.seeds.unsqueeze(0).expand(B, -1, -1)
        return self.mab(S, Z, mask=mask)


class SetEncoder(nn.Module):
    """Encode a variable-size set into a fixed-size summary vector.

    Architecture::

        x (B, N, input_dim)
          → LinearProj → (B, N, d_model)
          → ISAB       → (B, N, d_model)
          → PMA        → (B, 1,  d_model)
          → LinearProj → (B, output_dim)

    Parameters
    ----------
    input_dim : int
        Dimensionality of each set element.
    d_model : int
        Internal dimension of the attention layers.
    output_dim : int
        Dimensionality of the output summary vector.
    num_heads : int
        Number of attention heads (must divide d_model).
    num_inducing : int
        Number of inducing points in the ISAB.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: int,
        num_heads: int = 4,
        num_inducing: int = 8,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.isab = ISAB(d_model, num_heads, num_inducing)
        self.pma = PMA(d_model, num_heads, num_seeds=1)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x : Tensor (B, N, input_dim)
            Set of elements (padded to N_max).
        mask : Tensor (B, N) bool, optional
            True where element exists.  If None, all elements are used.

        Returns
        -------
        Tensor (B, output_dim)
            Permutation-invariant summary of the set.
        """
        h = self.input_proj(x)  # (B, N, d_model)
        h = self.isab(h, mask=mask)  # (B, N, d_model)
        h = self.pma(h, mask=mask)  # (B, 1, d_model)
        h = h.squeeze(-2)  # (B, d_model)
        return self.output_proj(h)  # (B, output_dim)


class MeanPoolSetEncoder(nn.Module):
    """Encode a variable-size set into a fixed-size summary via mean pooling.

    Simpler alternative to attention-based SetEncoder, inspired by the
    Neural Statistician (Edwards & Storkey, 2017).

    Architecture::

        x (B, N, input_dim)
          → LinearProj   → (B, N, d_model)
          → ReLU
          → LinearProj   → (B, N, d_model)   [element-wise encoding]
          → MaskedMean   → (B, d_model)       [mean pooling over set]
          → LinearProj   → (B, d_model)
          → ReLU
          → LinearProj   → (B, output_dim)   [post-pool processing]

    Parameters
    ----------
    input_dim : int
        Dimensionality of each set element.
    d_model : int
        Internal hidden dimension.
    output_dim : int
        Dimensionality of the output summary vector.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: int = 5,
    ):
        super().__init__()
        self.output_dim = output_dim
        # Pre-pool: element-wise encoding
        self.pre_pool = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # Post-pool: process the aggregated representation
        self.post_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x : Tensor (B, N, input_dim)
            Set of elements (padded to N_max).
        mask : Tensor (B, N) bool, optional
            True where element exists.  If None, all elements are used.

        Returns
        -------
        Tensor (B, output_dim)
            Permutation-invariant summary of the set.
        """
        B, N, _ = x.shape

        # Element-wise encoding
        h = self.pre_pool(x)  # (B, N, d_model)

        # Mean pooling with mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            h = h * mask_expanded
            counts = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
            h = h.sum(dim=1) / counts  # (B, d_model)
        else:
            h = h.mean(dim=1)  # (B, d_model)

        # Post-pool processing
        return self.post_pool(h)  # (B, output_dim)
