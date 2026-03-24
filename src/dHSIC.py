import math
import warnings

import torch
from pyro import poutine

# Global variable for debugging pen.grad
_debug_pen = None

# class TraceELBOdHSIC(Trace_ELBO):
#     """Wrapper around Trace_ELBO that adds HSIC penalisation."""

#     def __init__(
#         self,
#         *,
#         hsic_lambda: float = 1.0,
#     ):
#         super().__init__()
#         self.hsic_lambda = float(hsic_lambda)

#     def loss_and_grads(self, model, guide, *args, **kwargs):
#         total = 0.0
#         for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
#             particle_loss = -(model_trace.log_prob_sum() - guide_trace.log_prob_sum())

#             c = args[1]
#             if "latent" not in guide_trace.nodes:
#                 raise KeyError("Could not find guide sample site named 'latent'.")

#             z = guide_trace.nodes["latent"]["value"]

#             n = z.shape[0]
#             if n < 32:
#                 warnings.warn(
#                     f"Mini-batch size {n} is likely too small for stable HSIC estimates; "
#                     f"consider batch_size >= 32.",
#                     RuntimeWarning,
#                 )

#             particle_loss = particle_loss + self.hsic_lambda * hsic_std(z, c)

#             total = total + particle_loss

#         num_particles = getattr(self, "num_particles", 1)
#         return total / float(num_particles)


def make_elbo_hsic(hsic_lambda):
    def _loss(model, guide, *args, **kwargs):
        return elbo_hsic(model, guide, *args, hsic_lambda=hsic_lambda, **kwargs)

    return _loss


def elbo_hsic(
    model,
    guide,
    *args,
    z_site: str = "latent",
    c_pos: int = 1,
    hsic_lambda: float = 1.0,
    hsic_min_batch_warn: int = 32,
    hsic_scale_by_batch: bool = True,
    **kwargs,
):
    """
    A simple ELBO objective (minimize negative ELBO) with HSIC penalty between z and c.
    Returns:
      torch scalar loss (requires grad).
    """
    # trace guide
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    # trace model replayed with guide samples
    model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(
        *args, **kwargs
    )

    # negative ELBO (minimize)
    neg_elbo = -(model_trace.log_prob_sum() - guide_trace.log_prob_sum())

    # extract z and c
    if z_site not in guide_trace.nodes:
        raise KeyError(f"Guide missing sample site '{z_site}'")
    z = guide_trace.nodes[z_site]["value"]

    if len(args) <= c_pos:
        raise ValueError(
            f"Expected c at position args[{c_pos}] but got only {len(args)} args."
        )
    c = args[c_pos]

    # reshape to [n, ...] (any D allowed); HSIC implementation flattens internally
    n = z.shape[0]
    if n < hsic_min_batch_warn:
        warnings.warn(
            f"Mini-batch size {n} is likely too small for stable HSIC estimates; "
            f"consider batch_size >= {hsic_min_batch_warn}.",
            RuntimeWarning,
        )

    # pen = hsic_std(z, c)
    pen = hsic(z, c)
    if hsic_scale_by_batch:
        pen = pen * n

    global _debug_pen
    _debug_pen = pen
    if pen.requires_grad:
        pen.retain_grad()

    return neg_elbo + hsic_lambda * pen


def bandwidth_from_d(d: int) -> float:
    # same analytic rule as HCV.py: 1/(2*gz^2)
    d = float(d)
    gz = 2.0 * math.exp(math.lgamma(0.5 * (d + 1.0)) - math.lgamma(0.5 * d))
    return 1.0 / (2.0 * (gz**2) + 1e-12)


def K_rbf(x1: torch.Tensor, x2: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    x1: [n1, ...], x2: [n2, ...]  ->  K: [n1, n2]
    RBF kernel on flattened samples.
    """
    x1 = x1.reshape(x1.shape[0], -1).contiguous()
    x2 = x2.reshape(x2.shape[0], -1).contiguous()

    x1_norm = (x1**2).sum(dim=1, keepdim=True)  # [n1, 1]
    x2_norm = (x2**2).sum(dim=1, keepdim=True).t()  # [1, n2]
    dist2 = x1_norm + x2_norm - 2.0 * (x1 @ x2.t())  # [n1, n2]
    dist2 = torch.clamp(dist2, min=0.0)
    return torch.exp(-gamma * dist2)


def hsic(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Biased V-statistic HSIC in the same reduce_mean form as HCV.py.
    Works for z:[n,...], c:[n,...] (any D), same batch n.
    Returns HSIC (no sqrt).
    """
    if z.shape[0] != c.shape[0]:
        raise ValueError(
            f"Batch mismatch: z has {z.shape[0]} rows, c has {c.shape[0]} rows."
        )

    d_z = int(torch.tensor(z.shape[1:]).prod().item()) if z.dim() > 1 else 1
    d_c = int(torch.tensor(c.shape[1:]).prod().item()) if c.dim() > 1 else 1

    zz = K_rbf(z, z, gamma=bandwidth_from_d(d_z))
    cc = K_rbf(c, c, gamma=bandwidth_from_d(d_c))

    hs = (zz * cc).mean()
    hs = hs + zz.mean() * cc.mean()
    hs = hs - 2.0 * ((zz.mean(dim=1) * cc.mean(dim=1)).mean())
    return torch.clamp(hs, min=0.0)


def hsic_std(z: torch.Tensor, c: torch.Tensor):
    """Normalised HSIC."""
    return hsic(z, c) / torch.sqrt(hsic(z, z) * hsic(c, c))


def _test_hsic_1d_and_3d(
    n: int = 512, a: float = 1.0, noise: float = 0.2, device: str = "cpu"
):
    torch.manual_seed(42)
    device = "cpu"
    n = 256
    p = 0.3

    Y = torch.randn(n, 1, device=device)
    eps = torch.randn(n, 1, device=device)
    X = p * (Y**2) + (1 - p) * eps

    hs = hsic(X, Y).item()
    nhs = hsic_std(X, Y).item()

    print("[1D] HSIC(X,epsilon) =", hs)
    print("[1D] Normalised HSIC(X,Y) =", nhs)

    # --- 3D test: reshape to [n, 2, 2, 2] without changing samplewise dependence ---
    # Make a 3D tensor by repeating features; HSIC should still be larger for dependent pairs.
    X3 = X.repeat(1, 8).reshape(n, 2, 2, 2)
    Y3 = Y.repeat(1, 8).reshape(n, 2, 2, 2)
    E3 = eps.repeat(1, 8).reshape(n, 2, 2, 2)

    hs_xy_3d = hsic(X3, Y3).item()
    hs_xe_3d = hsic(Y3, E3).item()

    print("[3D] HSIC(X3,Y3)      =", hs_xy_3d)
    print("[3D] HSIC(Y3,epsilon) =", hs_xe_3d)


if __name__ == "__main__":
    _test_hsic_1d_and_3d()
