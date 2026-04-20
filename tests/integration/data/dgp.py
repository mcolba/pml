from inspect import signature
from typing import Callable, Optional, Tuple

import numpy as np

LatentSampler = Callable[..., np.ndarray]

DEFAULT_HIERARCHICAL_LATENT_COV = np.array(
    [
        [1.0, 0.20],
        [0.20, 0.5],
    ],
    dtype=float,
)


def _validate_covariance_matrix(
    matrix: np.ndarray,
    z_dim: int,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (z_dim, z_dim):
        raise ValueError(f"Covariance matrix must have shape ({z_dim}, {z_dim}).")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("Covariance matrix must be symmetric.")
    if np.any(np.linalg.eigvalsh(matrix) < -1e-10):
        raise ValueError(" must be positive semidefinite.")
    return matrix


def _call_latent_sampler(
    latent_sampler: LatentSampler,
    n_samples: int,
    z_dim: int,
    rng: np.random.Generator,
    cov: np.ndarray,
) -> np.ndarray:
    sampler_params = signature(latent_sampler).parameters

    if "cov" in sampler_params:
        return latent_sampler(n_samples, z_dim, rng, cov=cov)
    if "scale" in sampler_params:
        return latent_sampler(n_samples, z_dim, rng, scale=cov)

    return latent_sampler(n_samples, z_dim, rng)


def sample_gaussian_latents(
    n_samples: int,
    z_dim: int,
    rng: np.random.Generator,
    cov: Optional[np.ndarray] = None,
    mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample z ~ N(mean, cov).
    """
    if mean is None:
        mean = np.zeros(z_dim)
    else:
        mean = np.asarray(mean, dtype=float)
        if mean.shape != (z_dim,):
            raise ValueError(f"mean must have shape ({z_dim},).")

    if cov is None:
        cov = np.eye(z_dim)

    cov = _validate_covariance_matrix(cov, z_dim)

    return rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)


def sample_student_t_latents(
    n_samples: int,
    z_dim: int,
    rng: np.random.Generator,
    df: float = 5.0,
    scale: Optional[np.ndarray] = None,
    mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample multivariate Student-t:
        z = mean + g / sqrt(u / df)
    where g ~ N(0, scale), u ~ ChiSq(df).
    """
    if df <= 0:
        raise ValueError("df must be positive.")

    if mean is None:
        mean = np.zeros(z_dim)
    else:
        mean = np.asarray(mean, dtype=float)
        if mean.shape != (z_dim,):
            raise ValueError(f"mean must have shape ({z_dim},).")

    if scale is None:
        scale = np.eye(z_dim)

    scale = _validate_covariance_matrix(scale, z_dim)

    g = rng.multivariate_normal(mean=np.zeros(z_dim), cov=scale, size=n_samples)
    u = rng.chisquare(df=df, size=n_samples)
    return mean + g / np.sqrt(u / df)[:, None]


def sample_bimodal_latents(
    n_samples: int,
    z_dim: int,
    rng: np.random.Generator,
    means: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    weights: Tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """
    Sample from a 2-component Gaussian mixture.
    """
    if means is None:
        if z_dim != 2:
            raise ValueError("Default bimodal means are only defined for z_dim=2.")
        means = np.array(
            [
                [-1.5, 1.5],
                [1.5, -1.5],
            ],
            dtype=float,
        )
    else:
        means = np.asarray(means, dtype=float)
        if means.shape != (2, z_dim):
            raise ValueError(f"means must have shape (2, {z_dim}).")

    if cov is None:
        cov = np.eye(z_dim)

    cov = _validate_covariance_matrix(cov, z_dim)

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (2,) or np.any(weights < 0):
        raise ValueError("weights must be a length-2 nonnegative tuple.")
    weights = weights / weights.sum()

    comp = rng.choice(2, size=n_samples, p=weights)

    z0 = rng.multivariate_normal(mean=means[0], cov=cov, size=n_samples)
    z1 = rng.multivariate_normal(mean=means[1], cov=cov, size=n_samples)

    z = np.where(comp[:, None] == 0, z0, z1)

    return z


def generate_conditional_hierarchical_sets(
    n_datasets: int,
    latent_sampler: LatentSampler,
    n_conditions: int = 50,
    x_dim: int = 10,
    z_dim: int = 2,
    sigma_c: float = 1.0,
    sigma_x1: float = 0.10,
    sigma_z2: float = 0.10,
    sigma_x2: float = 0.20,
    rotation_scale: float = 0.6,
    stretch_strength: float = 0.35,
    condition_shift: float = 0.15,
    cov: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate datasets from the model

        z1 ~ latent_sampler(...)
        x1 | z1 ~ N(A z1, sigma_x1^2 I)
        c_j ~ N(0, sigma_c^2)
        z2_j | z1, c_j ~ N(M(c_j) z1 + b(c_j), sigma_z2^2 I_2)
        x2_j | z2_j ~ N(B z2_j, sigma_x2^2 I)

    Parameters
    ----------
    cov : optional, shape (z_dim, z_dim)
        Covariance-like matrix forwarded to `latent_sampler` when it accepts
        a `cov` or `scale` keyword. If omitted,
        `DEFAULT_HIERARCHICAL_LATENT_COV` is used.

    Returns
    -------
    x1 : (n_datasets, 10)
    x2 : (n_datasets, n_conditions, 10)
    c2 : (n_datasets, n_conditions, 1)
    """
    if x_dim != 10:
        raise ValueError("This implementation is set up for x_dim=10.")
    if z_dim != 2:
        raise ValueError("This implementation is set up for z_dim=2.")

    rng = np.random.default_rng(seed)

    latent_cov = DEFAULT_HIERARCHICAL_LATENT_COV if cov is None else cov
    latent_cov = _validate_covariance_matrix(latent_cov, z_dim)

    A = np.array(
        [
            [0.32, 0.50],
            [0.32, 0.46],
            [0.32, 0.39],
            [0.32, 0.24],
            [0.32, 0.09],
            [0.32, -0.20],
            [0.32, -0.35],
            [0.32, -0.57],
            [0.32, -1.31],
            [0.32, -1.67],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [0.42, 0.58],
            [0.42, 0.53],
            [0.42, 0.45],
            [0.42, 0.28],
            [0.42, 0.11],
            [0.42, -0.24],
            [0.42, -0.40],
            [0.42, -0.66],
            [0.42, -1.50],
            [0.42, -1.92],
        ],
        dtype=float,
    )

    z1 = _call_latent_sampler(
        latent_sampler=latent_sampler,
        n_samples=n_datasets,
        z_dim=z_dim,
        rng=rng,
        cov=latent_cov,
    )
    if z1.shape != (n_datasets, z_dim):
        raise ValueError(
            f"latent_sampler must return shape ({n_datasets}, {z_dim}), got {z1.shape}."
        )

    x1_mean = z1 @ A.T
    x1 = x1_mean + sigma_x1 * rng.normal(size=x1_mean.shape)

    c2 = sigma_c * rng.normal(size=(n_datasets, n_conditions, 1))
    for i in range(n_datasets):
        rng.shuffle(c2[i], axis=0)

    condition_values = c2.squeeze(-1)

    angle = rotation_scale * condition_values
    t = np.tanh(condition_values)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    s1 = 1.0 + stretch_strength * t
    s2 = 1.0 - stretch_strength * t

    R = np.stack(
        [
            np.stack([cos_a, -sin_a], axis=-1),
            np.stack([sin_a, cos_a], axis=-1),
        ],
        axis=-2,
    )

    stretch_factors = np.stack([s1, s2], axis=-1)
    S = stretch_factors[..., :, None] * np.eye(2, dtype=float)

    M = R @ S

    z2_linear = (M @ z1[:, None, :, None]).squeeze(-1)
    b = condition_shift * np.stack([condition_values, -condition_values], axis=-1)
    z2_mean = z2_linear + b
    z2 = z2_mean + sigma_z2 * rng.normal(size=z2_mean.shape)

    x2_mean = z2 @ B.T
    x2 = x2_mean + sigma_x2 * rng.normal(size=x2_mean.shape)

    return x1, x2, c2




if __name__ == "__main__":
    pass
