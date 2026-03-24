from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from statsmodels.graphics.tsaplots import plot_acf

from src.pca import blend_3_colors, create_triangle_legend_ternary, pca_cps_2

plt.style.use("ggplot")


def show_overlaid_reconstruction(x, xhat, tenors, output_folder: Path = None):
    """Show original and reconstructed curves overlaid."""
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

    for i, ax_i in enumerate(ax.reshape(-1)):
        ax_i.plot(
            tenors,
            x[i],
            "o-",
            linewidth=2,
            markersize=6,
            label="Original",
            alpha=0.7,
        )
        ax_i.plot(
            tenors,
            xhat[i],
            "s--",
            linewidth=2,
            markersize=6,
            label="Reconstructed",
            alpha=0.7,
        )
        ax_i.tick_params(axis="x", rotation=45)
        ax_i.legend()
        ax_i.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_folder is not None:
        fig.savefig(output_folder / "reconstruction.png")


def show_latent_manifold_6x6(
    decoder,
    tenors,
    k=6,
    uncertainty=False,
    output_folder: Path = None,
):
    """Display a 6x6 grid of yield curves decoded from latent space."""
    # Build a uniform grid in (0,1) and map through Normal.icdf -> N(0,1)
    uniform = torch.linspace(0.01, 0.99, k)
    std_normal = torch.distributions.Normal(0.0, 1.0)
    z_grid_1d = std_normal.icdf(uniform)

    z1, z2 = torch.meshgrid(z_grid_1d, z_grid_1d, indexing="ij")
    z_grid = torch.dstack([z1, z2])

    # Decode p(x|z) for each latent location
    with torch.no_grad():
        mu, sigma = decoder(z_grid)
        curves = mu.reshape(k, k, -1).cpu().numpy()
        std = np.exp(sigma.reshape(k, k, -1).cpu().numpy())

    # Find global y-axis limits for consistent scaling
    y_min = curves.min()
    y_max = curves.max()
    y_margin = (y_max - y_min) * 0.1

    # Plot a k x k figure of decoded yield curves
    with plt.style.context("ggplot"):
        fig, axes = plt.subplots(k, k, figsize=(4 * k, 3 * k))

        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        for i in range(k):
            for j in range(k):
                ax = axes[k - i - 1, j]
                ax.plot(tenors, curves[j][i], "o-", linewidth=2, markersize=4)

                if uncertainty:
                    c = curves[j][i]
                    s = std[j][i]
                    ax.fill_between(
                        tenors,
                        c - 2 * s,
                        c + 2 * s,
                        color="gray",
                        alpha=0.3,
                    )

                ax.set_xticks(range(len(tenors)))
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.set_xticklabels(tenors, rotation=45, ha="right", fontsize=8)
                else:
                    ax.set_xticklabels([])

                if j != 0:
                    ax.set_yticklabels([])

    # Create a big, shared axis for z1/z2 labels
    big_ax = fig.add_subplot(111)
    big_ax.set_facecolor("none")

    big_ax.set_xticks(np.arange(k) + 0.5)
    big_ax.set_xticklabels([f"{v:.2f}" for v in z_grid_1d], rotation=0)
    big_ax.set_yticks(np.arange(k) + 0.5)
    big_ax.set_yticklabels([f"{v:.2f}" for v in z_grid_1d])

    big_ax.set_xlim(0, k)
    big_ax.set_ylim(0, k)
    big_ax.grid(False)

    # Configure ticks to be visible and padded away from subplot labels
    big_ax.tick_params(
        labelsize=18,
        colors="black",
        pad=10,
        labelleft=True,
        labelbottom=True,
    )

    # Ensure spines are visible (except top/right which is typical for graph axes)
    for spine in big_ax.spines.values():
        spine.set_visible(False)  # Turn off all initially

    # Explicitly set color and linewidth as ggplot style may hide them
    big_ax.spines["left"].set_visible(True)
    big_ax.spines["left"].set_color("black")
    big_ax.spines["left"].set_linewidth(2)
    big_ax.spines["left"].set_position(("outward", 30))  # Move spine slightly outward

    big_ax.spines["bottom"].set_visible(True)
    big_ax.spines["bottom"].set_color("black")
    big_ax.spines["bottom"].set_linewidth(2)
    big_ax.spines["bottom"].set_position(("outward", 40))

    big_ax.set_xlabel("Z1", fontsize=20, labelpad=15, color="black")
    big_ax.set_ylabel("Z2", fontsize=20, labelpad=15, color="black")

    if output_folder is not None:
        plt.savefig(output_folder / "latent_manifold.png")
    plt.show()


def vae_encode_all(vae, data_loader):
    vae.eval()

    all_z = []
    all_x = []
    with torch.no_grad():
        for (xb,) in data_loader:
            if getattr(vae, "use_cuda", False):
                xb = xb.cuda()
            all_x.append(xb.cpu())
            z_loc, _ = vae.encoder(xb)
            all_z.append(z_loc.cpu())

    z = torch.cat(all_z, dim=0).numpy()
    x = torch.cat(all_x, dim=0).numpy()

    return z, x


def cvae_encode_all(vae, data_loader):
    vae.eval()
    all_z = []
    all_x = []
    all_c = []
    with torch.no_grad():
        for xb, cb in data_loader:
            if getattr(vae, "use_cuda", False):
                xb = xb.cuda()
                cb = cb.cuda()
            all_x.append(xb.cpu())
            all_c.append(cb.cpu())
            z_loc, _ = vae.encoder(xb, cb)
            all_z.append(z_loc.cpu())

    z = torch.cat(all_z, dim=0).numpy()
    x = torch.cat(all_x, dim=0).numpy()
    c = torch.cat(all_c, dim=0).numpy()
    return z, x, c


# For CVAE: scatter plot colored by each conditioning variable
def latent_space_scatter_cvae(
    vae, loader, output_folder: Path, c_names=None, lims=None
):
    """Plot latent space colored by each conditioning variable, with colorbars."""

    z, _, c = cvae_encode_all(vae, loader)

    n_cond = c.shape[1]
    if c_names is None:
        c_names = [f"c{i + 1}" for i in range(n_cond)]

    # Arrange subplots: up to 3 columns, multiple rows if needed
    n_cols = min(3, n_cond)
    n_rows = int(np.ceil(n_cond / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    axes = np.array(axes).reshape(-1)  # flatten in case axes is 2D
    for i in range(n_cond):
        ax = axes[i]
        sc = ax.scatter(z[:, 0], z[:, 1], s=10, c=c[:, i], cmap="viridis", alpha=0.7)
        ax.set_title(f"Latent space colored by {c_names[i]}", fontsize=14)
        ax.set_facecolor("white")
        ax.grid(False)
        # Remove axis tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Remove box/spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        if lims is not None:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        else:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)
    # Hide any unused axes
    for j in range(n_cond, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    plt.show()
    fig.savefig(output_folder / "latent_space_scatter_c_all.png")
    plt.close(fig)


def _latent_space_scatter(z, ax, x=False, fig=None):
    """Plot the latent space representation of test data."""

    if x is not False:
        w = pca_cps_2(pd.DataFrame(x), 3)
        col = np.apply_along_axis(blend_3_colors, 1, w)
    else:
        col = "blue"

    ax.scatter(z[:, 0], z[:, 1], s=6, alpha=0.5, c=col)
    ax.set_xlabel("z1 (mean)", fontsize=12)
    ax.set_ylabel("z2 (mean)", fontsize=12)
    # ax.set_title("Yield Curve Latent Space", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    if x is not False and fig is not None:
        ax_legend = fig.add_axes([0.15, 0.12, 0.15, 0.15])
        create_triangle_legend_ternary(ax=ax_legend, scale=20)


def _latent_space_kde(z):
    """Plot joint KDE of latent space."""
    df = pd.DataFrame(z, columns=["Z1", "Z2"])
    g = sns.jointplot(
        data=df,
        x="Z1",
        y="Z2",
        kind="kde",
        fill=True,
        height=7,
    )
    g.ax_joint.set_xlim(-3, 3)
    g.ax_joint.set_ylim(-3, 3)

    return g


def plot_latent_space(vae, loader, output_folder: Path, pca_colormap=False):
    """Plot combined latent space visualization with scatter and KDE."""
    z, x = vae_encode_all(vae, loader)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    _latent_space_scatter(z, ax, x=x if pca_colormap else False, fig=fig)
    fig.savefig(output_folder / "latent_space_scatter.png")

    g = _latent_space_kde(z)
    g.savefig(output_folder / "latent_space_kde.png")


def plot_heatmaps(z1, z2, output_folder):
    """Plot autocorrelation and cross-correlation heatmaps for two series."""

    def autocorr_matrix(x, max_lag):
        """Return autocorrelation matrix for lags 0..max_lag-1."""
        lags = [np.roll(x, -lag) for lag in range(max_lag)]
        mat = np.column_stack(lags)
        valid_idx = np.arange(len(x) - max_lag + 1)
        mat = mat[valid_idx]
        return np.corrcoef(mat.T)

    max_lag = 8
    # 5x5 autocorrelation heatmaps for PC1 and PC2
    ac_pc1 = autocorr_matrix(z1, max_lag)
    ac_pc2 = autocorr_matrix(z2, max_lag)

    # 5x5 autocorrelation heatmaps for squared PC1 and PC2
    ac_pc1_sq = autocorr_matrix(z1**2, max_lag)
    ac_pc2_sq = autocorr_matrix(z2**2, max_lag)

    # 2x2 matrices for PC1/PC2: correlation, lag-one and lag-two cross-correlation
    def cross_corr_matrix(x, y, lags=[0, 1, 2]):
        vals = []
        for lag in lags:
            if lag == 0:
                corr = np.corrcoef(x, y)[0, 1]
            else:
                corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            vals.append(corr)
        return np.array(vals)

    def pairwise_matrix(x, y, lag):
        return np.array(
            [
                [
                    cross_corr_matrix(x, x, [lag])[0],
                    cross_corr_matrix(x, y, [lag])[0],
                ],
                [
                    cross_corr_matrix(y, x, [lag])[0],
                    cross_corr_matrix(y, y, [lag])[0],
                ],
            ]
        )

    cc_matrix_corr = pairwise_matrix(z1, z2, 0)
    cc_matrix_lag1 = pairwise_matrix(z1, z2, 1)
    cc_matrix_lag2 = pairwise_matrix(z1, z2, 2)

    cc_matrix_corr_sq = pairwise_matrix(z1**2, z2**2, 0)
    cc_matrix_lag1_sq = pairwise_matrix(z1**2, z2**2, 1)
    cc_matrix_lag2_sq = pairwise_matrix(z1**2, z2**2, 2)

    # Compose all plots in one figure: 6 rows x 7 columns
    fig = plt.figure(figsize=(14, 12))

    # Big heatmaps (5x5) and their titles, each in a 3x3 subspace
    big_heatmaps = [
        (ac_pc1, "Z1 Autocorrelation", 0, 0),
        (ac_pc2, "Z2 Autocorrelation", 0, 3),
        (ac_pc1_sq, "Z1$^2$ Autocorrelation", 3, 0),
        (ac_pc2_sq, "Z2$^2$ Autocorrelation", 3, 3),
    ]
    for mat, title, row, col in big_heatmaps:
        ax = plt.subplot2grid((6, 7), (row, col), rowspan=3, colspan=3, fig=fig)
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
            xticklabels=False,
            yticklabels=False,
            vmin=-1,
            vmax=1,
            cbar=False,
            square=True,
            linewidth=0.2,
        )
        ax.set_title(title)
        # ax.set_aspect("equal")
        # ax.axis("off")

    # Small 2x2 matrices and their titles
    small_matrices = [
        (cc_matrix_corr, "Correlation", 0),
        (cc_matrix_lag1, "Lag-1 XCorr", 1),
        (cc_matrix_lag2, "Lag-2 XCorr", 2),
        (cc_matrix_corr_sq, "Sq. Corr Matrix", 3),
        (cc_matrix_lag1_sq, "Sq. Lag-1 XCorr", 4),
        (cc_matrix_lag2_sq, "Sq. Lag-2 XCorr", 5),
    ]
    for i, (mat, title, row) in enumerate(small_matrices):
        ax = plt.subplot2grid((6, 7), (row, 6), rowspan=1, colspan=1, fig=fig)
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
            xticklabels=False,
            yticklabels=False,
            vmin=-1,
            vmax=1,
            cbar=False,
            square=True,
            linewidth=0.2,
        )
        ax.set_title(title, fontsize=10)
        # ax.set_aspect("equal")
        # ax.axis("off")

    fig.suptitle("State Variables Autocorrelation and Cross-Correlation Summary")
    fig.savefig(output_folder / "state_variables_autocorr_crosscorr_summary.png")


def plot_2x2_acf(z1, z2, output_folder=None):
    fig_acf, axes_acf = plt.subplots(2, 2, figsize=(10, 8))
    acf_series = [z1, z2, z1**2, z2**2]
    acf_titles = ["Z1 ACF", "Z2 ACF", "Z1$^2$ ACF", "Z2$^2$ ACF"]
    for i, (series, title) in enumerate(zip(acf_series, acf_titles)):
        ax = axes_acf.flat[i]
        plot_acf(series, ax=ax, lags=10)
        ax.set_title(title)
    fig_acf.tight_layout()
    fig_acf.suptitle("Autocorrelograms (ACF) of State Variables", fontsize=18, y=1.02)
    if output_folder is not None:
        name = "state_variables_acf_autocorrelograms.png"
        fig_acf.savefig(output_folder / name)
