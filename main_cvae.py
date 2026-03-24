# %%
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from pyparsing import col
from sklearn.decomposition import PCA

from src.data_loaders import load_yeld_data_cvae
from src.plot_helper import (
    cvae_encode_all,
    latent_space_scatter_cvae,
    show_latent_manifold_6x6,
)
from src.vae.cvae import train

plt.style.use("ggplot")

loglogger = logging.getLogger(__name__)


if __name__ == "__main__":
    path = Path(r"C:\dev\git\pml\data\ns_curves_vol_cluster.csv")
    # path = Path(r"C:\dev\git\pml\data\ns_curves_local_vol.csv")
    output_folder = Path(r"C:\dev\git\pml\results\cvae")
    output_folder.mkdir(parents=True, exist_ok=True)

    heteroscedastic = False
    num_epochs = 100 if heteroscedastic else 150
    c_ref_col = 4

    loaders, tenors, data = load_yeld_data_cvae(
        file_path=path,
        batch_size=256,
        changes=True,
        normalise=True,
    )

    train_loader = loaders["train"]
    test_loader = loaders["test"]
    print(f"Tenors: {tenors}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # train VAE
    vae = train(
        data_loaders=(train_loader, test_loader),
        beta=1.5,
        anneling_start=0,
        z_dim=2,
        hidden_dim=50,
        num_epochs=num_epochs,
        heteroscedastic=heteroscedastic,
        dhsic_lambda=10000,
        learning_rate=1e-3,
    )

    # %%
    _, x_train, c_train = cvae_encode_all(vae, train_loader)
    z_test, x_test, c_test = cvae_encode_all(vae, test_loader)

    c_names = ["Level", "Slope", "Curvature", "RV5", "RV20"]
    c_median = torch.tensor(np.median(c_test, axis=0))

    pca = PCA(n_components=2)
    _ = pca.fit(x_train)
    print(pca.explained_variance_ratio_)

    pca_factors = x_test @ pca.components_.T
    vae_factors = z_test

    # %%
    # --- Traverse figure ---
    def decoder_fixed_c(z):
        c = c_median.expand(*z.shape[:-1], -1)
        return vae.decoder(z, c)

    show_latent_manifold_6x6(decoder_fixed_c, tenors, uncertainty=True)

    # %%
    # --- Traverse figure: 5x5 grid for conditioning variables ---
    vae.eval()
    reference_z = vae.encoder(torch.tensor(np.repeat(1, 9)), c_median)[0].detach()
    percentiles = [1, 25, 50, 75, 99]
    fig, axes = plt.subplots(5, 5, figsize=(18, 16), sharex=True, sharey=True)
    for row, c_idx in enumerate(range(5)):
        c_vals = np.percentile(c_test[:, c_idx], percentiles)
        for col, pval in enumerate(c_vals):
            # Use median values for other conditioning variables
            c_vec = np.median(c_test, axis=0)
            c_vec[c_idx] = pval
            c_tensor = torch.tensor(c_vec, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                x_reco, _ = vae.decoder(reference_z, c_tensor)
            ax = axes[row, col]
            ax.plot(tenors, x_reco.squeeze().numpy(), "o-", linewidth=2, markersize=5)
            if col == 0:
                ax.set_ylabel(c_names[row], fontsize=13)
            if row == 0:
                ax.set_title(f"{percentiles[col]}%", fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_ylim((-2), 2)
    for ax in axes[-1, :]:
        ax.set_xticks(range(len(tenors)))
        ax.set_xticklabels(tenors, rotation=45, ha="right")
    fig.suptitle("CVAE Traverse", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_folder / "traverse_figure.png")
    plt.show()

    # %%
    # --- Traverse figure: 5x5 grid for conditioning variables and percentiles ---
    latent_space_scatter_cvae(vae, test_loader, output_folder, c_names=c_names)

    # %%
    # --- Calculate reconstruction error for CVAE and PCA ---
    with torch.no_grad():
        x_cvae_reco = vae.reconstruct_map(torch.tensor(x_test), torch.tensor(c_test))
    cvae_mse = np.mean((x_test - x_cvae_reco.numpy()) ** 2)

    x_pca_reco = pca.inverse_transform(pca.transform(x_test))
    pca_mse = np.mean((x_test - x_pca_reco) ** 2)

    print(f"CVAE reconstruction MSE: {cvae_mse:.6f}")
    print(f"PCA reconstruction MSE: {pca_mse:.6f}")

    # --- Plot histogram of reconstruction errors ---
    cvae_errors = np.mean((x_test - x_cvae_reco.numpy()) ** 2, axis=1)
    pca_errors = np.mean((x_test - x_pca_reco) ** 2, axis=1)
    plt.figure(figsize=(8, 5))
    bins = np.histogram(np.hstack((cvae_errors, pca_errors)), bins=60)[1]
    plt.hist(cvae_errors, bins=bins, alpha=0.6, label="CVAE", color="tab:blue")
    plt.hist(pca_errors, bins=bins, alpha=0.6, label="PCA", color="tab:orange")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Reconstruction Errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / "reconstruction_error_histogram.png")
    plt.show()

    # %%
    # --- plot latent factors against C ---
    fig, ax = plt.subplots(figsize=(7, 7))
    threshold = np.percentile(c_test[:, c_ref_col], 80)
    idx_high = c_test[:, c_ref_col] >= threshold
    idx_low = c_test[:, c_ref_col] < threshold
    ax.scatter(
        z_test[idx_high, 0],
        z_test[idx_high, 1],
        s=6,
        alpha=0.7,
        c="tab:blue",
        label="Top 20%",
    )
    ax.scatter(
        z_test[idx_low, 0],
        z_test[idx_low, 1],
        s=6,
        alpha=0.7,
        c="tab:orange",
        label="Bottom 80%",
    )
    ax.set_xlabel("z1 (mean)", fontsize=12)
    ax.set_ylabel("z2 (mean)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(f"Latent Factors vs {c_names[c_ref_col].upper()}", fontsize=15)
    ax.legend()
    plt.show()
    # %%
    # --- plot histograms for Z1 split by RV20 threshold ---
    fig, ax = plt.subplots(figsize=(8, 5))
    threshold = np.percentile(c_test[:, c_ref_col], 80)
    z1_high = z_test[c_test[:, c_ref_col] >= threshold, 0]
    z1_low = z_test[c_test[:, c_ref_col] < threshold, 0]
    bins = np.histogram(np.hstack((z1_high, z1_low)), bins=60)[1]
    ax.hist(z1_low, bins=bins, alpha=0.8, label="Bottom 80%", color="tab:blue")
    ax.hist(z1_high, bins=bins, alpha=0.8, label="Top 20%", color="tab:orange")
    # Standard normal PDF line
    x = np.linspace(bins[0], bins[-1], 300)
    pdf = stats.norm.pdf(x)
    pdf_scaled = pdf * len(z_test) * (bins[1] - bins[0])
    ax.plot(x, pdf_scaled, "k--", label="Standard Normal PDF")
    ax.set_xlabel("Z1 values")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Z1 by {c_names[c_ref_col].upper()} threshold")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- plot histograms for Z2 split by RV20 threshold ---
    fig, ax = plt.subplots(figsize=(8, 5))
    z2_high = z_test[c_test[:, c_ref_col] >= threshold, 1]
    z2_low = z_test[c_test[:, c_ref_col] < threshold, 1]
    bins = np.histogram(np.hstack((z2_high, z2_low)), bins=60)[1]
    ax.hist(z2_low, bins=bins, alpha=0.8, label="Bottom 80%", color="tab:blue")
    ax.hist(z2_high, bins=bins, alpha=0.9, label="Top 20%", color="tab:orange")
    # Standard normal PDF line
    x = np.linspace(bins[0], bins[-1], 300)
    pdf = stats.norm.pdf(x)
    pdf_scaled = pdf * len(z_test) * (bins[1] - bins[0])
    ax.plot(x, pdf_scaled, "k--", label="Standard Normal PDF")
    ax.set_xlabel("Z2 values")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Z2 by {c_names[c_ref_col].upper()} threshold")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # %%
    # --- Autocorrelation and cross-correlation visualizations for PCA factors ---
    import numpy as np

    # Use first two PCA factors
    pc1 = pca_factors[:, 0]
    pc2 = pca_factors[:, 1]

    # plot_heatmaps(pc1, pc2, output_folder)
    # plot_2x2_acf(pc1, pc2, output_folder)
    # plot_heatmaps(vae_factors[:, 0], vae_factors[:, 1], output_folder)
    # plot_2x2_acf(vae_factors[:, 0], vae_factors[:, 1], output_folder)

    z_test_scale = vae.encoder(torch.tensor(x_test), torch.tensor(c_test))[1].detach()
    try:
        x_test_scale = vae.vol_scale(torch.tensor(c_test)).detach()
    except AttributeError:
        x_test_log_scale = vae.decoder(torch.tensor(z_test), torch.tensor(c_test))[
            1
        ].detach()
        x_test_scale = torch.exp(x_test_log_scale)

    plt.plot(z_test_scale, label="Encoder scale")
    plt.plot(x_test_scale, label="Decoder scale")
    plt.title("Encoder & decoder scales")
    plt.show()

    plt.plot(x_cvae_reco.detach() - x_test)
    plt.plot(c_test[:, c_ref_col], label=f"Standardised {c_names[c_ref_col].upper()}")
    plt.title("CVAE reconstruction errors")
    plt.legend()
    plt.show()

    plt.plot(z_test)
    plt.title("Latent means")
    plt.show()

    print(np.corrcoef(z_test[:, 0] ** 2, c_test[:, c_ref_col]))
    print(np.corrcoef(z_test[:, 1] ** 2, c_test[:, c_ref_col]))

    # %%
    if hasattr(vae, "prior_scale"):
        prior_scale_test = vae.prior_scale(torch.tensor(c_test)).detach()
        plt.plot(prior_scale_test)
        plt.title("Prior scale")
        plt.show()

    # %%
