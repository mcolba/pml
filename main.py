from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_loaders import load_yield_data_vae
from src.plot_helper import (
    plot_latent_space,
    show_latent_manifold_6x6,
    show_overlaid_reconstruction,
    vae_encode_all,
)
from src.vae.vae import train

plt.style.use("ggplot")
torch.manual_seed(42)
np.random.seed(42)

if __name__ == "__main__":
    path = Path(r"C:\dev\git\pml\data\synthetic_yield_curves.csv")
    output_folder = Path(r"C:\dev\git\pml\results\vae")

    loaders, tenors, data = load_yield_data_vae(
        file_path=path,
        batch_size=256,
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
        x_dim=len(tenors),
        beta=1.5,
        anneling_start=0,
        z_dim=2,
        hidden_dim=50,
        num_epochs=80,
        learning_rate=1e-3,
    )

    # Reconstruct on test samples
    vae.eval()

    idx = [0, 500, 750, 1000]
    x_sample = torch.stack(test_loader.dataset[idx])
    with torch.no_grad():
        reco_sample = vae.reconstruct_map(x_sample)

    x_sample = x_sample[0]

    print("\n=== Generating Overlaid Reconstruction ===")
    show_overlaid_reconstruction(
        x_sample.detach().numpy(),
        reco_sample.detach().numpy(),
        tenors,
        output_folder,
    )

    print("\n=== Generating 6x6 Latent Manifold ===")
    show_latent_manifold_6x6(
        vae.decoder, tenors, k=5, uncertainty=True, output_folder=output_folder
    )

    print("\n=== Generating Latent Space Scatter Plot ===")
    plot_latent_space(vae, test_loader, output_folder=output_folder, pca_colormap=True)

    from sklearn.decomposition import PCA

    _, x_train = vae_encode_all(vae, train_loader)
    z_test, x_test = vae_encode_all(vae, test_loader)

    pca = PCA(n_components=2)
    _ = pca.fit(x_train)
    print(pca.explained_variance_ratio_)

    pca_factors = x_test @ pca.components_.T
    vae_factors = z_test

    # --- Calculate reconstruction error for VAE and PCA ---
    with torch.no_grad():
        x_cvae_reco = vae.reconstruct_map(torch.tensor(x_test))
    cvae_mse = np.mean((x_test - x_cvae_reco.numpy()) ** 2)

    x_pca_reco = pca.inverse_transform(pca.transform(x_test))
    pca_mse = np.mean((x_test - x_pca_reco) ** 2)

    print(f"CVAE reconstruction MSE: {cvae_mse:.6f}")
    print(f"PCA reconstruction MSE: {pca_mse:.6f}")

    x_demo = x_train[35]

    # Encode x_demo to get mean and std of latent variables
    with torch.no_grad():
        z_mu, z_std = vae.encoder(torch.tensor(x_demo).unsqueeze(0))
        z_mu = z_mu.squeeze().cpu().numpy()

        # Monte Carlo sampling in latent space
        n_samples = 1000
        z_samples = np.random.normal(z_mu, z_std, size=(n_samples, z_mu.shape[0]))
        z_samples_tensor = torch.tensor(z_samples, dtype=torch.float32)
        x_reco_samples = vae.decoder(z_samples_tensor)[0].cpu().numpy()

        # Compute mean and confidence interval (e.g., 2.5 and 97.5 percentiles)
        x_reco_mean = np.mean(x_reco_samples, axis=0)
        x_reco_lower = np.percentile(x_reco_samples, 2.5, axis=0)
        x_reco_upper = np.percentile(x_reco_samples, 97.5, axis=0)

    burn_in = 100
    num_steps = 5000
    x_with_nan = np.repeat(np.nan, x_demo.shape)
    x_with_nan[[2, 5]] = x_demo[[2, 5]]

    x = torch.tensor(x_with_nan, dtype=torch.float32, device="cpu")
    miss_mask = torch.isnan(x)  # True where missing
    obs_mask = ~miss_mask

    # Initialize missing values
    x_curr = x.clone()
    x_curr[miss_mask] = x_curr[obs_mask].mean()

    vae.eval()
    miss_samples = []
    x_loc_vec = []
    for t in range(num_steps):
        # 1) Sample z ~ q_theta(z|x)
        z_loc, z_scale = vae.encoder(x_curr.unsqueeze(0))
        z = torch.distributions.Normal(z_loc, z_scale).sample()

        # 2) Sample x ~ p_varphi(x|z)
        x_loc, log_x_scale = vae.decoder(z)
        x_scale = torch.exp(log_x_scale)

        x_prop = torch.distributions.Normal(x_loc, x_scale).sample().squeeze(0)

        # 3) Update only missing dims; keep observed fixed (conditioning)
        x_curr[miss_mask] = x_prop[miss_mask]
        # x_curr[obs_mask] = x[obs_mask]

        # 4) Collect samples after burn-in
        if t >= burn_in:
            miss_samples.append(x_curr[miss_mask].detach().cpu().numpy())
            x_loc_vec.append(x_loc.detach().cpu().numpy())

    miss_samples = np.stack(miss_samples, axis=0)
    miss_mean = miss_samples.mean(axis=0).astype(np.float32)
    miss_lower = np.percentile(miss_samples, 2.5, axis=0)
    miss_upper = np.percentile(miss_samples, 97.5, axis=0)

    x_imputed = x.cpu().numpy().copy()
    x_imputed[miss_mask] = miss_mean

    fig, ax = plt.subplots()
    ax.plot(tenors, x_demo, label="Original", color="orange", linewidth=2)
    ax.set_ylim(-1, 2)
    # Plot mean reconstruction
    # ax.plot(tenors, x_reco_mean, label="Reconstruction (mean)", color="blue")
    # Plot confidence interval for reconstruction
    # ax.fill_between(
    #     tenors,
    #     x_reco_lower,
    #     x_reco_upper,
    #     color="blue",
    #     alpha=0.2,
    #     label="95% CI (reconstruction)",
    # )

    # Plot imputed mean and its uncertainty using error bars
    ax.plot(tenors, x_imputed, label="Imputed (MCMM)", color="blue", linestyle="--")
    # Only show error bars for missing values
    yerr_lower = np.zeros_like(x_imputed)
    yerr_upper = np.zeros_like(x_imputed)
    yerr_lower[miss_mask] = x_imputed[miss_mask] - miss_lower
    yerr_upper[miss_mask] = miss_upper - x_imputed[miss_mask]
    ax.errorbar(
        tenors,
        x_imputed,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        color="blue",
        ecolor="blue",
        elinewidth=1,
        capsize=2,
    )
    # ax.plot(np.vstack(x_loc_vec).mean(axis=0))
    ax.legend()
    ax.set_title("VAE Imputation")
    plt.show()
