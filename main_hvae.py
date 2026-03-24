"""End-to-end smoke test: load synthetic IV data → train HVAE."""

from pathlib import Path

import torch

from src.data_loaders import load_iv_data_hvae
from src.vae.hvae import train

torch.manual_seed(42)

if __name__ == "__main__":
    path = Path(r"C:\dev\git\pml\data\synthetic_iv_surfaces.csv")

    loaders, cols, data = load_iv_data_hvae(
        file_path=path,
        batch_size=256,
        normalise=True,
        value_types=["IVS"],
    )

    c_dim = cols["c1_dim"] + cols["c2_dim"]

    print(f"Grid size : {cols['x1_dim']}")
    print(
        f"c1_dim    : {cols['c1_dim']}  c2_dim: {cols['c2_dim']}  total c_dim: {c_dim}"
    )
    print(f"Train     : {len(loaders['train'].dataset)}")
    print(f"Test      : {len(loaders['test'].dataset)}")

    # quick shape sanity check
    x1, c1, x2, c2 = next(iter(loaders["train"]))
    print(f"Batch shapes: x1={x1.shape}, c1={c1.shape}, x2={x2.shape}, c2={c2.shape}")

    vae = train(
        data_loaders=(loaders["train"], loaders["test"]),
        x1_dim=cols["x1_dim"],
        x2_dim=cols["x2_dim"],
        c_dim=c_dim,
        z1_dim=3,
        z2_dim=3,
        hidden_dim=64,
        beta=1.0,
        annealing_start=0.01,
        num_epochs=30,
        test_frequency=5,
        learning_rate=1e-3,
    )

    # quick reconstruction check on test set
    vae.eval()
    x1t = torch.tensor(data["X1"], dtype=torch.float32)
    c1t = torch.tensor(data["C1"], dtype=torch.float32)
    x2t = torch.tensor(data["X2"], dtype=torch.float32)
    c2t = torch.tensor(data["C2"], dtype=torch.float32)
    ct = torch.cat([c1t, c2t], dim=-1)

    with torch.no_grad():
        x1_rec, x2_rec = vae.reconstruct_map(x1t, x2t, ct)

    mse1 = ((x1t - x1_rec) ** 2).mean().item()
    mse2 = ((x2t - x2_rec) ** 2).mean().item()
    print(f"\nTest reconstruction MSE — x1: {mse1:.6f}  x2: {mse2:.6f}")
