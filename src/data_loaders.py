from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class ConditionalDataset(Dataset):
    def __init__(self, x, c):
        self.tds = TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(c, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.tds)

    def __getitem__(self, item):
        x, c = self.tds[item]
        return x, c


class HierarchicalDataset(Dataset):
    """Dataset yielding (x1, c1, x2, c2) tuples for hierarchical models.

    Each sample pairs one OMX observation (x1) with one single-name
    observation (x2) on the same date.  c1 contains OMX t-1 levels;
    c2 contains single-name t-1 levels, sector one-hot, and event flag.
    The OMX row is repeated once per name, giving
    N_valid_dates × N_single_names total samples.
    """

    def __init__(self, x1, c1, x2, c2):
        self.x1 = torch.as_tensor(x1, dtype=torch.float32)
        self.c1 = torch.as_tensor(c1, dtype=torch.float32)
        self.x2 = torch.as_tensor(x2, dtype=torch.float32)
        self.c2 = torch.as_tensor(c2, dtype=torch.float32)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.c1[idx], self.x2[idx], self.c2[idx]


def pre_process_data(df: pd.DataFrame, normalise: bool, changes=True) -> dict:
    if changes:
        x = df.diff()
    else:
        x = df.copy()

    if normalise:
        mean, std = x.mean().to_numpy(), x.std().to_numpy()
        x = (x - mean) / std

    is_na = x.isna().any(axis=1)

    return {
        "original": df.to_numpy(),
        "X": x[~is_na].to_numpy(),
        "get_x": lambda x: x * std + mean if normalise else x,
    }


def attach_conditional_features(
    dataset: dict[np.ndarray], normalise: bool = False
) -> dict:

    shifts = np.diff(dataset["original"], axis=0)
    curves = dataset["original"][1:, :]

    # Shape factors
    mid_point = curves.shape[1] // 2
    level = curves[:, mid_point]
    rr = curves[:, -1] - curves[:, 0]
    fly = curves[:, -1] - 2 * level + curves[:, 0]

    # Volatility factors
    rvol_5 = pd.Series(shifts[:, mid_point]).rolling(5).std()
    rvol_20 = pd.Series(shifts[:, mid_point]).rolling(20).std()

    c = np.vstack([level, rr, fly, rvol_5, rvol_20]).T
    mask = ~np.isnan(c).any(axis=1)
    mask[-1] = False  # not observable

    c = c[mask, :]
    x = dataset["X"][-c.shape[0] :, :]

    assert x.shape[0] == c.shape[0]

    if normalise:
        c_mean, c_std = c.mean(axis=0), c.std(axis=0)
        c = (c - c_mean) / c_std

    output = dataset.copy()
    output["X"] = x
    output["C"] = c
    output["get_c"] = lambda c: c * c_std + c_mean if normalise else c

    return output


def load_yield_data_vae(
    file_path: Path,
    batch_size: int = 256,
    train_split: float = 0.8,
    normalise: bool = False,
):
    original = pd.read_csv(file_path)

    # Split into train and test sets
    n_train = int(len(original) * train_split)
    subsets = (original[:n_train], original[n_train:])

    transform = [lambda df: pre_process_data(df, normalise=normalise)]

    def make_loader(data: dict, shuffle: bool):
        tds = TensorDataset(torch.tensor(data["X"], dtype=torch.float32))
        return DataLoader(tds, batch_size=batch_size, shuffle=shuffle)

    dataloaders = {}
    modes = ["train", "test"]
    for mode, data in zip(modes, subsets):
        processed = reduce(lambda acc, fn: fn(acc), transform, data)
        dataloaders[mode] = make_loader(processed, shuffle=(mode == "train"))

    return dataloaders, original.columns.tolist(), processed


def load_yield_data_cvae(
    file_path: Path,
    batch_size: int = 256,
    train_split: float = 0.8,
    normalise: bool = False,
    changes: bool = True,
):
    original = pd.read_csv(file_path)

    # Split into train and test sets
    n_train = int(len(original) * train_split)
    subsets = (original[:n_train], original[n_train:])

    transform = [
        lambda df: pre_process_data(df, normalise=normalise, changes=changes),
        lambda df: attach_conditional_features(df, normalise=normalise),
    ]

    def make_loader(data: dict, shuffle: bool):
        dataset = ConditionalDataset(data["X"], data["C"])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    dataloaders = {}
    modes = ["train", "test"]
    for mode, data in zip(modes, subsets):
        processed = reduce(lambda acc, fn: fn(acc), transform, data)
        dataloaders[mode] = make_loader(processed, shuffle=(mode == "train"))

    return dataloaders, original.columns.tolist(), processed


# ---------------------------------------------------------------------------
# Implied-volatility data loader for the Hierarchical VAE
# ---------------------------------------------------------------------------


def _pivot_iv_surfaces(
    df: pd.DataFrame, value_types: list[str] | None = None
) -> tuple[dict[str, pd.DataFrame], pd.MultiIndex]:
    """Pivot long-format IV rows to one wide DataFrame per name.

    Returns (surfaces, grid) where *surfaces* maps name → DataFrame with
    a DatetimeIndex (or whatever the date column holds) and a consistent
    MultiIndex(type, strike, tau) across all names.
    """
    if value_types is not None:
        df = df[df["type"].isin(value_types)]

    grid = pd.MultiIndex.from_product(
        [sorted(df[c].unique()) for c in ("type", "strike", "tau")],
        names=["type", "strike", "tau"],
    )

    surfaces = {}
    for name, grp in df.groupby("name"):
        wide = grp.pivot_table(
            index="date", columns=["type", "strike", "tau"], values="value"
        )
        surfaces[name] = wide.reindex(columns=grid).sort_index()

    return surfaces, grid


def _zscore(arr, mean, std):
    return (arr - mean) / std


def load_iv_data_hvae(
    file_path: Path,
    batch_size: int = 256,
    train_split: float = 0.8,
    normalise: bool = True,
    omx_name: str = "OMX",
    value_types: list[str] | None = None,
):
    """Load implied-volatility surfaces for the hierarchical VAE.

    Splits data by the ``name`` column into an index surface (OMX → x1,
    stage-1) and single-name surfaces (→ x2, stage-2).

    Pairing strategy
    ~~~~~~~~~~~~~~~~
    For each date *t* every single-name observation is paired with the
    single OMX observation on that date.  The OMX row is therefore
    repeated once per available single name, yielding
    ``N_valid_dates × N_single_names`` training samples.  During HVAE
    training each mini-batch contains random ``(x1, x2, c)`` triples
    drawn from this Cartesian pairing so the stage-1 encoder sees the
    same OMX surface from multiple gradient paths while the stage-2
    components learn across the cross-section of single names.

    Pre-processing per surface
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Pivot to wide format  (date × grid of type / strike / tau).
    2. Log-changes:  ``Δ_t = log V_t − log V_{t−1}``.
    3. Z-score normalise (train-set statistics applied to both splits).

    Conditioning vector **c1** (per OMX observation)::

        [ t−1 OMX IV level (n_grid) ]

    Conditioning vector **c2** (per single-name observation)::

        [ t−1 IV level (n_grid) | sector one-hot (n_sectors) | event_flag (1) ]

    If ``sector`` or ``event_flag`` columns are absent a zero dummy is used.

    Expected CSV columns
        date, name, strike, tau, tau_label, type, value
    Optional columns
        sector, event_flag

    Returns
    -------
    dataloaders : dict  – {"train": DataLoader, "test": DataLoader}
                  each batch yields ``(x1, c1, x2, c2)``.
    columns     : dict  – grid labels, conditioning names, and dimension info
                  (keys: grid, c1_names, c2_names, x1_dim, x2_dim, c1_dim, c2_dim).
    processed   : dict  – test-split arrays and inverse-transform lambdas
                  (keys: X1, C1, X2, C2, get_x1, get_x2, get_c1, get_c2).
    """
    raw = pd.read_csv(file_path)
    surfaces, grid = _pivot_iv_surfaces(raw, value_types)
    n_grid = len(grid)

    # --- OMX vs single names ------------------------------------------------
    omx_wide = surfaces.pop(omx_name)
    sn_names = sorted(surfaces.keys())

    # dates present in OMX *and* every single name
    common = reduce(
        lambda idx, nm: idx.intersection(surfaces[nm].index),
        sn_names,
        omx_wide.index,
    ).sort_values()

    # temporal train / test split
    n_train = int(len(common) * train_split)
    date_splits = {"train": common[:n_train], "test": common[n_train:]}

    # --- optional categorical / event columns --------------------------------
    has_sector = "sector" in raw.columns
    if has_sector:
        sec_map = raw.drop_duplicates("name").set_index("name")["sector"]
        all_sectors = sorted(sec_map.dropna().unique())
        sec_idx = {s: i for i, s in enumerate(all_sectors)}
        n_sectors = len(all_sectors)
    else:
        n_sectors = 1

    has_event = "event_flag" in raw.columns
    if has_event:
        _ev = (
            raw[["name", "date", "event_flag"]]
            .drop_duplicates(["name", "date"])
            .set_index(["name", "date"])["event_flag"]
        )

    # --- pair OMX with each single name per date -----------------------------
    def _make_pairs(dates):
        omx_vals = omx_wide.loc[dates]
        omx_log = np.log(omx_vals).diff()
        omx_lag = omx_vals.shift(1)
        x1_all, c1_all, x2_all, c2_all = [], [], [], []

        for nm in sn_names:
            sn = surfaces[nm].loc[dates]
            sn_log = np.log(sn).diff()
            sn_lag = sn.shift(1)

            valid = ~(
                omx_log.isna().any(axis=1)
                | omx_lag.isna().any(axis=1)
                | sn_log.isna().any(axis=1)
                | sn_lag.isna().any(axis=1)
            )
            vd = dates[valid]
            if vd.empty:
                continue

            x1_all.append(omx_log.loc[vd].values)
            c1_all.append(omx_lag.loc[vd].values)
            x2_all.append(sn_log.loc[vd].values)

            n = len(vd)
            level = sn_lag.loc[vd].values

            sec = np.zeros((n, n_sectors))
            if has_sector:
                sec[:, sec_idx.get(sec_map.get(nm), 0)] = 1.0

            evt = np.zeros((n, 1))
            if has_event:
                idx = pd.MultiIndex.from_arrays(
                    [np.full(n, nm), vd], names=["name", "date"]
                )
                evt = _ev.reindex(idx).fillna(0).values.reshape(-1, 1)

            c2_all.append(np.hstack([level, sec, evt]))

        return (
            np.vstack(x1_all),
            np.vstack(c1_all),
            np.vstack(x2_all),
            np.vstack(c2_all),
        )

    splits = {m: _make_pairs(d) for m, d in date_splits.items()}

    # --- normalise using training statistics ---------------------------------
    # indices: 0=x1, 1=c1, 2=x2, 3=c2
    inv_x1 = inv_x2 = lambda x: x
    if normalise:
        for i, tag in enumerate(["x1", "x2"]):
            pos = {"x1": 0, "x2": 2}[tag]
            m, s = splits["train"][pos].mean(0), splits["train"][pos].std(0)
            s[s == 0] = 1.0
            for mode in splits:
                arr = list(splits[mode])
                arr[pos] = _zscore(arr[pos], m, s)
                splits[mode] = tuple(arr)
            if tag == "x1":
                inv_x1 = lambda x, _m=m, _s=s: x * _s + _m
            else:
                inv_x2 = lambda x, _m=m, _s=s: x * _s + _m

    # normalise c1 (OMX levels — all continuous)
    inv_c1 = lambda c: c
    if normalise:
        c1m = splits["train"][1].mean(0)
        c1s = splits["train"][1].std(0)
        c1s[c1s == 0] = 1.0
        for mode in splits:
            arr = list(splits[mode])
            arr[1] = _zscore(arr[1], c1m, c1s)
            splits[mode] = tuple(arr)
        inv_c1 = lambda c, _m=c1m, _s=c1s: c * _s + _m

    # normalise level block of c2 (leave one-hots / flags untouched)
    inv_c2 = lambda c: c
    if normalise:
        c2m = splits["train"][3][:, :n_grid].mean(0)
        c2s = splits["train"][3][:, :n_grid].std(0)
        c2s[c2s == 0] = 1.0
        for mode in splits:
            arr = list(splits[mode])
            arr[3] = arr[3].copy()
            arr[3][:, :n_grid] = _zscore(arr[3][:, :n_grid], c2m, c2s)
            splits[mode] = tuple(arr)
        inv_c2 = lambda c, _m=c2m, _s=c2s, _ng=n_grid: np.hstack(
            [c[:, :_ng] * _s + _m, c[:, _ng:]]
        )

    # --- data loaders --------------------------------------------------------
    def make_loader(data, shuffle):
        return DataLoader(
            HierarchicalDataset(*data), batch_size=batch_size, shuffle=shuffle
        )

    dataloaders = {m: make_loader(splits[m], m == "train") for m in splits}

    c1_dim = n_grid
    c2_dim = n_grid + n_sectors + 1
    columns = {
        "grid": grid,
        "c1_names": [f"omx_level_{i}" for i in range(n_grid)],
        "c2_names": (
            [f"level_{i}" for i in range(n_grid)]
            + [f"sector_{i}" for i in range(n_sectors)]
            + ["event_flag"]
        ),
        "x1_dim": n_grid,
        "x2_dim": n_grid,
        "c1_dim": c1_dim,
        "c2_dim": c2_dim,
    }

    processed = {
        "X1": splits["test"][0],
        "C1": splits["test"][1],
        "X2": splits["test"][2],
        "C2": splits["test"][3],
        "get_x1": inv_x1,
        "get_x2": inv_x2,
        "get_c1": inv_c1,
        "get_c2": inv_c2,
    }

    return dataloaders, columns, processed
