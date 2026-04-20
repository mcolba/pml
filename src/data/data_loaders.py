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


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-returns: Δ_t = log V_t − log V_{t−1}."""
    return np.log(df).diff()


def standardised_spot_return(spot: pd.Series, window: int) -> pd.Series:
    """Spot log-return standardised by rolling realised volatility.

    .. math::

        s_t = \\frac{r_t}{\\sigma_w(t-1)}

    where :math:`r_t = \\log S_t - \\log S_{t-1}` and
    :math:`\\sigma_w(t-1)` is the rolling standard deviation of
    log-returns over the previous *window* days (excluding today).

    The measure signals how large the current spot move is relative
    to recent realised volatility.

    Parameters
    ----------
    spot : pd.Series
        Date-indexed spot (or shortest-tenor forward) prices.
    window : int
        Rolling window in trading days (e.g. 5 for 1W, 21 for 1M).
    """
    log_ret = np.log(spot).diff()
    rvol = log_ret.shift(1).rolling(window).std()
    return log_ret / rvol


def make_normalisation_functions(
    x_train: np.ndarray,
) -> tuple:
    """Create z-score normalisation / unnormalisation from training data.

    Returns ``(norm_fn, unnorm_fn)`` where columns with ``std == 0``
    are clamped to 1.0 to avoid division by zero.
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return (
        lambda x, _m=mean, _s=std: (x - _m) / _s,
        lambda x, _m=mean, _s=std: x * _s + _m,
    )


def normalise(x: np.ndarray, fn) -> np.ndarray:
    """Apply a normalisation function to an array."""
    return fn(x)


# ---------------------------------------------------------------------------
# Pivot & feature extraction
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


def pivot_and_ffill(
    raw: pd.DataFrame,
    omx_name: str = "OMX",
    value_types: list[str] | None = None,
) -> dict:
    """Pivot long-format IV data into structured wide DataFrames.

    Parameters
    ----------
    raw : DataFrame
        Long-format IV data with columns
        ``date, name, type, strike, tau, value``.
    omx_name : str
        Ticker that identifies the index surface.
    value_types : list[str] | None
        If given, filter to these ``type`` values for the IVS grid
        (e.g. ``["IVS"]``).

    Returns
    -------
    dict with keys
        index         – DataFrame (date × grid) for the index surface.
        sn            – dict[name → DataFrame (date × grid)] single-name
                        surfaces.
        spot          – dict[name → Series (date → float)] shortest-tenor
                        forward price per name (proxy for spot).
        grid          – MultiIndex(type, strike, tau) for IVS columns.
        n_grid        – int, number of IVS grid points.
        sn_names      – sorted list of single-name tickers.
        common_dates  – Index of dates present in index AND every single
                        name.
    """
    surfaces, grid = _pivot_iv_surfaces(raw, value_types)

    # --- Extract spot proxy: shortest-tenor FWD per name ---
    fwd = raw[raw["type"] == "FWD"].copy()
    spot: dict[str, pd.Series] = {}
    if not fwd.empty:
        min_tau = fwd["tau"].min()
        fwd_short = fwd[fwd["tau"] == min_tau]
        for name, grp in fwd_short.groupby("name"):
            spot[name] = grp.set_index("date")["value"].sort_index().astype(float)

    # --- Separate index from single names ---
    index_surface = surfaces.pop(omx_name)
    sn_names = sorted(surfaces.keys())

    # --- Common dates across index and every single name ---
    common = reduce(
        lambda idx, nm: idx.intersection(surfaces[nm].index),
        sn_names,
        index_surface.index,
    ).sort_values()

    return {
        "index": index_surface,
        "sn": surfaces,
        "spot": spot,
        "grid": grid,
        "n_grid": len(grid),
        "sn_names": sn_names,
        "common_dates": common,
    }


def _build_rv_dict(
    spot: dict[str, pd.Series],
    sn_names: list[str],
) -> dict[tuple[str, str], tuple[float, float]]:
    """Build ``(name, date) → (std_ret_1W, std_ret_1M)`` lookup.

    Each value is the current spot log-return divided by the trailing
    realised volatility at the 1-week (5-day) and 1-month (21-day)
    horizons.
    """
    rv_dict: dict[tuple[str, str], tuple[float, float]] = {}
    for nm in sn_names:
        if nm not in spot:
            continue
        s = spot[nm]
        sr_1w = standardised_spot_return(s, 5)
        sr_1m = standardised_spot_return(s, 21)
        for date in s.index:
            w = sr_1w.get(date)
            m = sr_1m.get(date)
            if w is not None and m is not None and np.isfinite(w) and np.isfinite(m):
                rv_dict[(nm, date)] = (float(w), float(m))
    return rv_dict


# ---------------------------------------------------------------------------
# Pairing: match index (OMX) rows with each single-name row per date
# ---------------------------------------------------------------------------


def _extract_categorical_info(
    raw: pd.DataFrame,
) -> dict:
    """Pre-compute sector one-hot and event-flag lookups from *raw*.

    Only used when ``c2_mode == "full"``.  Returns a dict with keys
    ``has_sector``, ``has_event``, and associated lookup structures.
    """
    info: dict = {}

    info["has_sector"] = "sector" in raw.columns
    if info["has_sector"]:
        sec_map = raw.drop_duplicates("name").set_index("name")["sector"]
        all_sectors = sorted(sec_map.dropna().unique())
        info["sec_map"] = sec_map
        info["sec_idx"] = {s: i for i, s in enumerate(all_sectors)}
        info["n_sectors"] = len(all_sectors)
    else:
        info["n_sectors"] = 1

    info["has_event"] = "event_flag" in raw.columns
    if info["has_event"]:
        info["event_series"] = (
            raw[["name", "date", "event_flag"]]
            .drop_duplicates(["name", "date"])
            .set_index(["name", "date"])["event_flag"]
        )

    return info


def _make_pairs(
    index_surface: pd.DataFrame,
    sn_surfaces: dict[str, pd.DataFrame],
    sn_names: list[str],
    dates: pd.Index,
    c2_mode: str,
    rv_dict: dict,
    cat_info: dict,
) -> tuple[np.ndarray, ...]:
    """Build ``(x1, c1, x2, c2, names, dates)`` arrays for one date split.

    For each date the index observation is Cartesian-paired with every
    available single-name observation.

    ``x1 = log_returns(index)``, ``c1 = lag(index)``,
    ``x2 = log_returns(sn)``, ``c2`` depends on *c2_mode*.
    """
    omx_vals = index_surface.loc[dates]
    omx_log = log_returns(omx_vals)
    omx_lag = omx_vals.shift(1)

    n_sectors = cat_info.get("n_sectors", 1)

    x1_all, c1_all, x2_all, c2_all = [], [], [], []
    names_all, dates_all = [], []

    for nm in sn_names:
        sn = sn_surfaces[nm].loc[dates]
        sn_log = log_returns(sn)
        sn_lag = sn.shift(1)

        valid = ~(
            omx_log.isna().any(axis=1)
            | omx_lag.isna().any(axis=1)
            | sn_log.isna().any(axis=1)
            | sn_lag.isna().any(axis=1)
        )

        if c2_mode == "vol_shape_rv":
            drv_avail = pd.Series(
                [(nm, d) in rv_dict for d in dates],
                index=dates,
            )
            valid = valid & drv_avail

        vd = dates[valid]
        if vd.empty:
            continue

        x1_all.append(omx_log.loc[vd].values)
        c1_all.append(omx_lag.loc[vd].values)
        x2_all.append(sn_log.loc[vd].values)

        n = len(vd)
        names_all.append(np.full(n, nm))
        dates_all.append(vd.values)

        # ---- build C2 based on c2_mode ----
        if c2_mode == "none":
            c2_all.append(np.empty((n, 0)))
        elif c2_mode == "vol_shape":
            c2_all.append(sn_lag.loc[vd].values)
        elif c2_mode == "vol_shape_rv":
            level = sn_lag.loc[vd].values
            drv_vals = np.array([rv_dict[(nm, d)] for d in vd], dtype=np.float64)
            c2_all.append(np.hstack([level, drv_vals]))
        else:
            # "full" mode: level + sector one-hot + event_flag
            level = sn_lag.loc[vd].values
            sec = np.zeros((n, n_sectors))
            if cat_info["has_sector"]:
                sec[:, cat_info["sec_idx"].get(cat_info["sec_map"].get(nm), 0)] = 1.0
            evt = np.zeros((n, 1))
            if cat_info["has_event"]:
                idx = pd.MultiIndex.from_arrays(
                    [np.full(n, nm), vd], names=["name", "date"]
                )
                evt = (
                    cat_info["event_series"]
                    .reindex(idx)
                    .fillna(0)
                    .values.reshape(-1, 1)
                )
            c2_all.append(np.hstack([level, sec, evt]))

    return (
        np.vstack(x1_all),
        np.vstack(c1_all),
        np.vstack(x2_all),
        np.vstack(c2_all),
        np.concatenate(names_all),
        np.concatenate(dates_all),
    )


# ---------------------------------------------------------------------------
# Normalise paired splits
# ---------------------------------------------------------------------------


def _normalise_paired_splits(
    splits: dict[str, tuple],
    do_normalise: bool,
    c2_mode: str,
    n_grid: int,
) -> tuple[dict[str, tuple], dict]:
    """Z-score normalise x1, c1, x2, c2 using training statistics.

    Returns ``(normalised_splits, inv_fns)`` where *inv_fns* maps
    ``"x1"``, ``"x2"``, ``"c1"``, ``"c2"`` to their unnormalisation
    callables.
    """
    inv = {
        "x1": lambda x: x,
        "x2": lambda x: x,
        "c1": lambda c: c,
        "c2": lambda c: c,
    }

    if not do_normalise:
        return splits, inv

    # --- x1 and x2 ---
    for tag, pos in [("x1", 0), ("x2", 2)]:
        norm_fn, unnorm_fn = make_normalisation_functions(splits["train"][pos])
        inv[tag] = unnorm_fn
        for mode in splits:
            arr = list(splits[mode])
            arr[pos] = normalise(arr[pos], norm_fn)
            splits[mode] = tuple(arr)

    # --- c1 (OMX levels — all continuous) ---
    norm_c1, unnorm_c1 = make_normalisation_functions(splits["train"][1])
    inv["c1"] = unnorm_c1
    for mode in splits:
        arr = list(splits[mode])
        arr[1] = normalise(arr[1], norm_c1)
        splits[mode] = tuple(arr)

    # --- c2 (strategy depends on c2_mode) ---
    if c2_mode != "none":
        c2_train = splits["train"][3]
        if c2_mode in ("vol_shape", "vol_shape_rv"):
            # All columns are continuous → normalise everything
            norm_c2, unnorm_c2 = make_normalisation_functions(c2_train)
            inv["c2"] = unnorm_c2
            for mode in splits:
                arr = list(splits[mode])
                arr[3] = normalise(arr[3], norm_c2)
                splits[mode] = tuple(arr)
        else:
            # "full" mode: normalise continuous level block only
            norm_c2, unnorm_c2 = make_normalisation_functions(c2_train[:, :n_grid])
            for mode in splits:
                arr = list(splits[mode])
                arr[3] = arr[3].copy()
                arr[3][:, :n_grid] = normalise(arr[3][:, :n_grid], norm_c2)
                splits[mode] = tuple(arr)
            inv["c2"] = lambda c, _fn=unnorm_c2, _ng=n_grid: np.hstack(
                [_fn(c[:, :_ng]), c[:, _ng:]]
            )

    return splits, inv


# ---------------------------------------------------------------------------
# Date-batched hierarchical dataset (for hvae.py)
# ---------------------------------------------------------------------------


class DateBatchedHierarchicalDataset(Dataset):
    """Yields ``(x1, x2_padded, c2_padded, child_mask)`` per date.

    Parameters
    ----------
    x1 : ndarray (D, x1_dim)           one index observation per date
    x2 : ndarray (D, N_max, x2_dim)    padded child surfaces
    c2 : ndarray (D, N_max, c2_dim)    padded child conditioning
    child_mask : ndarray (D, N_max)     1.0 where child exists, 0.0 otherwise
    """

    def __init__(self, x1, x2, c2, child_mask):
        self.x1 = torch.as_tensor(x1, dtype=torch.float32)
        self.x2 = torch.as_tensor(x2, dtype=torch.float32)
        self.c2 = torch.as_tensor(c2, dtype=torch.float32)
        self.child_mask = torch.as_tensor(child_mask, dtype=torch.float32)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.c2[idx], self.child_mask[idx]


def _flat_to_date_batched(x1, x2, c2, dates):
    """Group flat (x1, x2, c2) pairs by date and pad children.

    Returns (x1_out, x2_out, c2_out, mask_out) arrays suitable for
    :class:`DateBatchedHierarchicalDataset`.
    """
    unique_dates, date_idx = np.unique(dates, return_inverse=True)
    D = len(unique_dates)
    x1_dim = x1.shape[1]
    x2_dim = x2.shape[1]
    c2_dim = c2.shape[1] if c2.ndim == 2 else 0

    children_per_date = np.bincount(date_idx, minlength=D)
    N_max = int(children_per_date.max())

    x1_out = np.zeros((D, x1_dim))
    x2_out = np.zeros((D, N_max, x2_dim))
    c2_out = np.zeros((D, N_max, max(c2_dim, 0)))
    mask_out = np.zeros((D, N_max))

    child_counters = np.zeros(D, dtype=int)
    for k in range(len(x1)):
        d = date_idx[k]
        x1_out[d] = x1[k]  # same for all children on the same date
        n = child_counters[d]
        x2_out[d, n] = x2[k]
        if c2_dim > 0:
            c2_out[d, n] = c2[k]
        mask_out[d, n] = 1.0
        child_counters[d] += 1

    return x1_out, x2_out, c2_out, mask_out


# ---------------------------------------------------------------------------
# Main HVAE data loader
# ---------------------------------------------------------------------------


def load_iv_data_hvae(
    file_path: Path,
    batch_size: int = 256,
    train_split: float = 0.8,
    normalise: bool = True,
    omx_name: str = "OMX",
    value_types: list[str] | None = None,
    column_map: dict[str, str] | None = None,
    c2_mode: str = "full",
):
    """Load implied-volatility surfaces for the hierarchical VAE.

    The loading pipeline mirrors the structure of
    :func:`load_yield_data_cvae`: a chain of pure transform functions is
    applied in sequence to go from the raw CSV to normalised
    ``(x1, c1, x2, c2)`` arrays ready for ``DataLoader`` construction.

    Pipeline overview
    ~~~~~~~~~~~~~~~~~
    1. ``pivot_and_ffill``  – raw CSV → structured surfaces + spot series.
    2. ``_build_rv_dict``   – spot → per-(name, date) ΔRV features.
    3. ``_make_pairs``      – match index with each single name per date.
    4. ``_normalise_paired_splits`` – z-score using training statistics.

    Pairing strategy
    ~~~~~~~~~~~~~~~~
    For each date *t* every single-name observation is paired with the
    single OMX observation on that date.  The OMX row is therefore
    repeated once per available single name, yielding
    ``N_valid_dates × N_single_names`` training samples.

    Pre-processing per surface
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Pivot to wide format  (date × grid of type / strike / tau).
    2. Log-changes via :func:`log_returns`:
       ``Δ_t = log V_t − log V_{t−1}``.
    3. Z-score normalise (train-set statistics applied to both splits).

    c2_mode
    ~~~~~~~
    Controls which conditioning variables appear in **c2**:

    - ``"none"``         – empty (0 columns).
    - ``"vol_shape"``    – t−1 IV surface levels for the single name.
    - ``"vol_shape_rv"`` – t−1 IV levels + ΔRV_1W, ΔRV_1M computed via
      :func:`rv_impact` on the shortest-tenor forward (spot proxy).
    - ``"full"``         – (default) t−1 IV levels + sector one-hot +
      event_flag.

    Parameters
    ----------
    file_path : Path
        CSV with columns ``date, name, strike, tau, type, value``
        (or aliased via *column_map*).
    batch_size : int
    train_split : float
    normalise : bool
    omx_name : str
        Ticker identifying the index surface.
    value_types : list[str] | None
        Filter to these ``type`` values for the grid (e.g. ``["IVS"]``).
    column_map : dict[str, str] | None
        Rename columns before processing
        (e.g. ``{"ticker": "name", "anchor": "date"}``).
    c2_mode : str
        One of ``"none"``, ``"vol_shape"``, ``"vol_shape_rv"``,
        ``"full"``.

    Returns
    -------
    dataloaders : dict
        ``{"train": DataLoader, "test": DataLoader}`` – each batch
        yields ``(x1, c1, x2, c2)``.
    columns : dict
        Grid labels, conditioning names, and dimension info (keys:
        ``grid``, ``c1_names``, ``c2_names``, ``x1_dim``, ``x2_dim``,
        ``c1_dim``, ``c2_dim``).
    processed : dict
        Test-split arrays and inverse-transform lambdas (keys: ``X1``,
        ``C1``, ``X2``, ``C2``, ``get_x1``, ``get_x2``, ``get_c1``,
        ``get_c2``, ``train_names``, ``train_dates``, ``names``,
        ``dates``).
    """
    raw = pd.read_csv(file_path)
    if column_map:
        raw = raw.rename(columns=column_map)

    # --- Transform pipeline (pre-split) ------------------------------------
    transform = [
        lambda d: pivot_and_ffill(d, omx_name, value_types),
        lambda d: {
            **d,
            "rv_dict": (
                _build_rv_dict(d["spot"], d["sn_names"])
                if c2_mode == "vol_shape_rv"
                else {}
            ),
        },
    ]
    surfaces = reduce(lambda acc, fn: fn(acc), transform, raw)

    grid = surfaces["grid"]
    n_grid = surfaces["n_grid"]
    sn_names = surfaces["sn_names"]
    rv_dict = surfaces["rv_dict"]

    # --- Temporal split -----------------------------------------------------
    common = surfaces["common_dates"]
    n_train = int(len(common) * train_split)
    date_splits = {"train": common[:n_train], "test": common[n_train:]}

    # --- Categorical info (only for "full" mode) ----------------------------
    cat_info = _extract_categorical_info(raw)

    # --- Build (x1, c1, x2, c2) pairs per split ----------------------------
    _raw_splits = {
        mode: _make_pairs(
            surfaces["index"],
            surfaces["sn"],
            sn_names,
            dates,
            c2_mode,
            rv_dict,
            cat_info,
        )
        for mode, dates in date_splits.items()
    }
    splits = {m: _raw_splits[m][:4] for m in _raw_splits}
    _meta = {m: _raw_splits[m][4:] for m in _raw_splits}

    # --- Normalise using training statistics --------------------------------
    splits, inv_fns = _normalise_paired_splits(splits, normalise, c2_mode, n_grid)

    # --- Data loaders -------------------------------------------------------
    def make_loader(data, shuffle):
        return DataLoader(
            HierarchicalDataset(*data), batch_size=batch_size, shuffle=shuffle
        )

    dataloaders = {m: make_loader(splits[m], m == "train") for m in splits}

    # --- c2_dim depends on mode ---------------------------------------------
    n_sectors = cat_info.get("n_sectors", 1)
    c1_dim = n_grid
    if c2_mode == "none":
        c2_dim = 0
        c2_names: list[str] = []
    elif c2_mode == "vol_shape":
        c2_dim = n_grid
        c2_names = [f"level_{i}" for i in range(n_grid)]
    elif c2_mode == "vol_shape_rv":
        c2_dim = n_grid + 2
        c2_names = [f"level_{i}" for i in range(n_grid)] + [
            "std_ret_1w",
            "std_ret_1m",
        ]
    else:
        c2_dim = n_grid + n_sectors + 1
        c2_names = (
            [f"level_{i}" for i in range(n_grid)]
            + [f"sector_{i}" for i in range(n_sectors)]
            + ["event_flag"]
        )

    columns = {
        "grid": grid,
        "c1_names": [f"omx_level_{i}" for i in range(n_grid)],
        "c2_names": c2_names,
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
        "get_x1": inv_fns["x1"],
        "get_x2": inv_fns["x2"],
        "get_c1": inv_fns["c1"],
        "get_c2": inv_fns["c2"],
        "train_names": _meta["train"][0],
        "train_dates": _meta["train"][1],
        "names": _meta["test"][0],
        "dates": _meta["test"][1],
    }

    return dataloaders, columns, processed


def make_datebatched_loaders(
    flat_loaders: dict,
    processed: dict,
    batch_size: int = 16,
) -> dict:
    """Convert flat-pair HVAE loaders to date-batched format for ``hvae.py``.

    Parameters
    ----------
    flat_loaders : dict returned by :func:`load_iv_data_hvae`
    processed    : dict returned by :func:`load_iv_data_hvae` (with metadata)
    batch_size   : int  (number of *dates* per batch)

    Returns
    -------
    dict  {"train": DataLoader, "test": DataLoader}
          each batch yields ``(x1, x2, c2, child_mask)`` with shapes
          ``(B, x1_dim)``, ``(B, N_max, x2_dim)``, ``(B, N_max, c2_dim)``,
          ``(B, N_max)``.
    """
    out = {}
    for mode in ("train", "test"):
        ds = flat_loaders[mode].dataset
        x1 = ds.x1.numpy()
        x2 = ds.x2.numpy()
        c2 = ds.c2.numpy()
        dates_key = "train_dates" if mode == "train" else "dates"
        dates = processed[dates_key]
        x1_db, x2_db, c2_db, mask_db = _flat_to_date_batched(x1, x2, c2, dates)
        out[mode] = DataLoader(
            DateBatchedHierarchicalDataset(x1_db, x2_db, c2_db, mask_db),
            batch_size=batch_size,
            shuffle=(mode == "train"),
        )
    return out
