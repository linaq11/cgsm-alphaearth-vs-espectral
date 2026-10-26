#!/usr/bin/env python
"""
Build the spatial train/test split for the CGSM RF-vs-AlphaEarth benchmark.

Produces ONE assignment file that both feature arms consume, so the comparison
is paired at the point level.

Design (see BENCHMARK_DESIGN.md §2):
  * one contiguous spatial holdout, ~22% of labels, all four classes, defined
    once and shared by every regime, touched once at the end;
  * k-fold assignment of whole blocks on the remaining labels;
  * a train-side buffer of L/2 applied at training time, not baked into the
    file, so the buffer width stays inspectable.

Three regimes are emitted rather than one, because no single block size is
defensible here: the NDVI and AlphaEarth-PC1 variograms never reach a sill in
either the raw or the detrended series, while the minor components reach one
only in specific series - PC2 at 10.0 km raw and 7.2 km detrended, PC3 at 9.4 km
detrended but never raw. Blocks wide enough for the widest of those correlation
lengths leave too few blocks to form five folds. Reporting
random / 6 km / 16 km turns an unresolvable parameter choice into a sensitivity
analysis: if the RF-vs-AlphaEarth ranking is stable across all three, the
conclusion does not depend on the block size.

Usage
-----
python build_split.py --labels alphaearth2021_etiquetas.csv --out split
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyproj

    _pd = pyproj.datadir.get_data_dir()
    os.environ.setdefault("PROJ_DATA", _pd)
    os.environ.setdefault("PROJ_LIB", _pd)
except Exception:  # pragma: no cover
    pass

WORK_CRS = "EPSG:9377"          # MAGNA-SIRGAS 2018 / Origen-Nacional, metres
L_PRIMARY = 6_000               # m, primary block edge
L_CONSERVATIVE = 16_000         # m, conservative sensitivity block edge
K_FOLDS = 5
HOLDOUT_TARGET = 0.22           # fraction of labels
SEED = 42


# --------------------------------------------------------------------------
def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"lon", "lat", "clase", "clase_nombre"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"labels file is missing columns: {sorted(missing)}")
    tr = pyproj.Transformer.from_crs("EPSG:4326", WORK_CRS, always_xy=True)
    df["x"], df["y"] = tr.transform(df.lon.values, df.lat.values)
    df["point_id"] = np.arange(len(df))
    return df


def block_ids(xy: np.ndarray, L: float):
    """Integer block index per point, plus the unique block table."""
    bid = np.floor(xy / L).astype(np.int64)
    keys, inv = np.unique(bid, axis=0, return_inverse=True)
    return keys, inv


# --------------------------------------------------------------------------
def carve_holdout(df: pd.DataFrame, L: float = L_PRIMARY,
                  target: float = HOLDOUT_TARGET, seed: int = SEED) -> np.ndarray:
    """Grow a spatially contiguous region of blocks to ~`target` of the labels.

    Contiguity matters: a holdout scattered across the AOI measures
    interpolation, whereas the question is whether the model transfers to ground
    it has never seen. Grown from every candidate seed; the region closest to the
    target fraction that still contains all four classes wins.
    """
    xy = df[["x", "y"]].to_numpy()
    cls = df["clase_nombre"].to_numpy()
    n_cls = len(np.unique(cls))
    keys, inv = block_ids(xy, L)
    counts = np.bincount(inv, minlength=len(keys))
    kset = {tuple(k): i for i, k in enumerate(keys)}

    best = None
    for s in range(len(keys)):
        region = {s}
        frontier = {s}
        while True:
            frac = counts[list(region)].sum() / len(df)
            if frac >= target:
                break
            # candidate neighbours of the current region (8-connectivity)
            cand = set()
            for bi in frontier:
                kx, ky = keys[bi]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        j = kset.get((kx + dx, ky + dy))
                        if j is not None and j not in region:
                            cand.add(j)
            if not cand:
                break
            # keep the region compact: take the candidate nearest its centroid
            cen = keys[list(region)].mean(axis=0)
            nxt = min(cand, key=lambda j: ((keys[j] - cen) ** 2).sum())
            region.add(nxt)
            frontier = region
        mask = np.isin(inv, list(region))
        frac = mask.sum() / len(df)
        if len(np.unique(cls[mask])) < n_cls:
            continue
        score = abs(frac - target)
        if best is None or score < best[0]:
            best = (score, mask.copy(), sorted(region), frac)
    if best is None:
        sys.exit("could not grow a contiguous holdout containing all four classes")
    print(f"holdout: {best[1].sum()} points ({100 * best[3]:.1f}%), "
          f"{len(best[2])} blocks of {L / 1000:.0f} km")
    return best[1]


# --------------------------------------------------------------------------
def assign_blocked_folds(df: pd.DataFrame, mask_pool: np.ndarray, L: float,
                         k: int = K_FOLDS, seed: int = SEED) -> np.ndarray:
    """Assign whole blocks to folds, balancing class counts across folds.

    Greedy: biggest blocks first, each to the fold that currently minimises the
    resulting class imbalance. Whole blocks move together, which is the property
    that makes the folds spatially independent.
    """
    fold = np.full(len(df), -1, int)
    idx = np.nonzero(mask_pool)[0]
    xy = df[["x", "y"]].to_numpy()[idx]
    cls = df["clase_nombre"].to_numpy()[idx]
    classes = np.unique(df["clase_nombre"])
    cmap = {c: i for i, c in enumerate(classes)}

    keys, inv = block_ids(xy, L)
    nb = len(keys)
    vecs = np.zeros((nb, len(classes)))
    for bi in range(nb):
        for c in cls[inv == bi]:
            vecs[bi, cmap[c]] += 1
    sizes = vecs.sum(1)

    # Longest-processing-time-first: largest blocks placed first, each into the
    # fold that is currently smallest. Balancing totals is the primary objective
    # (a fold with 36 training points cannot support a model), class evenness the
    # secondary one.
    order = np.argsort(-sizes)
    counts = np.zeros((k, len(classes)))
    assign = np.full(nb, -1, int)
    for bi in order:
        tot = counts.sum(1) + sizes[bi]
        imb = (counts + vecs[bi]).max(1) - (counts + vecs[bi]).min(1)
        f = int(np.argmin(tot + 0.5 * imb))
        counts[f] += vecs[bi]
        assign[bi] = f

    # Repair pass: every fold must contain all classes when used as a test set,
    # otherwise per-class metrics are undefined for that fold. Move a block that
    # supplies the missing class from a fold that can spare it.
    for _ in range(200):
        missing = [(f, c) for f in range(k) for c in range(len(classes))
                   if counts[f, c] == 0]
        if not missing:
            break
        f, c = missing[0]
        donors = [bi for bi in range(nb)
                  if vecs[bi, c] > 0 and assign[bi] != f
                  and counts[assign[bi], c] - vecs[bi, c] > 0]
        if not donors:
            break
        bi = min(donors, key=lambda b: sizes[b])
        counts[assign[bi]] -= vecs[bi]
        counts[f] += vecs[bi]
        assign[bi] = f
    else:
        pass
    still = [(int(f), classes[c]) for f in range(k) for c in range(len(classes))
             if counts[f, c] == 0]
    if still:
        print(f"  WARNING L={L / 1000:.0f} km, k={k}: folds still missing classes "
              f"{still} - blocks are too coarse to cover every class in every fold")

    for bi in range(nb):
        fold[idx[inv == bi]] = assign[bi]
    return fold


def assign_random_folds(df: pd.DataFrame, mask_pool: np.ndarray,
                        k: int = K_FOLDS, seed: int = SEED) -> np.ndarray:
    """Class-stratified random point folds — the deliberately leaky baseline."""
    from sklearn.model_selection import StratifiedKFold

    fold = np.full(len(df), -1, int)
    idx = np.nonzero(mask_pool)[0]
    y = df["clase_nombre"].to_numpy()[idx]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(idx)), y)):
        fold[idx[te]] = f
    return fold


# --------------------------------------------------------------------------
def buffered_train_mask(df: pd.DataFrame, fold: np.ndarray, test_fold: int,
                        buffer_m: float, pool: np.ndarray) -> np.ndarray:
    """Training points of `pool`, excluding any within `buffer_m` of a test point.

    Kept as a function rather than a column so the buffer width remains a visible
    parameter of the analysis instead of a baked-in decision.
    """
    xy = df[["x", "y"]].to_numpy()
    te = pool & (fold == test_fold)
    tr = pool & (fold != test_fold) & (fold >= 0)
    if te.sum() == 0 or tr.sum() == 0:
        return np.zeros(len(df), bool)
    d = np.sqrt(((xy[tr][:, None, :] - xy[te][None, :, :]) ** 2).sum(-1)).min(1)
    keep = np.zeros(len(df), bool)
    keep[np.nonzero(tr)[0][d > buffer_m]] = True
    return keep


def summarise(df: pd.DataFrame, regimes: dict, pool: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, (fold, buf, k) in regimes.items():
        for f in range(k):
            te = pool & (fold == f)
            keep = buffered_train_mask(df, fold, f, buf, pool)
            if te.sum() == 0:
                continue
            xy = df[["x", "y"]].to_numpy()
            d = (np.sqrt(((xy[keep][:, None, :] - xy[te][None, :, :]) ** 2).sum(-1)).min()
                 if keep.any() else np.nan)
            rows.append(dict(
                regime=name, fold=f, buffer_m=buf,
                n_test=int(te.sum()), n_train=int(keep.sum()),
                n_train_unbuffered=int((pool & (fold != f) & (fold >= 0)).sum()),
                min_train_test_m=float(d),
                classes_test=int(df.loc[te, "clase_nombre"].nunique()),
                classes_train=int(df.loc[keep, "clase_nombre"].nunique())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", default=Path("split"), type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df = load_labels(a.labels)
    hold = carve_holdout(df)
    pool = ~hold

    # k is per regime: 16 km blocks yield only ~16 blocks over the pool, so five
    # folds would leave ~3 blocks each and folds too small to train on.
    regimes = {
        "random": (assign_random_folds(df, pool, k=K_FOLDS), 0.0, K_FOLDS),
        f"block_{L_PRIMARY // 1000}km":
            (assign_blocked_folds(df, pool, L_PRIMARY, k=K_FOLDS), L_PRIMARY / 2, K_FOLDS),
        f"block_{L_CONSERVATIVE // 1000}km":
            (assign_blocked_folds(df, pool, L_CONSERVATIVE, k=3), L_CONSERVATIVE / 2, 3),
    }

    out = df[["point_id", "lon", "lat", "x", "y", "clase", "clase_nombre"]].copy()
    out["holdout"] = hold
    for name, (fold, _, _) in regimes.items():
        out[f"fold_{name}"] = fold
    out.to_csv(a.out / "split_assignments.csv", index=False)

    summ = summarise(df, regimes, pool)
    summ.to_csv(a.out / "split_summary.csv", index=False)

    meta = {"work_crs": WORK_CRS, "seed": SEED,
            "holdout_fraction": float(hold.mean()),
            "n_holdout": int(hold.sum()), "n_pool": int(pool.sum()),
            "regimes": {n: {"buffer_m": b, "k_folds": k}
                        for n, (_, b, k) in regimes.items()}}
    (a.out / "split_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {a.out}/split_assignments.csv, split_summary.csv, split_meta.json")
    print(summ.to_string(index=False))
    # sanity: no point in two folds, holdout never assigned a fold
    for name, (fold, _, _) in regimes.items():
        assert (fold[hold] == -1).all(), f"{name}: holdout point got a fold"
        assert (fold[pool] >= 0).all(), f"{name}: pool point without a fold"
    print("\nchecks passed: holdout disjoint from all folds; every pool point assigned")


if __name__ == "__main__":
    main()
