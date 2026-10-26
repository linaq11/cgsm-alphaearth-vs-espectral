#!/usr/bin/env python
"""
RF on Sentinel-2 spectral features vs RF on AlphaEarth embeddings, CGSM 2021.

Both arms use the SAME learner (Random Forest) and the SAME points and folds, so
what is compared is the feature set, not the classifier. Evaluated under three
split regimes (random / 6 km blocks / 16 km blocks) plus one contiguous spatial
holdout, per BENCHMARK_DESIGN.md §2.

Hyperparameters are selected inside the training folds only, by an inner CV that
is grouped by block for the blocked regimes, so tuning cannot see the outer test
fold or its spatial neighbourhood.

Usage
-----
python run_benchmark.py --spectral spectral_features.csv \
    --embedding alphaearth2021_etiquetas.csv \
    --split split/split_assignments.csv --out results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold, StratifiedKFold

SEED = 42
N_TREES = 500
GRID = [{"max_features": mf, "min_samples_leaf": ml}
        for mf in ("sqrt", 0.5) for ml in (1, 2)]
SPECTRAL_FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "MNDWI"]
EMB_FEATURES = [f"A{i:02d}" for i in range(64)]
REGIMES = {"random": 0.0, "block_6km": 3000.0, "block_16km": 8000.0}
BLOCK_L = {"random": None, "block_6km": 6000.0, "block_16km": 16000.0}


# --------------------------------------------------------------------------
def load(spectral: Path, embedding: Path, split: Path):
    sp = pd.read_csv(split)
    sf = pd.read_csv(spectral)
    em = pd.read_csv(embedding)
    if not (len(sp) == len(sf) == len(em)):
        sys.exit(f"row counts differ: split {len(sp)}, spectral {len(sf)}, "
                 f"embedding {len(em)}")
    em = em.reset_index(drop=True)
    em["point_id"] = np.arange(len(em))
    df = sp.merge(sf.drop(columns=["x", "y", "clase", "clase_nombre"]),
                  on="point_id", validate="1:1")
    df = df.merge(em[["point_id", "clase_nombre"] + EMB_FEATURES],
                  on="point_id", suffixes=("", "_emb"), validate="1:1")
    # alignment check: the class must agree across all three sources
    bad = int((df["clase_nombre"] != df["clase_nombre_emb"]).sum())
    if bad:
        sys.exit(f"{bad} points disagree on class between split and embedding tables")
    df = df.drop(columns=["clase_nombre_emb"])
    for c in SPECTRAL_FEATURES + EMB_FEATURES:
        if c not in df.columns:
            sys.exit(f"missing feature column {c}")
    assert df[SPECTRAL_FEATURES + EMB_FEATURES].notna().all().all()
    print(f"aligned table: {len(df)} points, "
          f"{len(SPECTRAL_FEATURES)} spectral + {len(EMB_FEATURES)} embedding features")
    return df


def buffered_train(df: pd.DataFrame, fold: np.ndarray, test_fold: int,
                   buffer_m: float, pool: np.ndarray) -> np.ndarray:
    xy = df[["x", "y"]].to_numpy()
    te = pool & (fold == test_fold)
    tr = pool & (fold != test_fold) & (fold >= 0)
    if buffer_m <= 0:
        return tr
    d = np.sqrt(((xy[tr][:, None, :] - xy[te][None, :, :]) ** 2).sum(-1)).min(1)
    keep = np.zeros(len(df), bool)
    keep[np.nonzero(tr)[0][d > buffer_m]] = True
    return keep


def tune_and_fit(X, y, xy, L, seed=SEED):
    """Pick hyperparameters by inner CV grouped by block, then refit on all of X."""
    if len(np.unique(y)) < 2 or len(y) < 20:
        best = GRID[0]
    else:
        if L is None:
            splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
            folds = list(splitter.split(X, y))
        else:
            g = np.floor(xy / L).astype(np.int64)
            _, gid = np.unique(g, axis=0, return_inverse=True)
            n_g = len(np.unique(gid))
            if n_g < 3:
                folds = list(StratifiedKFold(3, shuffle=True,
                                             random_state=seed).split(X, y))
            else:
                folds = list(GroupKFold(n_splits=min(3, n_g)).split(X, y, groups=gid))
        scores = []
        for p in GRID:
            s = []
            for tr_i, te_i in folds:
                if len(np.unique(y[tr_i])) < 2:
                    continue
                m = RandomForestClassifier(n_estimators=200, random_state=seed,
                                           n_jobs=-1, **p).fit(X[tr_i], y[tr_i])
                s.append(f1_score(y[te_i], m.predict(X[te_i]), average="macro",
                                  zero_division=0))
            scores.append(np.mean(s) if s else -1)
        best = GRID[int(np.argmax(scores))]
    model = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                   n_jobs=-1, **best).fit(X, y)
    return model, best


# --------------------------------------------------------------------------
def run_cv(df, arms, classes, out: Path):
    pool = ~df["holdout"].to_numpy()
    xy = df[["x", "y"]].to_numpy()
    y_all = df["clase_nombre"].to_numpy()
    rows, per_fold, preds = [], [], []

    for regime, buf in REGIMES.items():
        fold = df[f"fold_{regime}"].to_numpy()
        L = BLOCK_L[regime]
        for arm, feats in arms.items():
            X_all = df[feats].to_numpy()
            yt, yp, fid = [], [], []
            for f in sorted(set(fold[fold >= 0])):
                tr = buffered_train(df, fold, f, buf, pool)
                te = pool & (fold == f)
                if tr.sum() < 20 or te.sum() == 0:
                    continue
                model, best = tune_and_fit(X_all[tr], y_all[tr], xy[tr], L)
                p = model.predict(X_all[te])
                yt.append(y_all[te]); yp.append(p); fid.append(np.full(te.sum(), f))
                per_fold.append(dict(regime=regime, arm=arm, fold=int(f),
                                     n_train=int(tr.sum()), n_test=int(te.sum()),
                                     macro_f1=f1_score(y_all[te], p, average="macro",
                                                       zero_division=0),
                                     oa=float((y_all[te] == p).mean()),
                                     **{f"hp_{k}": v for k, v in best.items()}))
            yt = np.concatenate(yt); yp = np.concatenate(yp); fid = np.concatenate(fid)
            preds.append(pd.DataFrame(dict(regime=regime, arm=arm, fold=fid,
                                          y_true=yt, y_pred=yp)))
            pr, rc, f1, sup = precision_recall_fscore_support(
                yt, yp, labels=classes, zero_division=0)
            rows.append(dict(regime=regime, arm=arm, n=len(yt),
                             oa=float((yt == yp).mean()),
                             kappa=cohen_kappa_score(yt, yp),
                             macro_f1=f1_score(yt, yp, average="macro", zero_division=0),
                             **{f"f1_{c}": v for c, v in zip(classes, f1)},
                             **{f"recall_{c}": v for c, v in zip(classes, rc)},
                             **{f"precision_{c}": v for c, v in zip(classes, pr)}))
            np.savetxt(out / f"confusion_{regime}_{arm}.csv",
                       confusion_matrix(yt, yp, labels=classes), fmt="%d", delimiter=",",
                       header=",".join(classes), comments="")
    return pd.DataFrame(rows), pd.DataFrame(per_fold), pd.concat(preds, ignore_index=True)


def run_holdout(df, arms, classes, out: Path, buffer_m=3000.0):
    """Train on the buffered pool, evaluate once on the contiguous holdout."""
    hold = df["holdout"].to_numpy()
    xy = df[["x", "y"]].to_numpy()
    y_all = df["clase_nombre"].to_numpy()
    d = np.sqrt(((xy[~hold][:, None, :] - xy[hold][None, :, :]) ** 2).sum(-1)).min(1)
    tr = np.zeros(len(df), bool)
    tr[np.nonzero(~hold)[0][d > buffer_m]] = True
    print(f"holdout: train on {tr.sum()} buffered pool points "
          f"(of {(~hold).sum()}), test on {hold.sum()}")

    rows, pred = [], {}
    for arm, feats in arms.items():
        X = df[feats].to_numpy()
        model, best = tune_and_fit(X[tr], y_all[tr], xy[tr], BLOCK_L["block_6km"])
        p = model.predict(X[hold])
        pred[arm] = p
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_all[hold], p, labels=classes, zero_division=0)
        rows.append(dict(arm=arm, n=int(hold.sum()), n_train=int(tr.sum()),
                         oa=float((y_all[hold] == p).mean()),
                         kappa=cohen_kappa_score(y_all[hold], p),
                         macro_f1=f1_score(y_all[hold], p, average="macro",
                                           zero_division=0),
                         **{f"f1_{c}": v for c, v in zip(classes, f1)},
                         **{f"recall_{c}": v for c, v in zip(classes, rc)},
                         **{"hp_" + k: v for k, v in best.items()}))
        np.savetxt(out / f"confusion_holdout_{arm}.csv",
                   confusion_matrix(y_all[hold], p, labels=classes), fmt="%d",
                   delimiter=",", header=",".join(classes), comments="")

    # paired McNemar on identical test pixels (exact binomial on discordant pairs)
    a, b = list(arms)
    ca = pred[a] == y_all[hold]
    cb = pred[b] == y_all[hold]
    n01 = int((~ca & cb).sum())   # only b correct
    n10 = int((ca & ~cb).sum())   # only a correct
    bt = binomtest(n10, n10 + n01, 0.5) if (n10 + n01) > 0 else None
    mc = {"arm_a": a, "arm_b": b, "only_a_correct": n10, "only_b_correct": n01,
          "both_correct": int((ca & cb).sum()), "both_wrong": int((~ca & ~cb).sum()),
          "p_value": (bt.pvalue if bt else None),
          "test": "exact binomial McNemar on discordant pairs",
          "n_discordant": n10 + n01}
    hold_pred = pd.DataFrame({"point_id": df.loc[hold, "point_id"].to_numpy(),
                              "y_true": y_all[hold],
                              **{f"pred_{k}": v for k, v in pred.items()}})
    return pd.DataFrame(rows), mc, hold_pred


def residual_range(df, preds, out: Path):
    """Autocorrelation of the misclassification indicator (Roberts et al. 2017).

    What must be independent across folds is the model residual, not the
    predictor. If the residual's correlation length is below the block edge, the
    6 km choice is justified after the fact.
    """
    sys.path.insert(0, ".")
    from inspect_data import _variogram_core

    res = {}
    sub = preds[(preds.regime == "block_6km")]
    for arm in sub.arm.unique():
        s = sub[sub.arm == arm]
        # run_cv concatenated test folds in ascending fold order, so rebuilding
        # the same index order recovers each prediction's coordinates
        pool = ~df["holdout"].to_numpy()
        fold = df["fold_block_6km"].to_numpy()
        idx = np.concatenate([np.nonzero(pool & (fold == f))[0]
                              for f in sorted(set(fold[fold >= 0]))])
        err = (s.y_true.to_numpy() != s.y_pred.to_numpy()).astype(float)
        assert len(idx) == len(err), f"{arm}: {len(idx)} coords vs {len(err)} residuals"
        assert (df["clase_nombre"].to_numpy()[idx] == s.y_true.to_numpy()).all(), \
            f"{arm}: coordinate/prediction order mismatch"
        xy = df[["x", "y"]].to_numpy()[idx]
        vg = _variogram_core(err, xy, n_lags=14, max_lag_pct=30.0, detrend=False)
        res[arm] = {k: v for k, v in vg.items() if k != "n_pairs"} if vg else None
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spectral", required=True, type=Path)
    ap.add_argument("--embedding", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--out", default=Path("results"), type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df = load(a.spectral, a.embedding, a.split)
    classes = sorted(df["clase_nombre"].unique())
    arms = {"spectral": SPECTRAL_FEATURES, "alphaearth": EMB_FEATURES}

    cv, per_fold, preds = run_cv(df, arms, classes, a.out)
    cv.to_csv(a.out / "cv_metrics.csv", index=False)
    per_fold.to_csv(a.out / "cv_per_fold.csv", index=False)
    preds.to_csv(a.out / "cv_predictions.csv", index=False)

    hold, mc, hold_pred = run_holdout(df, arms, classes, a.out)
    hold.to_csv(a.out / "holdout_metrics.csv", index=False)
    hold_pred.to_csv(a.out / "holdout_predictions.csv", index=False)

    rr = residual_range(df, preds, a.out)
    (a.out / "mcnemar_holdout.json").write_text(json.dumps(mc, indent=2))
    (a.out / "residual_autocorrelation.json").write_text(
        json.dumps(rr, indent=2, default=float))

    print("\n== cross-validation (pooled over folds) ==")
    print(cv[["regime", "arm", "n", "oa", "kappa", "macro_f1"]].round(4).to_string(index=False))
    print("\n== per-fold macro-F1 (mean +/- sd) ==")
    print(per_fold.groupby(["regime", "arm"]).macro_f1.agg(["mean", "std", "count"])
          .round(4).to_string())
    print("\n== contiguous spatial holdout ==")
    print(hold[["arm", "n", "n_train", "oa", "kappa", "macro_f1"]].round(4).to_string(index=False))
    print("\nMcNemar:", json.dumps(mc))
    print("\nresidual autocorrelation (block_6km):",
          {k: (f"range={v['practical_range_m']}, knee={v['knee_m']:.0f} m"
               if v else None) for k, v in rr.items()})


if __name__ == "__main__":
    main()
