#!/usr/bin/env python
"""
Inspection report for a Sentinel-2 raster + labeled point set.

Reports everything needed before designing a spatially-blocked benchmark:
raster CRS / extent / bands / dtype / nodata, label class balance, and the
label-geometry diagnostics that decide whether a split is honest
(pixel-level duplicates, nearest-neighbour spacing, autocorrelation range).

Usage
-----
python inspect_data.py --raster S2_cgsm.tif --labels labels.geojson \
    [--label-field clase] [--out inspection] [--no-variogram]

Writes <out>/inspection_report.md, <out>/inspection_report.json and
diagnostic PNGs. Nothing is modified on disk apart from <out>/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# Some conda builds leave PROJ_DATA unset, so GDAL cannot find proj.db and every
# CRS lookup fails. Resolve it from pyproj before rasterio initialises GDAL.
try:
    import pyproj

    _pd = pyproj.datadir.get_data_dir()
    os.environ.setdefault("PROJ_DATA", _pd)
    os.environ.setdefault("PROJ_LIB", _pd)
except Exception:  # pragma: no cover - fall through to rasterio's own defaults
    pass

try:
    import rasterio
    from rasterio.warp import transform_bounds
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    sys.exit(f"missing dependency: {exc}. Install the geo-bench environment.")

# Sentinel-2 L2A band-name hints, used only to annotate the report.
S2_HINTS = {
    "B1": "443 nm coastal aerosol (60 m)", "B2": "490 nm blue (10 m)",
    "B3": "560 nm green (10 m)", "B4": "665 nm red (10 m)",
    "B5": "705 nm red-edge (20 m)", "B6": "740 nm red-edge (20 m)",
    "B7": "783 nm red-edge (20 m)", "B8": "842 nm NIR (10 m)",
    "B8A": "865 nm narrow NIR (20 m)", "B9": "945 nm water vapour (60 m)",
    "B11": "1610 nm SWIR1 (20 m)", "B12": "2190 nm SWIR2 (20 m)",
    "SCL": "scene classification layer",
}


def jsonable(obj):
    """numpy scalars -> python, for json.dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.ndarray,)):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


# --------------------------------------------------------------------------
# raster
# --------------------------------------------------------------------------
def inspect_raster(path: Path, max_sample: int = 2_000_000) -> dict:
    with rasterio.open(path) as src:
        px, py = src.res
        bounds = src.bounds
        info = {
            "path": str(path),
            "driver": src.driver,
            "crs": str(src.crs),
            "crs_epsg": src.crs.to_epsg() if src.crs else None,
            "crs_is_projected": bool(src.crs.is_projected) if src.crs else None,
            "crs_units": src.crs.linear_units if (src.crs and src.crs.is_projected) else "degrees",
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtypes": list(src.dtypes),
            "nodata": src.nodata,
            "nodatavals": list(src.nodatavals),
            "pixel_size": [abs(px), abs(py)],
            "transform": list(src.transform)[:6],
            "bounds_native": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "block_shapes": [list(b) for b in src.block_shapes],
            "compression": str(src.compression) if src.compression else None,
            "interleave": src.interleaving.name if src.interleaving else None,
            "overview_levels": src.overviews(1),
            "tags": dict(src.tags()),
            "band_descriptions": [src.descriptions[i] for i in range(src.count)],
        }
        # extent in WGS84 + projected extent in km
        if src.crs:
            info["bounds_wgs84"] = list(
                transform_bounds(src.crs, "EPSG:4326", *bounds, densify_pts=21)
            )
        if src.crs and src.crs.is_projected:
            info["extent_km"] = [
                (bounds.right - bounds.left) / 1000.0,
                (bounds.top - bounds.bottom) / 1000.0,
            ]
            info["area_km2"] = (
                (bounds.right - bounds.left) * (bounds.top - bounds.bottom) / 1e6
            )

        # decimated read so a large scene still fits in memory
        n_full = src.width * src.height
        step = max(1, int(math.ceil(math.sqrt(n_full / max_sample))))
        out_shape = (src.count, max(1, src.height // step), max(1, src.width // step))
        arr = src.read(out_shape=out_shape, masked=True).astype("float64")
        info["stats_decimation_step"] = step
        info["stats_sampled_pixels"] = int(arr.shape[1] * arr.shape[2])

        bands = []
        for i in range(src.count):
            b = arr[i]
            valid = b.compressed()
            valid = valid[np.isfinite(valid)]
            desc = src.descriptions[i]
            rec = {
                "index": i + 1,
                "description": desc,
                "hint": S2_HINTS.get(str(desc).upper().replace("_", ""), None),
                "dtype": src.dtypes[i],
                "nodata": src.nodatavals[i],
                "valid_fraction": float(valid.size / b.size) if b.size else 0.0,
            }
            if valid.size:
                qs = np.percentile(valid, [1, 25, 50, 75, 99])
                rec.update(
                    min=float(valid.min()), max=float(valid.max()),
                    mean=float(valid.mean()), std=float(valid.std()),
                    p1=float(qs[0]), p25=float(qs[1]), p50=float(qs[2]),
                    p75=float(qs[3]), p99=float(qs[4]),
                )
            bands.append(rec)
        info["bands"] = bands

        # scaling guess: L2A DN are ints ~0-10000, reflectance floats are 0-1
        finite_max = max((b.get("max", 0) or 0) for b in bands) if bands else 0
        if finite_max > 1.5:
            info["scaling_guess"] = (
                f"values reach {finite_max:g}: looks like L2A DN, divide by 10000 for reflectance"
            )
        else:
            info["scaling_guess"] = "values <= 1.5: looks like reflectance already scaled"
    return info


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def guess_label_field(gdf, explicit=None) -> str:
    if explicit:
        if explicit not in gdf.columns:
            sys.exit(f"--label-field {explicit!r} not in {list(gdf.columns)}")
        return explicit
    prefer = ["class", "clase", "label", "landcover", "cobertura", "cover", "tipo", "categoria", "cl"]
    lowered = {c.lower(): c for c in gdf.columns if c != gdf.geometry.name}
    for p in prefer:
        if p in lowered:
            return lowered[p]
    # fall back to the first low-cardinality non-geometry column
    for c in gdf.columns:
        if c == gdf.geometry.name:
            continue
        if gdf[c].nunique(dropna=True) <= max(12, int(0.05 * len(gdf))):
            return c
    sys.exit(f"could not identify a class column; pass --label-field. columns={list(gdf.columns)}")


def inspect_labels(path: Path, raster_info: dict, label_field=None) -> dict:
    gdf = gpd.read_file(path)
    field = guess_label_field(gdf, label_field)
    info = {
        "path": str(path),
        "n_features": int(len(gdf)),
        "crs": str(gdf.crs),
        "crs_epsg": gdf.crs.to_epsg() if gdf.crs else None,
        "columns": [c for c in gdf.columns],
        "geometry_types": sorted(gdf.geom_type.dropna().unique().tolist()),
        "label_field": field,
        "n_missing_label": int(gdf[field].isna().sum()),
        "n_invalid_geometry": int((~gdf.geometry.is_valid).sum()),
        "n_empty_geometry": int(gdf.geometry.is_empty.sum()),
    }
    counts = gdf[field].value_counts(dropna=False)
    info["class_counts"] = {str(k): int(v) for k, v in counts.items()}
    info["class_proportions"] = {str(k): float(v / len(gdf)) for k, v in counts.items()}
    info["n_classes"] = int(counts.size)
    info["imbalance_ratio"] = float(counts.max() / counts.min()) if counts.size else None

    # reproject to raster CRS for all geometric work
    rcrs = raster_info["crs"]
    g = gdf.to_crs(rcrs) if (gdf.crs and str(gdf.crs) != rcrs) else gdf.copy()
    info["reprojected_to_raster_crs"] = bool(gdf.crs and str(gdf.crs) != rcrs)
    pts = g.geometry.representative_point()
    xy = np.column_stack([pts.x.to_numpy(), pts.y.to_numpy()])

    left, bottom, right, top = raster_info["bounds_native"]
    inside = (xy[:, 0] >= left) & (xy[:, 0] <= right) & (xy[:, 1] >= bottom) & (xy[:, 1] <= top)
    info["n_outside_raster_bbox"] = int((~inside).sum())

    # pixel indices -> duplicate / conflicting labels in one pixel
    with rasterio.open(raster_info["path"]) as src:
        # rowcol vectorises over coordinate arrays; DatasetReader.index does not.
        rows, cols = rasterio.transform.rowcol(src.transform, xy[:, 0], xy[:, 1])
        rows = np.asarray(rows).ravel()
        cols = np.asarray(cols).ravel()
        sampled = np.array(
            [v for v in src.sample(zip(xy[:, 0], xy[:, 1]), masked=True)], dtype=object
        ) if len(xy) else np.array([])

    labels = gdf[field].astype(str).to_numpy()
    key = np.array([f"{r}_{c}" for r, c in zip(rows, cols)])
    uniq, first_idx, inv, cnt = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
    info["n_unique_pixels"] = int(uniq.size)
    info["n_points_sharing_a_pixel"] = int((cnt[inv] > 1).sum())
    conflicts = 0
    for u in uniq[cnt > 1]:
        m = key == u
        if np.unique(labels[m]).size > 1:
            conflicts += 1
    info["n_pixels_with_conflicting_labels"] = int(conflicts)

    # points on nodata / masked pixels
    if len(sampled):
        allmask = []
        for v in sampled:
            v = np.ma.asarray(v)
            allmask.append(bool(np.ma.getmaskarray(v).any()))
        info["n_points_on_nodata"] = int(np.sum(allmask))

    # nearest-neighbour spacing: exposes clumped digitising
    nn = nearest_neighbour_distances(xy)
    if nn is not None:
        qs = np.percentile(nn, [5, 25, 50, 75, 95])
        info["nn_distance_m"] = {
            "min": float(nn.min()), "p5": float(qs[0]), "p25": float(qs[1]),
            "median": float(qs[2]), "p75": float(qs[3]), "p95": float(qs[4]),
            "max": float(nn.max()),
        }
        px = raster_info["pixel_size"][0]
        info["nn_within_1_pixel_fraction"] = float((nn <= px).mean())
        info["nn_within_10_pixels_fraction"] = float((nn <= 10 * px).mean())

    # per-class spatial spread
    spread = {}
    for cls in np.unique(labels):
        m = labels == cls
        if m.sum() >= 2:
            sx, sy = xy[m, 0], xy[m, 1]
            spread[str(cls)] = {
                "n": int(m.sum()),
                "bbox_km": [float((sx.max() - sx.min()) / 1000), float((sy.max() - sy.min()) / 1000)],
                "centroid": [float(sx.mean()), float(sy.mean())],
            }
    info["per_class_spread"] = spread
    return info, g, xy, labels, field


def nearest_neighbour_distances(xy: np.ndarray):
    """1-NN distance per point. Brute force in blocks; fine to ~50k points."""
    n = len(xy)
    if n < 2:
        return None
    if n > 60000:
        rng = np.random.default_rng(0)
        xy = xy[rng.choice(n, 60000, replace=False)]
        n = len(xy)
    out = np.empty(n)
    block = 2000
    for s in range(0, n, block):
        e = min(s + block, n)
        d = np.sqrt(((xy[s:e, None, :] - xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d[:, s:e], np.inf)
        out[s:e] = d.min(axis=1)
    return out


# --------------------------------------------------------------------------
# autocorrelation range (drives the block size of the spatial split)
# --------------------------------------------------------------------------
def empirical_variogram(path: Path, red_band: int, nir_band: int,
                        n_sample: int = 4000, n_lags: int = 18, seed: int = 0):
    """Binned semivariance of NDVI over random valid pixels.

    The lag at which semivariance plateaus is the practical autocorrelation
    range; spatial-CV blocks must be at least this wide.
    """
    rng = np.random.default_rng(seed)
    with rasterio.open(path) as src:
        if max(red_band, nir_band) > src.count:
            return None
        step = max(1, int(math.ceil(math.sqrt(src.width * src.height / 4_000_000))))
        out_shape = (max(1, src.height // step), max(1, src.width // step))
        red = src.read(red_band, out_shape=out_shape, masked=True).astype("float64")
        nir = src.read(nir_band, out_shape=out_shape, masked=True).astype("float64")
        ndvi = (nir - red) / (nir + red + 1e-9)
        valid = ~np.ma.getmaskarray(ndvi) & np.isfinite(ndvi.filled(np.nan))
        ri, ci = np.nonzero(valid)
        if ri.size < 200:
            return None
        take = rng.choice(ri.size, min(n_sample, ri.size), replace=False)
        ri, ci = ri[take], ci[take]
        vals = np.asarray(ndvi.filled(np.nan))[ri, ci]
        xs, ys = rasterio.transform.xy(src.transform, ri * step, ci * step)
        xy = np.column_stack([np.asarray(xs), np.asarray(ys)])

    # Raw curve over a wide window, and the detrended curve over a narrower one.
    # The detrended residual is the interpretable estimate for block sizing; the
    # raw curve is kept so the non-stationarity is visible rather than hidden.
    raw = _variogram_core(vals, xy, n_lags, max_lag_pct=40.0, detrend=False)
    det = _variogram_core(vals, xy, n_lags, max_lag_pct=20.0, detrend=True)
    if det is None:
        return raw if raw is None else dict(raw, variable="NDVI (raw)")
    det["variable"] = "NDVI (quadratic trend removed)"
    if raw is not None:
        det["raw"] = dict(raw, variable="NDVI (raw)")
    return det


def _detrend_xy(vals: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Remove a quadratic surface in (x, y).

    A variogram only has a range if the field is second-order stationary. Over an
    AOI that spans sea, lagoon, mangrove and dry land, a large-scale gradient
    dominates and the variogram climbs without ever reaching a sill, so the
    "first lag at 95% of the sill" rule returns the last lag rather than a real
    range. Removing the trend first makes the residual variogram interpretable.
    """
    x = (xy[:, 0] - xy[:, 0].mean()) / 1000.0
    y = (xy[:, 1] - xy[:, 1].mean()) / 1000.0
    A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return vals - A @ coef


def _variogram_core(vals: np.ndarray, xy: np.ndarray, n_lags: int = 18,
                    max_lag_pct: float = 40.0, detrend: bool = False):
    """Binned semivariance of `vals` at coordinates `xy`, plus a range estimate.

    Returns practical_range_m=None when the variogram has not plateaued inside
    the lag window: in that case there is no range to report, and reporting the
    last lag would be an artefact of the estimator rather than a measurement.
    """
    if detrend:
        vals = _detrend_xy(vals, xy)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    gam = 0.5 * (vals[:, None] - vals[None, :]) ** 2
    iu = np.triu_indices(len(xy), k=1)
    d, gam = d[iu], gam[iu]
    ok = np.isfinite(d) & np.isfinite(gam)
    d, gam = d[ok], gam[ok]
    if d.size == 0:
        return None
    max_lag = np.percentile(d, max_lag_pct)  # unreliable beyond ~half the extent
    edges = np.linspace(0, max_lag, n_lags + 1)
    idx = np.digitize(d, edges) - 1
    lags, semis, npair = [], [], []
    for b in range(n_lags):
        m = idx == b
        if m.sum() >= 30:
            lags.append(float(0.5 * (edges[b] + edges[b + 1])))
            semis.append(float(gam[m].mean()))
            npair.append(int(m.sum()))
    if len(lags) < 4:
        return None
    lags_a, semis_a = np.asarray(lags), np.asarray(semis)
    sill = float(np.median(semis_a[-max(2, len(semis_a) // 4):]))

    # Plateau test: if the last third is still clearly above the middle third,
    # the curve is still climbing and no sill has been reached.
    t = max(2, len(semis_a) // 3)
    mid = float(semis_a[len(semis_a) // 3: 2 * len(semis_a) // 3].mean())
    last = float(semis_a[-t:].mean())
    rise = last / mid if mid > 0 else np.inf
    plateaued = bool(rise < 1.10)

    if plateaued:
        hit = np.nonzero(semis_a >= 0.95 * sill)[0]
        prange = float(lags_a[hit[0]]) if hit.size else None
        note = "practical range = first lag reaching 95% of the sill plateau"
    else:
        prange = None
        note = (f"NO PLATEAU inside {lags_a[-1]:.0f} m: semivariance still rising "
                f"({rise:.2f}x from mid to far lags). The field is not second-order "
                f"stationary at this scale, so no practical range can be read off "
                f"this variogram.")
    # Knee of the curve: lag of maximum downward curvature. Reported separately
    # from practical_range_m because it is a different quantity - it locates the
    # short-range structure in a nested/non-stationary variogram, and it is NOT
    # a sill-based range. Useful when no plateau exists.
    knee = None
    if len(lags_a) >= 5:
        g = semis_a / (sill if sill > 0 else 1.0)
        d2 = np.gradient(np.gradient(g, lags_a), lags_a)
        hi = max(3, int(0.7 * len(lags_a)))
        j = int(np.argmin(d2[:hi]))
        knee = float(lags_a[j])

    return {
        "n_sample_pixels": int(len(xy)),
        "lag_m": lags, "semivariance": semis, "n_pairs": npair,
        "sill_estimate": sill,
        "practical_range_m": prange,
        "knee_m": knee,
        "plateaued": plateaued,
        "rise_ratio": float(rise),
        "max_lag_m": float(lags_a[-1]),
        "note": note,
    }


def embedding_variogram(path: Path, n_pc: int = 3, n_sample: int = 4000,
                        n_lags: int = 18, seed: int = 0):
    """Practical autocorrelation range of an AlphaEarth-style embedding stack.

    Google does not publish AlphaEarth's effective receptive-field width, so it
    cannot be looked up; this measures the spatial smoothness of the delivered
    embedding field instead. Runs the variogram on the leading principal
    components of the band stack and returns the widest range found, which is
    the conservative choice for sizing CV blocks.
    """
    rng = np.random.default_rng(seed)
    with rasterio.open(path) as src:
        step = max(1, int(math.ceil(math.sqrt(src.width * src.height / 1_000_000))))
        out_shape = (src.count, max(1, src.height // step), max(1, src.width // step))
        stack = src.read(out_shape=out_shape, masked=True).astype("float64")
        transform = src.transform
        n_bands = src.count

    valid = ~np.ma.getmaskarray(stack).any(axis=0)
    valid &= np.isfinite(stack.filled(np.nan)).all(axis=0)
    ri, ci = np.nonzero(valid)
    if ri.size < 200:
        return None
    take = rng.choice(ri.size, min(n_sample, ri.size), replace=False)
    ri, ci = ri[take], ci[take]
    X = np.asarray(stack.filled(np.nan))[:, ri, ci].T          # (n_pixels, n_bands)
    X = X - X.mean(axis=0, keepdims=True)
    # PCA via SVD; embeddings are already unit-scaled so no variance scaling.
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    k = min(n_pc, vt.shape[0])
    pcs = X @ vt[:k].T
    var_share = (sv ** 2 / (sv ** 2).sum())[:k]

    xs, ys = rasterio.transform.xy(transform, ri * step, ci * step)
    xy = np.column_stack([np.asarray(xs), np.asarray(ys)])

    per_pc, ranges = [], []
    for j in range(k):
        vg = _variogram_core(pcs[:, j], xy, n_lags)
        if vg:
            vg["variable"] = f"PC{j + 1}"
            vg["variance_share"] = float(var_share[j])
            per_pc.append(vg)
            ranges.append(vg["practical_range_m"])
    if not per_pc:
        return None
    widest = int(np.argmax(ranges))
    return {
        "variable": f"embedding PCs 1-{k} of {n_bands} bands",
        "n_sample_pixels": int(len(xy)),
        "per_pc": per_pc,
        "practical_range_m": float(max(ranges)),
        "range_from": per_pc[widest]["variable"],
        "lag_m": per_pc[widest]["lag_m"],
        "semivariance": per_pc[widest]["semivariance"],
        "sill_estimate": per_pc[widest]["sill_estimate"],
        "note": "widest practical range across leading PCs; conservative proxy for "
                "AlphaEarth's undocumented receptive-field width",
    }


# --------------------------------------------------------------------------
# grid alignment + split sizing
# --------------------------------------------------------------------------
def check_grid_alignment(ras: dict, emb: dict) -> dict:
    """Warn loudly if the embedding raster is not on the Sentinel-2 grid.

    A sub-pixel origin offset silently mixes neighbouring pixels into one arm of
    the benchmark, which is exactly the kind of error that shows up as a model
    difference.
    """
    res = {"crs_match": ras["crs"] == emb["crs"],
           "pixel_size_match": np.allclose(ras["pixel_size"], emb["pixel_size"], atol=1e-6)}
    px, py = ras["pixel_size"]
    dx = (emb["transform"][2] - ras["transform"][2]) % px
    dy = (emb["transform"][5] - ras["transform"][5]) % py
    off_x = min(dx, px - dx)
    off_y = min(dy, py - dy)
    res["origin_offset_m"] = [float(off_x), float(off_y)]
    res["origin_aligned"] = bool(off_x < 1e-6 and off_y < 1e-6)
    res["aligned"] = bool(res["crs_match"] and res["pixel_size_match"] and res["origin_aligned"])
    if not res["aligned"]:
        print("WARNING: embedding raster is not on the Sentinel-2 grid "
              f"(crs_match={res['crs_match']}, pixel_size_match={res['pixel_size_match']}, "
              f"origin offset={res['origin_offset_m']} m). Re-export with an explicit "
              "crs + crsTransform copied from the Sentinel-2 raster.", file=sys.stderr)
    return res


def recommend_split(vario, emb_vario, ras: dict, k: int = 5):
    """Turn measured ranges into a concrete block edge, buffer and fold count."""
    ranges = {}
    if vario and vario.get("practical_range_m"):
        ranges[vario.get("variable", "NDVI")] = vario["practical_range_m"]
    if emb_vario and emb_vario.get("practical_range_m"):
        ranges["embedding"] = emb_vario["practical_range_m"]
    if not ranges:
        return {"ranges_m": {}, "requested_folds": k,
                "verdict": "No variogram reached a plateau, so no autocorrelation "
                           "range could be measured and the block size cannot be "
                           "derived from the data. Size the blocks from the "
                           "AlphaEarth residual variogram once the embedding "
                           "sample is available, and report the random-split vs "
                           "blocked-split gap as the empirical check."}
    driver = max(ranges, key=ranges.get)
    r = ranges[driver]
    L = 2.0 * r
    out = {
        "ranges_m": ranges,
        "binding_range_m": float(r),
        "binding_variable": driver,
        "block_edge_m": float(L),
        "buffer_m": float(r),
        "requested_folds": k,
    }
    if "extent_km" in ras:
        W, H = (v * 1000.0 for v in ras["extent_km"])
        nb = int(math.floor(W / L) * math.floor(H / L))
        out["approx_blocks"] = nb
        out["blocks_per_fold"] = nb / k if nb else 0
        if nb < 2 * k:
            out["verdict"] = (
                f"AOI supports only ~{nb} blocks of {L:.0f} m, under 2 per fold at k={k}. "
                "Use leave-one-block-out with fewer folds, or enlarge the export. "
                "Do not shrink the blocks to manufacture folds."
            )
        else:
            out["verdict"] = (
                f"AOI supports ~{nb} blocks of {L:.0f} m, about {nb / k:.1f} per fold "
                f"at k={k}. Workable."
            )
    return out


# --------------------------------------------------------------------------
# figures + report
# --------------------------------------------------------------------------
def make_figures(out: Path, lab: dict, xy, labels, vario, raster_info, emb_vario=None):
    figs = []
    fig, ax = plt.subplots(figsize=(6, 3.4))
    cc = lab["class_counts"]
    ax.bar(list(cc.keys()), list(cc.values()), color="#3d6b9c")
    ax.set_ylabel("labeled points")
    ax.set_title("Class balance")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p = out / "class_balance.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p.name)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for cls in sorted(set(labels)):
        m = np.asarray(labels) == cls
        ax.scatter(xy[m, 0], xy[m, 1], s=8, label=f"{cls} (n={m.sum()})")
    left, bottom, right, top = raster_info["bounds_native"]
    ax.add_patch(plt.Rectangle((left, bottom), right - left, top - bottom,
                               fill=False, ec="k", lw=0.8, ls="--"))
    ax.set_aspect("equal")
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title("Label locations vs raster extent")
    ax.legend(fontsize=7, markerscale=1.5)
    fig.tight_layout()
    p = out / "label_locations.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p.name)

    cand = []
    if vario and vario.get("raw"):
        cand.append((vario["raw"], "NDVI (raw)", "#8c8c8c"))
    if vario:
        cand.append((vario, vario.get("variable", "NDVI"), "#3d6b9c"))
    if emb_vario:
        cand.append((emb_vario, "AlphaEarth embedding", "#2e8b6b"))
    panels = [(v, t, c) for v, t, c in cand if v]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 3.5),
                                 squeeze=False)
        for ax, (v, title, colour) in zip(axes[0], panels):
            # semivariance is normalised by the sill so the two panels are comparable
            sill = v["sill_estimate"] or 1.0
            ax.plot(v["lag_m"], np.asarray(v["semivariance"]) / sill, "o-", color=colour)
            ax.axhline(1.0, ls=":", c="grey", label="sill estimate")
            if v.get("practical_range_m"):
                ax.axvline(v["practical_range_m"], ls="--", c="#b4432f",
                           label=f"range ≈ {v['practical_range_m']:.0f} m")
            else:
                ax.text(0.04, 0.62, "no plateau:\nno sill-based range", ha="left",
                        va="top", transform=ax.transAxes, fontsize=8, color="#b4432f")
                if v.get("knee_m"):
                    ax.axvline(v["knee_m"], ls="-.", c="#d08b2c",
                               label=f"knee ≈ {v['knee_m']:.0f} m")
            ax.set_xlabel("lag distance (m)")
            ax.set_ylabel("semivariance / sill")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
        fig.suptitle("Spatial autocorrelation: a sill-based range is needed to size "
                     "the CV blocks", fontsize=11)
        fig.tight_layout()
        p = out / "variograms.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append(p.name)
    return figs


def write_markdown(out: Path, ras: dict, lab: dict, vario, figs, emb_vario=None,
                   split=None, align=None):
    L = []
    A = L.append
    A("# Data inspection report\n")
    A("## Raster\n")
    A(f"- File: `{Path(ras['path']).name}` ({ras['driver']})")
    A(f"- CRS: {ras['crs']} (EPSG:{ras['crs_epsg']}, "
      f"{'projected' if ras['crs_is_projected'] else 'geographic'}, units={ras['crs_units']})")
    A(f"- Size: {ras['width']} x {ras['height']} px, {ras['count']} bands")
    A(f"- Pixel size: {ras['pixel_size'][0]:g} x {ras['pixel_size'][1]:g} {ras['crs_units']}")
    A(f"- Extent (native): {['%.1f' % v for v in ras['bounds_native']]}")
    if "bounds_wgs84" in ras:
        A(f"- Extent (WGS84): {['%.5f' % v for v in ras['bounds_wgs84']]}")
    if "extent_km" in ras:
        A(f"- Extent: {ras['extent_km'][0]:.2f} x {ras['extent_km'][1]:.2f} km "
          f"({ras['area_km2']:.2f} km²)")
    A(f"- dtypes: {sorted(set(ras['dtypes']))}")
    A(f"- nodata: {ras['nodata']} (per band: {ras['nodatavals']})")
    A(f"- Compression: {ras['compression']}, blocks: {ras['block_shapes'][0]}, "
      f"overviews: {ras['overview_levels'] or 'none'}")
    A(f"- Scaling: {ras['scaling_guess']}")
    A(f"\nBand statistics (decimation step {ras['stats_decimation_step']}, "
      f"{ras['stats_sampled_pixels']:,} pixels sampled):\n")
    A("| # | name | dtype | valid % | min | p50 | max | mean | std |")
    A("|---|------|-------|---------|-----|-----|-----|------|-----|")
    for b in ras["bands"]:
        A(f"| {b['index']} | {b['description'] or '-'} | {b['dtype']} | "
          f"{100 * b['valid_fraction']:.1f} | {b.get('min', float('nan')):.4g} | "
          f"{b.get('p50', float('nan')):.4g} | {b.get('max', float('nan')):.4g} | "
          f"{b.get('mean', float('nan')):.4g} | {b.get('std', float('nan')):.4g} |")

    A("\n## Labels\n")
    A(f"- File: `{Path(lab['path']).name}`, {lab['n_features']} features, "
      f"geometry {lab['geometry_types']}")
    A(f"- CRS: {lab['crs']} (EPSG:{lab['crs_epsg']})"
      + ("  **reprojected to raster CRS for analysis**" if lab["reprojected_to_raster_crs"] else ""))
    A(f"- Class column: `{lab['label_field']}`, {lab['n_classes']} classes, "
      f"imbalance ratio {lab['imbalance_ratio']:.2f}")
    A("\n| class | n | share |")
    A("|-------|---|-------|")
    for k, v in lab["class_counts"].items():
        A(f"| {k} | {v} | {100 * lab['class_proportions'][k]:.1f}% |")
    A("\nGeometry / leakage diagnostics:\n")
    A(f"- Points outside raster bbox: {lab['n_outside_raster_bbox']}")
    A(f"- Points on nodata pixels: {lab.get('n_points_on_nodata', 'n/a')}")
    A(f"- Unique pixels covered: {lab['n_unique_pixels']} of {lab['n_features']} points")
    A(f"- Points sharing a pixel with another point: {lab['n_points_sharing_a_pixel']}")
    A(f"- Pixels carrying conflicting class labels: {lab['n_pixels_with_conflicting_labels']}")
    if "nn_distance_m" in lab:
        nn = lab["nn_distance_m"]
        A(f"- Nearest-neighbour distance (m): min {nn['min']:.1f}, "
          f"p5 {nn['p5']:.1f}, median {nn['median']:.1f}, p95 {nn['p95']:.1f}, max {nn['max']:.1f}")
        A(f"- Share of points within 1 pixel of another: {100 * lab['nn_within_1_pixel_fraction']:.1f}%; "
          f"within 10 pixels: {100 * lab['nn_within_10_pixels_fraction']:.1f}%")
    A(f"- Invalid / empty geometries: {lab['n_invalid_geometry']} / {lab['n_empty_geometry']}")
    A(f"- Missing class values: {lab['n_missing_label']}")

    if align:
        A("\n## Embedding / Sentinel-2 grid alignment\n")
        verdict = "ALIGNED" if align["aligned"] else "**NOT ALIGNED — re-export**"
        A(f"- Verdict: {verdict}")
        A(f"- CRS match: {align['crs_match']}; pixel size match: {align['pixel_size_match']}")
        A(f"- Origin offset: {align['origin_offset_m'][0]:.4g} m east, "
          f"{align['origin_offset_m'][1]:.4g} m north (must be 0)")
        if not align["aligned"]:
            A("- A sub-pixel offset mixes neighbouring pixels into the embedding arm "
              "only, which shows up as a spurious model difference. Re-export with an "
              "explicit crs and crsTransform copied from the Sentinel-2 raster.")

    A("\n## Spatial autocorrelation\n")
    def _vg_line(v, label):
        if v.get("practical_range_m"):
            return (f"- {label} on {v['n_sample_pixels']} random pixels: "
                    f"sill ≈ {v['sill_estimate']:.4g}, practical range ≈ "
                    f"**{v['practical_range_m']:.0f} m**")
        k = (f" Knee of the curve (max downward curvature, locates the short-range "
             f"structure — not a sill-based range): **≈ {v['knee_m']:.0f} m**."
             if v.get("knee_m") else "")
        return (f"- {label} on {v['n_sample_pixels']} random pixels: "
                f"**no range readable** — {v.get('note', '')}{k}")

    if vario:
        if vario.get("raw"):
            A(_vg_line(vario["raw"], "NDVI variogram, raw"))
        A(_vg_line(vario, vario.get("variable", "NDVI variogram")))
    else:
        A("- NDVI range not estimated (needs red and NIR band indices; "
          "rerun with --red/--nir).")
    if emb_vario:
        A(f"- AlphaEarth embedding variogram on {emb_vario['n_sample_pixels']} pixels "
          f"({emb_vario['variable']}): practical range ≈ "
          f"**{emb_vario['practical_range_m']:.0f} m** (widest, from "
          f"{emb_vario['range_from']})")
        for pc in emb_vario["per_pc"]:
            A(f"  - {pc['variable']} ({100 * pc['variance_share']:.1f}% of variance): "
              f"range ≈ {pc['practical_range_m']:.0f} m")
        A("- AlphaEarth's effective receptive-field width is not published; this "
          "measured range is the conservative stand-in for it.")
    else:
        A("- Embedding range not estimated (pass --embedding <alphaearth.tif>).")

    if split and not split.get("ranges_m"):
        A("\n## Recommended split geometry\n")
        A(f"- {split['verdict']}")
    elif split:
        A("\n## Recommended split geometry\n")
        A(f"- Binding range: **{split['binding_range_m']:.0f} m** "
          f"(from {split['binding_variable']}); all measured: "
          + ", ".join(f"{k} {v:.0f} m" for k, v in split["ranges_m"].items()))
        A(f"- Block edge L = 2 x binding range = **{split['block_edge_m']:.0f} m**")
        A(f"- Train-side buffer b = **{split['buffer_m']:.0f} m**")
        if "approx_blocks" in split:
            A(f"- Approx usable blocks in this AOI: **{split['approx_blocks']}** "
              f"({split['blocks_per_fold']:.1f} per fold at k={split['requested_folds']})")
            A(f"- Verdict: {split['verdict']}")

    A("\n## Figures\n")
    for f in figs:
        A(f"- `{f}`")
    (out / "inspection_report.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--label-field", default=None)
    ap.add_argument("--out", default=Path("inspection"), type=Path)
    ap.add_argument("--red", type=int, default=None, help="1-based index of the red band")
    ap.add_argument("--nir", type=int, default=None, help="1-based index of the NIR band")
    ap.add_argument("--embedding", type=Path, default=None,
                    help="optional AlphaEarth (A00-A63) GeoTIFF; its autocorrelation "
                         "range sets the CV block size")
    ap.add_argument("--no-variogram", action="store_true")
    a = ap.parse_args()

    for p in (a.raster, a.labels):
        if not p.exists():
            sys.exit(f"not found: {p}")
    a.out.mkdir(parents=True, exist_ok=True)

    ras = inspect_raster(a.raster)
    lab, gdf, xy, labels, field = inspect_labels(a.labels, ras, a.label_field)

    # resolve red / NIR from band descriptions when not given
    red, nir = a.red, a.nir
    if red is None or nir is None:
        names = [str(d).upper() if d else "" for d in ras["band_descriptions"]]
        for i, nm in enumerate(names, start=1):
            if red is None and nm in ("B4", "RED", "B04"):
                red = i
            if nir is None and nm in ("B8", "NIR", "B08"):
                nir = i
    vario = emb_vario = align = emb_ras = None
    if not a.no_variogram and red and nir:
        vario = empirical_variogram(a.raster, red, nir)
    if a.embedding is not None:
        if not a.embedding.exists():
            sys.exit(f"not found: {a.embedding}")
        emb_ras = inspect_raster(a.embedding)
        # alignment is a correctness check, not a diagnostic: always run it
        align = check_grid_alignment(ras, emb_ras)
        if not a.no_variogram:
            emb_vario = embedding_variogram(a.embedding)

    split = recommend_split(vario, emb_vario, ras)
    figs = make_figures(a.out, lab, xy, labels, vario, ras, emb_vario)
    write_markdown(a.out, ras, lab, vario, figs, emb_vario, split, align)
    payload = {"raster": ras, "labels": lab, "variogram": vario,
               "embedding_raster": emb_ras, "embedding_variogram": emb_vario,
               "grid_alignment": align, "split_recommendation": split,
               "red_band": red, "nir_band": nir}
    (a.out / "inspection_report.json").write_text(json.dumps(jsonable(payload), indent=2))
    print(f"wrote {a.out}/inspection_report.md and .json plus {len(figs)} figures")
    print(f"classes: {lab['class_counts']}")
    def _rng(v):
        if not v:
            return "n/a"
        if v.get("practical_range_m"):
            return f"{v['practical_range_m']:.0f} m"
        k = v.get("knee_m")
        return f"no plateau (knee ≈ {k:.0f} m)" if k else "no plateau"

    if vario:
        print(f"NDVI range [{vario.get('variable')}]: {_rng(vario)}")
        if vario.get("raw"):
            print(f"NDVI range [raw]: {_rng(vario['raw'])}")
    if emb_vario:
        print(f"embedding range: {_rng(emb_vario)}")
    if split and split.get("ranges_m"):
        print(f"recommended block edge: {split['block_edge_m']:.0f} m; "
              f"buffer {split['buffer_m']:.0f} m; "
              f"approx usable blocks: {split.get('approx_blocks')}")
    elif split:
        print("block size not derivable from the data: " + split["verdict"][:90] + "...")


if __name__ == "__main__":
    main()
