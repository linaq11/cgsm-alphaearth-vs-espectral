#!/usr/bin/env python
"""
Sample Sentinel-2 spectral features at the labelled points.

Scaling (confirmed by the analyst): the 1e-4 reflectance factor applies ONLY to
B2-B12; the stored NDVI/NDWI/CMRI/NDBaI bands are already in [-1, 1].

Indices are recomputed from the bands rather than read from the file, so the
definitions are documented and unambiguous:

    NDVI  = (B8  - B4 ) / (B8  + B4 )        Rouse et al. 1974
    NDWI  = (B3  - B8 ) / (B3  + B8 )        McFeeters 1996  (green - NIR)
    MNDWI = (B3  - B11) / (B3  + B11)        Xu 2006         (green - SWIR1)

Note the stored NDWI band is ambiguous between the McFeeters (green-NIR) and Gao
(NIR-SWIR) conventions; recomputing removes that ambiguity. Ratio indices are
invariant to the 1e-4 factor, so the scaling choice cannot affect them - it
affects only the six reflectance features.

Usage
-----
python extract_spectral.py --raster s2_202103.tif --split split/split_assignments.csv \
    --out spectral_features.csv
"""
from __future__ import annotations

import argparse
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

import rasterio

REFLECTANCE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
SCALE = 1e-4


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--out", default=Path("spectral_features.csv"), type=Path)
    a = ap.parse_args()

    sp = pd.read_csv(a.split)
    with rasterio.open(a.raster) as src:
        names = [d if d else f"band{i+1}" for i, d in enumerate(src.descriptions)]
        if str(src.crs) != "EPSG:9377":
            sys.exit(f"expected raster in EPSG:9377, got {src.crs}")
        xy = list(zip(sp.x.values, sp.y.values))
        vals = np.array([v for v in src.sample(xy, masked=True)], dtype="float64")
        mask = np.ma.getmaskarray(np.ma.masked_invalid(vals)).any(axis=1)
    raw = pd.DataFrame(vals, columns=names)
    print(f"sampled {len(raw)} points x {raw.shape[1]} bands; "
          f"points with any masked/invalid band: {int(mask.sum())}")

    missing = [b for b in REFLECTANCE_BANDS if b not in raw.columns]
    if missing:
        sys.exit(f"raster is missing reflectance bands {missing}; have {names}")

    out = sp[["point_id", "x", "y", "clase", "clase_nombre"]].copy()
    for b in REFLECTANCE_BANDS:
        out[b] = raw[b].to_numpy() * SCALE

    B3, B4, B8, B11 = (out[c].to_numpy() for c in ("B3", "B4", "B8", "B11"))
    eps = 1e-12
    out["NDVI"] = (B8 - B4) / (B8 + B4 + eps)
    out["NDWI"] = (B3 - B8) / (B3 + B8 + eps)
    out["MNDWI"] = (B3 - B11) / (B3 + B11 + eps)

    # cross-check the recomputed NDVI against the band shipped in the file: a
    # large discrepancy would mean the stored band used a different definition
    if "NDVI" in raw.columns:
        d = np.abs(out["NDVI"].to_numpy() - raw["NDVI"].to_numpy())
        print(f"recomputed vs stored NDVI: max |diff| = {np.nanmax(d):.2e}")
    if "NDWI" in raw.columns:
        d = np.abs(out["NDWI"].to_numpy() - raw["NDWI"].to_numpy())
        print(f"recomputed vs stored NDWI: max |diff| = {np.nanmax(d):.2e} "
              f"(large => stored band uses the Gao NIR-SWIR convention)")

    assert out.notna().all().all(), "NaNs in the spectral feature table"
    for c in ("NDVI", "NDWI", "MNDWI"):
        assert out[c].between(-1, 1).all(), f"{c} outside [-1, 1]"
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out} with {out.shape[1] - 5} features")
    print(out[REFLECTANCE_BANDS + ["NDVI", "NDWI", "MNDWI"]]
          .describe().loc[["min", "50%", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
