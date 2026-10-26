# Benchmark design — RF on spectral indices vs. foundation-model embeddings
### Ciénaga Grande de Santa Marta, Sentinel-2, 4 classes

Status: **labels inspected; raster still unreachable** (`D:\` is not mounted in
WSL, so `s2_201903.tif` cannot be opened). Everything below marked *measured*
comes from the actual label file; the raster sections remain unfilled.

---

## 0. What the data turned out to be

**Labels** — `etiquetas_debiles_worldcover_v1.geojson`, 600 points, EPSG:4326,
columns `clase` (1–4), `clase_nombre`, `origen`, `fuente`, `uso`. Exactly
balanced: 150 points per class. Self-documented provenance:
`debil_worldcover_v200_2021`, sampled from pure 3×3 neighbourhoods, seed
20260829, and flagged in the file itself as a pipeline pilot rather than the
Olofsson validation sample (which arrives in M6).

Three corrections to the original brief:

1. **The classes are not the ones first described.** They are `manglar`,
   `agua`, `humedal_herbaceo`, `otro` — herbaceous wetland, not bare soil.
   `humedal_herbaceo` (WorldCover class 90) borders both mangrove and water in
   spectral space, so it is the class most likely to absorb the confusion that
   would otherwise fall between the other three, and it is where the two arms
   will most plausibly differ.
2. **Temporal mismatch.** The labels are WorldCover epoch **2021**; the raster
   is `s2_201903` — **March 2019**. Two years apart, in a system that changes
   on that timescale through hypersalinity and flood pulses. Whatever the two
   feature sets do, part of the measured error is this gap rather than model
   quality. Resolving it is cheap and worth doing: re-export the Sentinel-2
   subset for 2021 so labels, spectral features and AlphaEarth year all agree.
3. **Weak labels bound what the benchmark can claim.** With WorldCover as the
   reference, the metric is *agreement with WorldCover*, not accuracy. The
   `3×3 pure` filter compounds this: it keeps homogeneous cores and discards
   mixed and edge pixels, which is precisely where mangrove mapping is hard.
   Expect both arms to score high and the gap between them to be compressed.
   This is fine for a pipeline pilot — as the file itself says — but the
   headline number must be reported as agreement with a weak reference, and the
   RF-vs-AlphaEarth comparison treated as a relative ranking under a shared
   label bias, not as an accuracy estimate.

**Label geometry (measured)** — good news for the split design. Points are well
dispersed: minimum 1-NN distance 94 m, median 746 m, p95 1548 m; no two points
share a 10 m cell, so there is no pixel-level duplication to clean. Median
nearest same-class distance is 913–1129 m depending on class. Label bounding box
is 53.2 × 59.2 km in EPSG:9377; the AOI polygon is 2042 km². At a block edge of
2–3 km this yields hundreds of blocks, so 5 spatially independent folds are
comfortably supported — the §2.8 feasibility worry does not apply.

One caveat visible in the label map: the **eastern lobe of the AOI contains
almost no labelled points**. The spatial holdout must be drawn from the
labelled western and central area, or it will be empty.

**Raster (reported, not yet verified)** — EPSG:9377 (MAGNA-SIRGAS 2018 /
Origen-Nacional, a projected CRS in metres, so suitable for blocking without
reprojection), 10 bands: `B2, B3, B4, B8, B11, B12, NDVI, NDWI, CMRI, NDBaI`.

- **MNDWI is not missing in practice.** It needs green and SWIR1, and both `B3`
  and `B11` are present, so it can be computed locally. No re-export needed for
  that reason alone.
- **Indices will be recomputed from the bands rather than taken as given.** The
  supplied `NDWI` is ambiguous between the McFeeters (green–NIR) and Gao
  (NIR–SWIR) definitions, and the benchmark needs documented formulas. The
  supplied `CMRI` (= NDVI − NDWI) is an exact linear combination of two other
  features; harmless for Random Forest but it dilutes feature importances
  across redundant inputs, so it will be reported as a redundant extra rather
  than as an independent predictor.

---

## 1. Inspection (ready to run, not yet run)

`inspect_data.py` produces the requested report in one command:

```bash
python inspect_data.py --raster <S2.tif> --labels <labels.geojson> \
       --label-field <class column> --out inspection
```

It reports, and writes to both Markdown and JSON:

**Raster** — driver, CRS (EPSG, projected/geographic, linear units), width ×
height, band count, per-band dtype, per-band nodata, pixel size, extent in
native CRS and WGS84, extent in km and km², compression, block shape,
overviews, per-band valid-pixel fraction, min/p1/p25/p50/p75/p99/max/mean/std
from a decimated read, and a reflectance-scaling check (L2A DN vs. scaled
reflectance).

**Labels** — feature count, CRS and whether reprojection to the raster CRS was
needed, geometry types, class column detection, per-class counts and shares,
imbalance ratio, invalid/empty geometries, missing class values.

**Split-validity diagnostics** — these decide whether any split can be honest:
points outside the raster bbox, points landing on nodata, unique pixels covered
vs. point count, points sharing a pixel with another point, pixels carrying
*conflicting* class labels, nearest-neighbour distance distribution, and the
share of points within 1 and 10 pixels of another point.

**Autocorrelation** — empirical NDVI variogram on random valid pixels, with a
practical range estimated as the first lag reaching 95 % of the sill plateau.

Once the AlphaEarth GeoTIFF exists, pass it too:

```bash
python inspect_data.py --raster <S2.tif> --labels <labels.geojson> \
       --embedding <alphaearth_cgsm_YYYY.tif> --out inspection
```

That adds: a **grid-alignment check** against the Sentinel-2 raster (CRS, pixel
size, and sub-pixel origin offset, which must be zero — see §3.2), the
**autocorrelation range of the embedding field** measured on the leading
principal components of the 64 bands, and a concrete **recommended block edge,
buffer width and usable block count** with a verdict on whether the AOI can
support 5 folds at all.

Verified against a synthetic fixture (int16 DN, nodata holes, mismatched label
CRS, clumped points, duplicate pixel). Two defects were found and fixed in the
process: an unset `PROJ_DATA` breaking every CRS lookup in this conda build, and
a non-vectorising `DatasetReader.index` call.

Environment: conda env `geo-bench` (rasterio, geopandas, rioxarray, xarray,
scikit-learn, pyproj, shapely, scikit-image, matplotlib, seaborn).

---

## 2. Proposed split: buffered, class-stratified spatial block CV

Random point splits are not usable here. Label points collected by
photo-interpretation or field transects are clumped, so a random split puts
near-duplicate points on both sides and inflates accuracy — often by tens of
points of overall accuracy in mangrove mapping. The design below removes three
distinct leakage channels.

### 2.1 Deduplicate to one label per pixel
Two points in the same 10 m pixel have *identical* features. Across a split
that is label duplication, not independent evidence. Keep one label per pixel;
where a pixel carries conflicting classes, drop it and report the count (it
indicates either positional error or genuine mixed pixels).

### 2.2 Block size from the data, not from convention
Let `r_spec` = practical autocorrelation range of NDVI, and `r_emb` = practical
autocorrelation range of the AlphaEarth embedding field (see below).

```
block edge  L = 2 · max(r_spec, r_emb)
buffer      b = max(r_spec, r_emb)
```

**Measured on `s2_202103.tif`: the NDVI variogram has no sill, so `r_spec` does
not exist at this scale.** The raw curve is still climbing at 20.7 km (far lags
1.23× the middle lags) and even after removing a quadratic surface in (x, y) it
is still climbing at 13.3 km (1.13×). This is what an AOI spanning open sea,
lagoon, mangrove fringe and dry farmland looks like: a large-scale gradient
dominates and the field is not second-order stationary, so the "first lag at
95 % of the sill" rule returns the last lag rather than a range. The script now
detects this and reports no range instead of a number, which is why the earlier
run's "17 km range" — and the 34 km block edge it implied — was an estimator
artefact and has been withdrawn.

What the curves do give is a **knee**, the lag of maximum downward curvature,
which locates the short-range structure: ≈ 1773 m raw, ≈ 1136 m detrended. That
is not a sill-based range and must not be reported as one, but it is a
defensible lower bound for the block edge.

### 2.2b AlphaEarth variogram, and why the block size is a sensitivity axis

Measured on the 4000-point AlphaEarth 2021 sample (PCA on the 64 bands; PC1
53.2 %, PC2 11.9 %, PC3 8.2 % of variance):

| component | raw | detrended |
|---|---|---|
| PC1 (53.2 %) | no plateau, knee ≈ 2945 m | no plateau, knee ≈ 1131 m |
| PC2 (11.9 %) | range ≈ 10.0 km | range ≈ 7.2 km |
| PC3 (8.2 %) | no plateau, knee ≈ 1767 m | range ≈ 9.4 km |

The estimates disagree by nearly an order of magnitude, and the disagreement is
structured: the dominant component never plateaus in either series, while the
two minor components reach a sill only in specific series — PC2 at 10.0 km raw
and 7.2 km detrended, PC3 at 9.4 km detrended but never in the raw series. So
"the" autocorrelation range of the embedding
field does not exist here either — the low-variance components carry
landscape-scale organisation (the lagoon system is ~30 km across) that the
quadratic detrend does not remove.

Applying §2.2 literally with the widest range from the detrended series — the
one the design treats as interpretable — is PC3 at 9.4 km, giving `L ≈ 19 km`.
(The widest plateau of any series is PC2 raw at 10.0 km, which would push `L`
wider still.) Measured consequence: 12 usable blocks over the labelled area, and folds
that cannot cover all four classes. The honest conclusion is that **no block
size simultaneously exceeds the widest correlation length, leaves enough blocks
for k folds, and retains enough training data** — the labelled area is only
about six correlation lengths across. Measured trade-off:

| L (km) | blocks | blocks/fold | train retained after buffer | min train–test | all 4 classes |
|---|---|---|---|---|---|
| 3 | 189 | 37.8 | 85.7 % | 1.50 km | yes |
| 6 | 63 | 12.6 | 70.1 % | 3.00 km | yes |
| 8 | 40 | 8.0 | 72.8 % | 4.00 km | **no** |
| 12 | 22 | 4.4 | 67.2 % | 6.03 km | yes |
| 16 | 16 | 3.2 | 53.7 % | 8.00 km | yes |
| 19 | 12 | 2.4 | 53.7 % | 9.52 km | **no** |

Block counts in that sweep are over all 600 labels, before the holdout is
carved. The realised primary regime therefore has **53 blocks over the 468-point
CV pool, 10.6 per fold** — not the 63 / 12.6 of the pre-holdout sweep. The
conclusion is unchanged; the sweep is for ranking candidate L values, and the
per-fold counts that matter are in `split_summary.csv`.

**Decision: report three regimes rather than defend one number.**

1. `random` (k = 5, no buffer) — deliberately leaky upper bound; min train–test separation 94 m.
2. `block_6km` (k = 5, buffer 3 km) — primary. Exceeds both NDVI knees and the PC1 knee, keeps 70 % of training data, 53 blocks over the CV pool (10.6 per fold), all four classes in every fold.
3. `block_16km` (k = 3, buffer 8 km) — conservative bound. Exceeds the PC2 plateau range; k drops to 3 because 16 blocks cannot support five folds.

If the RF-vs-AlphaEarth ranking is stable across all three, the conclusion does
not depend on the block-size choice, which is a stronger statement than any
single defensible L. If the ranking flips between regimes, that flip *is* the
result and must be reported as such.

**Closing check (Roberts et al. 2017).** Predictor-field ranges overestimate the
block size actually needed, because what must be independent is the model
*residual*, not the predictor. After fitting, measure the autocorrelation of the
residuals and confirm the chosen block edge exceeds it; that is the check that
retroactively justifies 6 km, and it costs one variogram.

The `r_emb` term is the one that is easy to miss and that specifically
threatens *this* benchmark. AlphaEarth is delivered as a per-pixel 64-band
product, which makes it look like an ordinary raster with no receptive field to
worry about. It is not: each embedding vector encodes conditions measured at
that pixel **and its surroundings** over the year. So an embedding at a test
pixel can carry information from ground that contains training pixels — leakage
that is invisible at the point level and that flatters the embedding arm only.

Google does not publish the effective receptive-field width, so it cannot be
looked up. It can be **measured**: run the same variogram machinery on the
embedding bands (on the first 3–5 principal components of the 64, which is
where most of the variance sits) and take the practical range. That is a
data-driven stand-in for the undocumented footprint, and it is likely to be the
binding constraint — embedding fields are typically smoother than NDVI. Expect
`r_emb > r_spec`, and expect the blocks to be sized by AlphaEarth rather than
by the spectral data.

### 2.3 Fold assignment
Tile the AOI into `L × L` blocks in the projected CRS. Assign **whole blocks**
to k = 5 folds by greedy class-stratified allocation, so every fold retains all
four classes — important because `bare soil` and `other` will likely be the
rare classes and a naive block draw can empty them.

### 2.4 Train-side buffer
For each fold, drop training points within `b` of any test block. This is
buffered (a.k.a. blocked-with-dead-zone) CV. Cost: fewer training points per
fold; benefit: the remaining train/test pairs are approximately independent.
Report points lost per fold — if the loss is severe, the AOI is too small for
this design and that is a finding, not something to tune away.

### 2.5 Reserved spatial holdout
Hold out one **contiguous** region of ~20–25 % of blocks, containing all four
classes, and touch it exactly once, at the end, for the head-to-head. CV inside
the remaining area is used for hyperparameters only. This is the estimate that
answers "does the FM arm generalise to ground it has never seen".

### 2.6 Report both regimes
Run the naive random point split as well and report both. The **gap** between
random and blocked CV is a headline result: it quantifies how much of the
apparent skill was spatial autocorrelation. Report it per arm — embeddings are
spatially smoother than spectral features, so the two arms need not suffer
equally.

### 2.7 Paired comparison
Both arms consume one shared fold-assignment file at identical label pixels, so
the comparison is paired. On the holdout, compare with **McNemar's test** on the
same test pixels. Report per-fold variance alongside the point estimate; with a
handful of spatial folds, a 2–3 point difference in macro-F1 is usually inside
fold-to-fold noise.

### 2.8 Feasibility constraint to check first
A small AOI cannot support this. If the AOI is `W × H` km, the number of usable
blocks is roughly `(W/L)·(H/L)`. With `L = 2.24 km`, a 10 × 10 km subset gives
~20 blocks — workable for 5 folds. A 5 × 5 km subset gives ~4 — not workable,
and the honest responses are leave-one-block-out with k = 3, or enlarging the
export. Deciding this needs the AOI extent and the label count, i.e. step 1.

### 2.9 Metrics
Overall accuracy, Cohen's kappa, per-class precision / recall / F1, macro-F1
(the metric to lead on, given imbalance), and confusion matrices. If you want
*area* statements for the mangrove class rather than pixel accuracy, say so —
that needs Olofsson-style area-adjusted estimation with the confusion matrix,
which is a different calculation and needs a probability sample.

---

## 3. Embedding arm: AlphaEarth (decided)

**Source: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`** — Google Satellite Embedding,
produced by AlphaEarth Foundations. 64 bands named `A00`–`A63`, 10 m, one image
per calendar year, values in [-1, 1], CC-BY-4.0, catalog availability
2017-01-01 to 2025-01-01. Coverage explicitly includes intertidal and coastal
shallow water, which matters for a lagoon-and-fringe-mangrove site like CGSM.
Images are produced in their native UTM zone; CGSM sits entirely in 18N, so no
cross-zone mosaicking issue.

### 3.1 How it gets here
This environment has **no GPU access** (the laptop RTX 4060 is not visible to
the sandbox), no remote compute registered, and `earthengine.googleapis.com` is
**off the network allowlist** — and the Earth Engine Python API would need your
OAuth credentials anyway, which are not configured here.

So the clean path is the one you already use: **export from the Code Editor
yourself** and hand me the result. No weights, no GPU, no credentials, no
allowlist changes. `export_alphaearth_gee.js` does it.

**Export points, not the raster.** The full 64-band image over this AOI is
7150 × 6030 × 64 float32 ≈ **11 GB**, and the benchmark only ever reads 600
labelled pixels. The script's recommended path is two small CSVs: the embedding
sampled at the 600 label points, and a 4000-point random sample used solely to
measure the embedding's autocorrelation range for block sizing. This also
sidesteps the reprojection problem below entirely, since sampling happens in
AlphaEarth's native grid at your point coordinates. The raster export remains in
the script, disabled by default, for the case where you want a wall-to-wall
classified map — and for that it is cheaper to classify inside Earth Engine and
export the single-band result.

**EPSG:9377 needs a WKT, not a code.** Earth Engine does not reliably resolve
the code 9377, so the script passes the full WKT for MAGNA-SIRGAS 2018 /
Origen-Nacional (also saved as `epsg9377_wkt1.txt`). If a raster export is ever
needed, note that AlphaEarth is produced in native UTM, so exporting on the
9377 grid reprojects it — nearest-neighbour only. Bilinear would average
64-dimensional unit vectors across cell boundaries, yielding vectors that are no
longer unit-norm.

The alternative — me running the export from here — would need the allowlist
entry plus your Earth Engine credentials in this environment. Not worth it for
a one-off export.

### 3.2 Grid alignment is the thing to get right
Sampling two rasters at the same points only gives a paired comparison if the
grids coincide. Do not rely on `scale: 10` alone — a bare scale lets Earth
Engine choose its own origin, and a half-pixel shift silently mixes neighbouring
pixels into the embedding arm. Export with an explicit `crs` **and**
`crsTransform` copied from the Sentinel-2 raster (the inspection report prints
both). The script asserts this.

### 3.3 The caveat this design must carry
AlphaEarth embeddings are **annual and multi-sensor**: each vector summarises a
full calendar year across several instruments. Your Sentinel-2 features are a
single date or a single composite. So this benchmark is **not** a clean
representation-learning ablation — it cannot separate "learned embeddings are a
better representation" from "a year of multi-sensor data contains more
information than one Sentinel-2 date". It is an honest and useful *operational*
comparison — "which readily available feature set classifies CGSM better" — and
should be reported in those words.

Two consequences worth stating in the writeup:

- **Pick the embedding year to match the labels, not the image.** If the labels
  were interpreted from a specific date, use the embedding year containing that
  date. Mismatched years turn label noise into an apparent model difference.
- **Annual averaging smears episodic change.** CGSM's hypersalinity dieback
  events are exactly the signal an annual summary compresses. If the labels
  distinguish degraded from healthy mangrove, the embedding arm may lose that
  distinction for reasons that have nothing to do with representation quality.

If you later want the strict ablation, that needs a Sentinel-2-only encoder
(Prithvi, Clay) run on your own bands — approval for `huggingface.co` plus a
`torch` install, CPU-only here. Worth doing as a follow-up, not as a blocker.

---

## 4. What I need from you

**Blocking — nothing can run without these**
1. The two files: Sentinel-2 GeoTIFF and the labeled points. Either grant a
   folder (tell me the path) or attach them.
2. The AlphaEarth GeoTIFF, exported with `export_alphaearth_gee.js`. You need
   the Sentinel-2 grid numbers from step 1's inspection report to fill in its
   header, so the natural order is: send me the first two files, I return the
   report, you run the export, send me the third file.
3. The **embedding year** to use — the year containing the date your labels
   refer to (see §3.3).

**Needed to build the features correctly**
4. Processing level and compositing: L1C or L2A, single date or composite; if a
   composite, the reducer and date window. Acquisition date(s).
5. Band order / names in the export, and confirmation that green, red, NIR and
   SWIR1 are all present — NDVI needs red + NIR, NDWI needs green + NIR, MNDWI
   needs green + SWIR1. A missing SWIR1 means no MNDWI, which matters for
   separating water from wet soil in this landscape.
6. Nodata convention and whether a cloud/SCL mask came along or masking was
   already applied.
7. Reflectance scaling actually applied on export (÷10000 or not).
8. Whether the original Sentinel-2 export used `scale:` or an explicit
   `crsTransform` — if `scale:`, re-export both rasters through the same
   transform (§6 of the export script) so the two arms sample identical pixels.

**Needed to interpret the labels and the result**
9. Label provenance: field GPS or photo-interpretation, approximate positional
   accuracy, and the date they were collected relative to the image date.
10. Class definitions, specifically: does `mangrove` include degraded and
    dieback stands, which are extensive in CGSM from the hypersalinity events?
    Are salt flats and aquaculture ponds `bare soil` or `other`? `other` as a
    catch-all is usually the class that breaks these benchmarks.
11. Whether the points were digitised in clumps per class (the inspection report
    will show this, but your intent tells me whether it is by design).

**Scope**
12. Point-level metrics only, or also a wall-to-wall classified map for the AOI.
13. Whether you want area-adjusted accuracy (§2.9) for mangrove extent claims.
14. Whether the AOI is large enough for 5 spatially independent folds — the
    inspection report answers this once the embedding range is measured, and if
    the answer is no, whether you would rather enlarge the export or drop to
    leave-one-block-out.
