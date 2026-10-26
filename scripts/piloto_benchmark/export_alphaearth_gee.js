/**
 * Export the Google Satellite Embedding (AlphaEarth Foundations) dataset
 * aligned pixel-for-pixel to an existing Sentinel-2 export.
 *
 * Ciénaga Grande de Santa Marta — RF-vs-embeddings benchmark.
 * Run in the Earth Engine Code Editor.
 *
 * Dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
 *   64 bands A00..A63, 10 m, one image per calendar year, values in [-1, 1],
 *   available 2017-01-01 .. 2025-01-01. Coverage includes intertidal and
 *   coastal shallow water. Produced in native UTM (CGSM is entirely 18N).
 *
 * WHY crsTransform AND NOT scale
 *   Passing `scale: 10` lets Earth Engine pick its own grid origin. A resulting
 *   half-pixel shift would mix neighbouring pixels into the embedding features
 *   but not into the Sentinel-2 features, producing a fake model difference.
 *   Pinning crs + crsTransform to the Sentinel-2 grid removes that risk.
 *   inspect_data.py verifies the alignment on the returned GeoTIFF.
 */

// ---------------------------------------------------------------------------
// 1. GRID — copy these from inspection_report.md, "Raster" section
// ---------------------------------------------------------------------------
// The thesis raster is in EPSG:9377 (MAGNA-SIRGAS 2018 / Origen-Nacional).
// Earth Engine does not reliably resolve this EPSG code, so the CRS is passed
// as WKT instead. This string is a REDUCED form of pyproj's WKT1 for EPSG:9377:
// the datum, spheroid, projection and all parameters are identical, but the
// six AUTHORITY[...] blocks have been stripped for readability. The complete
// unmodified WKT1 is saved separately as epsg9377_wkt1.txt; use that file if a
// tool needs the authority codes.
var CRS = 'PROJCS["MAGNA-SIRGAS 2018 / Origen-Nacional",' +
  'GEOGCS["MAGNA-SIRGAS 2018",DATUM["Marco_Geocentrico_Nacional_de_Referencia_2018",' +
  'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],' +
  'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],' +
  'PARAMETER["latitude_of_origin",4],PARAMETER["central_meridian",-73],' +
  'PARAMETER["scale_factor",0.9992],PARAMETER["false_easting",5000000],' +
  'PARAMETER["false_northing",2000000],UNIT["metre",1]]';

// If Earth Engine rejects the WKT, the fallback is to export in the native
// UTM 18N grid ('EPSG:32618') and reproject locally with rasterio using
// nearest-neighbour. Do not silently switch to `scale:` — see the header.

// PROVISIONAL — these are derived from the AOI polygon bbox, not from the
// raster, because s2_201903.tif was not reachable. Replace with transform[2],
// transform[5], width and height from inspection_report.md before exporting.
var ORIGIN_X = 4796860;        // report: transform[2]  (left edge, metres)
var ORIGIN_Y = 2784260;        // report: transform[5]  (top edge, metres)
var PIXEL    = 10;             // report: pixel size
var WIDTH    = 7150;           // report: width  (pixels)
var HEIGHT   = 6030;           // report: height (pixels)

// Embedding year. Use the year of the LABELS, not of the Sentinel-2 image.
// The weak labels come from ESA WorldCover v200, epoch 2021.
var YEAR = 2021;

// ---------------------------------------------------------------------------
// 2. Derived grid + region (no geodesic edges: this is a projected rectangle)
// ---------------------------------------------------------------------------
var crsTransform = [PIXEL, 0, ORIGIN_X, 0, -PIXEL, ORIGIN_Y];
var region = ee.Geometry.Rectangle(
  [ORIGIN_X, ORIGIN_Y - HEIGHT * PIXEL, ORIGIN_X + WIDTH * PIXEL, ORIGIN_Y],
  CRS, false);

// ---------------------------------------------------------------------------
// 3. Build the embedding image
// ---------------------------------------------------------------------------
var col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
  .filterBounds(region);

// mosaic(), not first(): the AOI can straddle two production tiles.
var emb = col.mosaic();

// DO NOT add .resample('bilinear') here. AlphaEarth is produced in native UTM,
// so exporting on the EPSG:9377 grid reprojects it; Earth Engine's default
// nearest-neighbour keeps each output pixel equal to one real embedding vector.
// Bilinear would average 64-dimensional unit vectors across a cell boundary,
// producing vectors that are no longer unit-norm and blurring class edges.

// ---------------------------------------------------------------------------
// 4. Pre-flight checks — read these in the Console BEFORE starting the task
// ---------------------------------------------------------------------------
print('images intersecting AOI for ' + YEAR + ' (expect >= 1):', col.size());
print('band count (expect 64):', emb.bandNames().size());
print('band names:', emb.bandNames());
print('AOI size (km):', WIDTH * PIXEL / 1000, 'x', HEIGHT * PIXEL / 1000);
print('crsTransform:', crsTransform);

// Visual sanity check: first 3 bands as RGB. Structure should track the
// landscape (lagoon, mangrove fringe, salt flats), not look like noise.
Map.centerObject(region, 11);
Map.addLayer(emb.select(['A01', 'A16', 'A09']), {min: -0.3, max: 0.3},
             'embedding A01/A16/A09');
Map.addLayer(ee.Image().paint(region, 0, 2), {palette: ['red']}, 'AOI');

// ---------------------------------------------------------------------------
// 5a. RECOMMENDED EXPORT — embeddings sampled at points (two small CSVs)
// ---------------------------------------------------------------------------
// The full 64-band raster over this AOI is ~11 GB (7150 x 6030 x 64 float32),
// which is neither necessary nor convenient for a point-based benchmark: only
// 600 labelled pixels are ever used. Sampling in Earth Engine returns two
// small CSVs and sidesteps the reprojection entirely.
//
// Upload etiquetas_debiles_worldcover_v1.geojson as a table asset first and put
// its id here.
var LABELS_ASSET = 'users/YOUR_USERNAME/etiquetas_debiles_worldcover_v1';
var labels = ee.FeatureCollection(LABELS_ASSET);

// (a) embeddings at the 600 labelled points. Keep 'clase' so the join is
//     verifiable, and keep the geometry so coordinates can be matched exactly.
var atLabels = emb.sampleRegions({
  collection: labels,
  properties: ['clase', 'clase_nombre'],
  scale: PIXEL,
  projection: emb.projection(),   // sample in AlphaEarth's native grid
  tileScale: 4,
  geometries: true
});
Export.table.toDrive({
  collection: atLabels,
  description: 'alphaearth_at_labels_' + YEAR,
  fileNamePrefix: 'alphaearth_at_labels_' + YEAR,
  fileFormat: 'CSV'
});

// (b) a random sample across the AOI, used only to measure the embedding's
//     spatial autocorrelation range, which sets the CV block size.
var randomPts = ee.FeatureCollection.randomPoints({
  region: region, points: 4000, seed: 42
});
Export.table.toDrive({
  collection: emb.sampleRegions({
    collection: randomPts, scale: PIXEL, projection: emb.projection(),
    tileScale: 4, geometries: true}),
  description: 'alphaearth_random_' + YEAR,
  fileNamePrefix: 'alphaearth_random_' + YEAR,
  fileFormat: 'CSV'
});

// ---------------------------------------------------------------------------
// 5b. OPTIONAL raster export — only needed for a wall-to-wall classified map
// ---------------------------------------------------------------------------
// ~11 GB over the full AOI. If you want the map, either clip to a smaller
// window first or do the classification inside Earth Engine and export only
// the single-band result.
Export.image.toDrive({
  image: emb.toFloat(),
  description: 'alphaearth_cgsm_' + YEAR,
  fileNamePrefix: 'alphaearth_cgsm_' + YEAR,
  crs: CRS,
  crsTransform: crsTransform,   // NB: never combine with `scale`
  region: region,
  maxPixels: 1e10,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

// ---------------------------------------------------------------------------
// 6. OPTIONAL — re-export the Sentinel-2 subset on the identical grid
// ---------------------------------------------------------------------------
// Only needed if the original Sentinel-2 export used `scale:` rather than an
// explicit crsTransform. Re-exporting both rasters through the same
// crsTransform guarantees the benchmark samples the same pixels in both arms.
//
// var s2 = /* your existing composite or single image */ null;
// Export.image.toDrive({
//   image: s2,
//   description: 's2_cgsm_aligned',
//   fileNamePrefix: 's2_cgsm_aligned',
//   crs: CRS,
//   crsTransform: crsTransform,
//   region: region,
//   maxPixels: 1e10,
//   fileFormat: 'GeoTIFF',
//   formatOptions: {cloudOptimized: true}
// });
