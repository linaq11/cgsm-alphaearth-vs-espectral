# Cabeceras y comentarios de los scripts

Texto tal como está en el código, en el orden del flujo de trabajo; conserva las notas de desarrollo. Generado por `generar_documentacion.py`. La documentación redactada está en `DOCUMENTACION_scripts.md`.


## 1. Motor de validación (piloto, sin modificar)


### `piloto_benchmark/inspect_data.py`

```
Inspection report for a Sentinel-2 raster + labeled point set.

Reports everything needed before designing a spatially-blocked benchmark:
raster CRS / extent / bands / dtype / nodata, label class balance, and the
label-geometry diagnostics that decide whether a split is honest
(pixel-level duplicates, nearest-neighbour spacing, autocorrelation range).

Usage
-----
python inspect_data.py --raster S2_cgsm.tif --labels labels.geojson     [--label-field clase] [--out inspection] [--no-variogram]

Writes <out>/inspection_report.md, <out>/inspection_report.json and
diagnostic PNGs. Nothing is modified on disk apart from <out>/.
```


Comentarios en el código:

- línea 1: !/usr/bin/env python
- línea 29: Some conda builds leave PROJ_DATA unset, so GDAL cannot find proj.db and every
- línea 30: CRS lookup fails. Resolve it from pyproj before rasterio initialises GDAL.
- línea 37: pragma: no cover - fall through to rasterio's own defaults
- línea 47: pragma: no cover
- línea 50: Sentinel-2 L2A band-name hints, used only to annotate the report.
- línea 77: --------------------------------------------------------------------------
- línea 78: raster
- línea 79: --------------------------------------------------------------------------
- línea 107: extent in WGS84 + projected extent in km
- línea 121: decimated read so a large scene still fits in memory
- línea 154: scaling guess: L2A DN are ints ~0-10000, reflectance floats are 0-1
- línea 165: --------------------------------------------------------------------------
- línea 166: labels
- línea 167: --------------------------------------------------------------------------
- línea 178: fall back to the first low-cardinality non-geometry column
- línea 208: reproject to raster CRS for all geometric work
- línea 219: pixel indices -> duplicate / conflicting labels in one pixel
- línea 221: rowcol vectorises over coordinate arrays; DatasetReader.index does not.
- línea 241: points on nodata / masked pixels
- línea 249: nearest-neighbour spacing: exposes clumped digitising
- línea 262: per-class spatial spread
- línea 296: --------------------------------------------------------------------------
- línea 297: autocorrelation range (drives the block size of the spatial split)
- línea 298: --------------------------------------------------------------------------
- línea 325: Raw curve over a wide window, and the detrended curve over a narrower one.
- línea 326: The detrended residual is the interpretable estimate for block sizing; the
- línea 327: raw curve is kept so the non-stationarity is visible rather than hidden.
- línea 372: unreliable beyond ~half the extent
- línea 387: Plateau test: if the last third is still clearly above the middle third,
- línea 388: the curve is still climbing and no sill has been reached.
- línea 405: Knee of the curve: lag of maximum downward curvature. Reported separately
- línea 406: from practical_range_m because it is a different quantity - it locates the
- línea 407: short-range structure in a nested/non-stationary variogram, and it is NOT
- línea 408: a sill-based range. Useful when no plateau exists.
- línea 455: (n_pixels, n_bands)
- línea 457: PCA via SVD; embeddings are already unit-scaled so no variance scaling.
- línea 491: --------------------------------------------------------------------------
- línea 492: grid alignment + split sizing
- línea 493: --------------------------------------------------------------------------
- línea 564: --------------------------------------------------------------------------
- línea 565: figures + report
- línea 566: --------------------------------------------------------------------------
- línea 611: semivariance is normalised by the sill so the two panels are comparable
- línea 784: resolve red / NIR from band descriptions when not given
- línea 800: alignment is a correctness check, not a diagnostic: always run it


### `piloto_benchmark/build_split.py`

```
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
```


Comentarios en el código:

- línea 1: !/usr/bin/env python
- línea 46: pragma: no cover
- línea 49: MAGNA-SIRGAS 2018 / Origen-Nacional, metres
- línea 50: m, primary block edge
- línea 51: m, conservative sensitivity block edge
- línea 53: fraction of labels
- línea 57: --------------------------------------------------------------------------
- línea 77: --------------------------------------------------------------------------
- línea 102: candidate neighbours of the current region (8-connectivity)
- línea 113: keep the region compact: take the candidate nearest its centroid
- línea 132: --------------------------------------------------------------------------
- línea 156: Longest-processing-time-first: largest blocks placed first, each into the
- línea 157: fold that is currently smallest. Balancing totals is the primary objective
- línea 158: (a fold with 36 training points cannot support a model), class evenness the
- línea 159: secondary one.
- línea 170: Repair pass: every fold must contain all classes when used as a test set,
- línea 171: otherwise per-class metrics are undefined for that fold. Move a block that
- línea 172: supplies the missing class from a fold that can spare it.
- línea 215: --------------------------------------------------------------------------
- línea 267: k is per regime: 16 km blocks yield only ~16 blocks over the pool, so five
- línea 268: folds would leave ~3 blocks each and folds too small to train on.
- línea 295: sanity: no point in two folds, holdout never assigned a fold


### `piloto_benchmark/extract_spectral.py`

```
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
python extract_spectral.py --raster s2_202103.tif --split split/split_assignments.csv     --out spectral_features.csv
```


Comentarios en el código:

- línea 1: !/usr/bin/env python
- línea 41: pragma: no cover
- línea 84: cross-check the recomputed NDVI against the band shipped in the file: a
- línea 85: large discrepancy would mean the stored band used a different definition


### `piloto_benchmark/run_benchmark.py`

```
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
python run_benchmark.py --spectral spectral_features.csv     --embedding alphaearth2021_etiquetas.csv     --split split/split_assignments.csv --out results
```


Comentarios en el código:

- línea 1: !/usr/bin/env python
- línea 45: --------------------------------------------------------------------------
- línea 59: alignment check: the class must agree across all three sources
- línea 120: --------------------------------------------------------------------------
- línea 196: paired McNemar on identical test pixels (exact binomial on discordant pairs)
- línea 200: only b correct
- línea 201: only a correct
- línea 228: run_cv concatenated test folds in ascending fold order, so rebuilding
- línea 229: the same index order recovers each prediction's coordinates


### `piloto_benchmark/_proj.py`

```
Aisla a GDAL de la instalacion de PROJ de PostgreSQL. IMPORTAR ANTES QUE RASTERIO.

CAUSA RAIZ del CRS roto de la serie. Esta maquina tiene la variable de entorno

    PROJ_LIB = C:\Program Files\PostgreSQL\14\share\contrib\postgis-3.5\proj

puesta a nivel de sistema por el instalador de PostgreSQL. Esa carpeta trae una
`proj.db` con DATABASE.LAYOUT.VERSION.MINOR = 2, cuando PROJ 9.5 espera >= 6.
GDAL la respeta, no puede resolver ningun codigo EPSG moderno, y al escribir un
GeoTIFF cae a `LOCAL_CS` sin avisar. Por eso los 220 .tif quedaron sin datum ni
proyeccion.

pyproj no se ve afectado porque usa su propio directorio empaquetado, y de ahi
que `pyproj.CRS.from_epsg(9377)` funcione mientras `rasterio.crs.CRS.from_epsg(9377)`
revienta. Esa asimetria es la que hace el fallo dificil de ver.

Este modulo apunta PROJ_LIB y PROJ_DATA al directorio que trae rasterio, solo
dentro del proceso. No toca la configuracion de la maquina.

Arreglo permanente, aparte de esto: quitar PROJ_LIB de las variables de entorno
del sistema, o dejarla apuntando al directorio de PROJ que se quiera usar por
defecto. Conviene, porque cualquier otro guion basado en GDAL sigue expuesto.
```


## 2. Línea base espectral multitemporal de 2021


### `linea_base_2021.py`

```
Linea base espectral multitemporal 2021 para el informe 1 de Percepcion Remota
Avanzada.

PARA QUE. La propuesta del curso compara variables espectrales con representaciones
de AlphaEarth (anuales, 2021). El piloto de agosto uso una sola fecha (marzo de
2021), de modo que no podia separar "mejor representacion" de "mas tiempo
observado". Este guion construye la linea base espectral con estacion seca y
humeda de 2021 desde el cubo, con los MISMOS 600 puntos, pliegues y motor de
Random Forest del piloto, para que la comparacion del informe 2 sea justa.

ENTRADAS. Compuestos mensuales del cubo publicados en Earth Engine
(projects/basic-buttress-338101/assets/cgsm_datacube_v1/s2/2021MM, B2-B12 x10000,
10 m, EPSG:9377). Se muestrean los 12 meses de 2021 en los 600 puntos del piloto
(PILOTO_BENCHMARK/split_assignments.csv) en una sola llamada.

VARIABLES. Seis reflectancias (B2, B3, B4, B8, B11, B12) mas NDVI, NDWI (McFeeters)
y MNDWI recalculados desde las bandas, como en el piloto.
  seca   : mediana de enero, febrero y marzo por banda
  humeda : mediana de septiembre, octubre y noviembre por banda
  fenologia : minimo, maximo, amplitud y desviacion de NDVI, NDWI y MNDWI en los
              12 meses (12 variables)

CONFIGURACIONES (mismo clasificador, mismos puntos, mismos pliegues):
  s2_marzo         : marzo de 2021 solo, 9 variables (replica del piloto desde el cubo)
  s2_seca          : mediana seca, 9
  s2_humeda        : mediana humeda, 9
  s2_seca_humeda   : seca + humeda, 18
  s2_multitemporal : seca + humeda + fenologia, 30

REGIMENES DE VALIDACION (de run_benchmark.py): aleatorio (control con fuga),
bloques de 6 km con buffer de 3 km (principal), bloques de 16 km con buffer de
8 km (conservador) y reserva contigua evaluada una sola vez, con McNemar pareado.

ETIQUETAS. Las 600 debiles de WorldCover 2021 del piloto: la metrica es acuerdo
con WorldCover, no exactitud. La muestra de referencia propia llega en el informe 2.

USO
    python linea_base_2021.py [--out linea_base_2021] [--solo-muestreo]
```


Comentarios en el código:

- línea 98: marzo solo, replica del piloto
- línea 101: medianas estacionales por banda, indices recalculados desde la mediana
- línea 106: fenologia sobre los 12 meses


## 3. Representaciones frente a variables espectrales


### `representaciones_2021.py`

```
Comparacion de representaciones para el informe 1 de Percepcion Remota Avanzada.

PARA QUE. El piloto de agosto comparo variables espectrales de UNA fecha (marzo de
2021, 9 variables) contra las representaciones anuales de AlphaEarth (64), de modo
que no separaba "mejor representacion" de "mas tiempo observado". La linea base
multitemporal (linea_base_2021.py) arreglo el lado espectral pero no volvio a
correr AlphaEarth contra ella. Este guion pone las dos en la misma mesa: mismos
600 puntos, mismos pliegues, mismo Random Forest, y solo cambia la entrada.

PREGUNTAS QUE RESPONDE
  1. Ventaja: cuanto aporta AlphaEarth frente a la linea base espectral cuando la
     espectral ya observa todo el ano.
  2. Complementariedad: la union de las dos supera a AlphaEarth sola?
  3. Generalizacion: esa ventaja sobrevive al aumentar la separacion espacial
     (aleatorio -> bloques de 6 km -> bloques de 16 km -> reserva contigua)?
  4. Sensibilidad al clasificador: Ball et al. (2026) muestran que la ventaja de
     las representaciones exige un clasificador no lineal. Se repite el nucleo de
     la comparacion con un perceptron multicapa pequeno.

ENTRADAS
  informe1/linea_base_2021/variables_puntos.csv   30 variables espectrales + split
  PILOTO_BENCHMARK/alphaearth2021_etiquetas.csv   A00-A63 (Satellite Embedding 2021)
  Motor de validacion: PILOTO_BENCHMARK/run_benchmark.py (pliegues con buffer,
  ajuste de hiperparametros por CV interna agrupada, semilla 42).

SALIDAS (informe1/representaciones_2021/)
  cv_metrics.csv, cv_per_fold.csv, cv_predictions.csv, holdout_metrics.csv,
  mcnemar_holdout.json, importancias.json, mlp_sensibilidad.csv, resumen.json
```


Comentarios en el código:

- línea 56: 30, la linea base del informe
- línea 60: 9   replica del piloto
- línea 61: 18
- línea 62: 30  linea base
- línea 63: 64
- línea 64: 94  complementariedad
- línea 74: las dos tablas vienen del mismo split en el mismo orden; se verifica antes de unir
- línea 113: McNemar pareado de cada configuracion contra la linea base espectral


### `permutacion_bloques.py`

```
Rehace las comparaciones pareadas del Informe 1 con una prueba valida bajo dependencia espacial.

McNemar sobre las predicciones fuera de pliegue reunidas trata cada punto como una prueba
independiente, y los puntos vecinos no lo son: su p sale optimista. Aqui la unidad que se
permuta es el bloque espacial del propio regimen (6 km o 16 km):

  - permutacion pareada: en cada repeticion se intercambian al azar, bloque por bloque, las
    predicciones de la linea base y las del otro brazo, y se recalcula la diferencia;
  - intervalo: remuestreo de bloques con reemplazo.

No reentrena nada. Para el analisis principal reconstruye a que punto corresponde cada fila
de cv_predictions.csv con el mismo orden que usa run_benchmark.residual_range, y lo comprueba
con y_true. Las dos pruebas de sensibilidad guardan ya sus predicciones con coordenadas.

USO: python permutacion_bloques.py
```


Comentarios en el código:

- línea 82: a hace de base, c de otro


## 4. Pruebas de sensibilidad


### `auditoria_etiquetas.py`

```
Auditoria de las 600 etiquetas debiles (WorldCover 2021) contra dos productos
independientes, sin reetiquetar nada:
  - Global Mangrove Watch v3.0, 2020 (vector local): la clase manglar
  - Dynamic World 2021, moda anual de la banda label: las cuatro clases
Mide cuanto ruido de etiqueta hay, no corrige la muestra.
Salida: auditoria_etiquetas/cruce_dw.csv, resumen.json
```


Comentarios en el código:

- línea 25: 1. GMW 2020
- línea 32: 2. Dynamic World 2021, moda anual
- línea 48: acuerdo con un mapeo declarado de antemano
- línea 53: conflicto declarado antes de ver resultados: contradice a GMW (manglar fuera, o no manglar
- línea 54: dentro) o es agua que Dynamic World no ve como agua. GMW manda para manglar porque
- línea 55: Dynamic World no tiene esa clase.


### `sensibilidad_etiquetas.py`

```
Prueba de sensibilidad del informe 1 a la calidad de las etiquetas.

Repite la comparacion central (linea base multitemporal, AlphaEarth, union) sin los
puntos cuya etiqueta WorldCover contradice a Global Mangrove Watch 2020 o, para agua,
a Dynamic World 2021 (auditoria_etiquetas.py, columna 'conflicto'). Mismos pliegues,
misma reserva, mismo motor. No reemplaza los resultados del informe: comprueba si
las dos conclusiones centrales se sostienen con etiquetas mas limpias.
Salida: sensibilidad_etiquetas/
```


Comentarios en el código:

- línea 58: prediccion por punto, para la prueba por bloques


### `diez_pliegues.py`

```
Informe 1: repetir el regimen principal con DIEZ pliegues de 6 km en vez de cinco.

PARA QUE. Con cinco pliegues la prueba de signos no puede bajar de 0,0625, de modo que
la evidencia estadistica del informe descansa en el McNemar agrupado, cuyo supuesto de
independencia entre puntos de un mismo pliegue solo se cumple de forma aproximada. Con
diez pliegues la prueba por pliegue puede alcanzar significancia por si sola (minimo
0,00195) y, ademas, cada punto se predice con un modelo distinto, lo que reduce la
dependencia entre las predicciones que entran al McNemar.

Mismos puntos, mismo buffer de 3 km, mismos bloques de 6 km, mismo Random Forest: lo
unico que cambia es a cuantos pliegues se reparten los bloques (assign_blocked_folds de
build_split.py con k=10, semilla 42).

Salida: diez_pliegues/ (cv_metrics.csv, resumen.json)
```


Comentarios en el código:

- línea 58: prediccion por punto, para la prueba por bloques


## 5. Aplicación espacial


### `mapa_clasificacion.py`

```
Mapa de clasificacion del informe 1: las tres configuraciones de entrada
llevadas a toda el area de estudio, y el mapa de desacuerdo entre ellas.

PARA QUE. El informe compara representaciones con metricas puntuales sobre 600
puntos. Una cifra de macro-F1 no dice DONDE cambia el mapa. Este guion lleva las
tres configuraciones al area completa para mostrar que la ventaja de 0,038 de
AlphaEarth no se reparte de forma uniforme, sino que se concentra en los bordes
del manglar y en el humedal herbaceo.

ADVERTENCIA DE IMPLEMENTACION. Las cifras del informe salen de scikit-learn sobre
los 600 puntos. Un Random Forest de scikit-learn no se puede aplicar a un raster
de 43 millones de pixeles desde este equipo, de modo que el mapa se produce con
ee.Classifier.smileRandomForest en Earth Engine, entrenado con los MISMOS 429
puntos con buffer, las MISMAS variables y los mismos hiperparametros elegidos por
la validacion interna (sqrt, hoja minima 2). No es el mismo modelo, es la misma
receta en otro motor. El guion mide esa brecha: reclasifica los 132 puntos de la
reserva con el modelo de Earth Engine y reporta el macro-F1 y el acuerdo punto a
punto contra las predicciones de scikit-learn. Esa cifra va al pie de la figura.

Dos diferencias menores de motor, declaradas: la desviacion de la fenologia es
muestral en Earth Engine (ddof=1) y poblacional en numpy (ddof=0), un factor de
1,044 sobre doce meses; y la mediana estacional de Earth Engine ignora meses
enmascarados igual que nanmedian.

ENTRADAS
  cubo:        projects/basic-buttress-338101/assets/cgsm_datacube_v1/s2/2021MM
  AlphaEarth:  GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, 2021
  puntos:      informe1/linea_base_2021/variables_puntos.csv
  AOI:         AVANCE TESIS/AOI/AOI_tesis_CGSM_v1.geojson

SALIDAS
  figuras/fig_mapa_clasificacion.png
  mapa_clasificacion/validacion_gee_vs_sklearn.json
  mapa_clasificacion/*.png  (miniaturas crudas, se reutilizan entre corridas)
```


Comentarios en el código:

- línea 65: 30
- línea 81: --------------------------------------------------------------------------
- línea 82: 1. Pila de variables en Earth Engine, con las mismas definiciones del informe
- línea 83: --------------------------------------------------------------------------
- línea 115: --------------------------------------------------------------------------
- línea 116: 2. Los mismos 429 puntos de entrenamiento que usa reserva() en el informe
- línea 117: --------------------------------------------------------------------------
- línea 119: misma union verificada (coordenadas y clases) que usa el informe
- línea 140: --------------------------------------------------------------------------
- línea 141: 3. Predicciones de scikit-learn en la reserva, para medir la brecha de motor
- línea 142: --------------------------------------------------------------------------
- línea 159: --------------------------------------------------------------------------
- línea 160: 4. Miniaturas
- línea 161: --------------------------------------------------------------------------
- línea 208: brecha de motor: reclasificar la reserva y comparar con scikit-learn
- línea 240: fraccion del area donde las dos representaciones discrepan


### `mapa_desacuerdo_clases.py`

```
Entre que clases ocurre el desacuerdo AlphaEarth vs S2 multitemporal del mapa, y
area por clase de cada configuracion. Mismo entrenamiento que mapa_clasificacion.py.
Salida: mapa_clasificacion/desacuerdo_por_clases.json
```


## 6. Figuras e informe


### `figura_representaciones.py`

```
Figura principal del informe 1: macro-F1 por configuracion de entrada y regimen de
validacion, con la desviacion entre pliegues como referencia de lo que es una diferencia
apreciable. Figuras en ingles. Salida: figuras/fig_representaciones.png
```


Comentarios en el código:

- línea 44: barra de error solo en el regimen principal: desviacion entre los cinco pliegues espaciales
- línea 48: etiquetas horizontales, siempre por encima de la barra y de su barra de error
- línea 62: panel b: diferencia PAREADA frente a la linea base, pliegue por pliegue. La barra es la
- línea 63: sd de la diferencia, no la sd de cada configuracion: los sectores dificiles lo son para
- línea 64: las dos entradas y esa parte comun no es incertidumbre de la diferencia.
- línea 76: p de la permutacion pareada por bloques (permutacion_bloques.py), no el de McNemar,
- línea 77: que trata como independientes puntos vecinos
- línea 95: resumen numerico para el texto


### `informe_a_word.py`

```
Convierte Informe1.md en Informe1.docx: carta, margenes 2,5 cm, Calibri 11,
cuerpo justificado, encabezados en negrita sin numerar, tablas markdown con fila
de encabezado sombreada y figuras insertadas desde ![pie](ruta).

El bloque de cabecera (lineas antes del primer "## ") va una por linea, sin
negrita, salvo la ultima, que es el titulo.

USO
    python informe_a_word.py
```
