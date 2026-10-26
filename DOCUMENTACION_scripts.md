# Documentación de los scripts

Descripción de cada script del repositorio en el orden del flujo de trabajo: propósito, entradas, procedimiento y salidas. Los nombres de archivos, rutas, parámetros y resultados son los que usan los scripts. Las cabeceras originales de cada archivo, que conservan las notas de desarrollo, están reunidas en `DOCUMENTACION_cabeceras.md`.

Convenciones comunes: los 600 puntos son etiquetas débiles de ESA WorldCover 2021 (150 por clase: manglar, agua, humedal herbáceo y otro), seleccionadas en vecindades puras de 3 × 3 píxeles; la malla de trabajo es la del cubo de datos de la tesis, 10 m en MAGNA-SIRGAS 2018 / Origen Nacional (EPSG:9377); el clasificador es un Random Forest de 500 árboles con semilla 42 y ajuste interno de hiperparámetros por validación cruzada agrupada por bloque; los cuatro esquemas de evaluación son partición aleatoria en cinco pliegues, bloques de 6 km en cinco pliegues con zona de exclusión de 3 km, bloques de 16 km en tres pliegues con zona de exclusión de 8 km y una reserva contigua de 132 puntos evaluada una sola vez.

## 1. Motor de validación espacial (`scripts/piloto_benchmark/`)

Estos cinco archivos se escribieron para la comparación inicial de una sola fecha y los scripts del informe los importan sin modificarlos.

### `inspect_data.py`

Propósito. Diagnóstico previo al diseño de la validación espacial: describe el ráster y los puntos etiquetados y mide la geometría de la muestra.

Entradas. Un ráster de Sentinel-2 (`--raster`) y los puntos etiquetados en GeoJSON (`--labels`, campo de clase con `--label-field`).

Procedimiento. Lee el ráster de forma decimada y reporta CRS, extensión, bandas, tipo de dato y valor nulo. Reproyecta los puntos al CRS del ráster y revisa duplicados por píxel, puntos sobre valores nulos, distancia al vecino más cercano y dispersión por clase. Calcula variogramas empíricos del NDVI y de las primeras componentes principales de AlphaEarth, en la serie cruda sobre una ventana amplia y en la serie sin tendencia sobre una ventana más corta; aplica una prueba de meseta (el último tercio de la curva no debe superar al tercio central) y reporta por separado el alcance práctico, cuando existe meseta, y el codo de la curva, que localiza la estructura de corto alcance en un variograma no estacionario y no es un alcance basado en meseta. Comprueba la alineación entre puntos y malla.

Salidas. `<out>/inspection_report.md`, `<out>/inspection_report.json` y figuras de diagnóstico. No modifica ningún archivo fuera de `<out>/`.

### `build_split.py`

Propósito. Construye una única asignación de partición que consumen todas las configuraciones de entrada, de modo que la comparación queda emparejada punto a punto.

Entradas. La tabla de puntos con coordenadas y clase (`--labels`).

Procedimiento. Aparta una reserva contigua de alrededor del 22 % de los puntos, con las cuatro clases, construida por crecimiento de región compacta en 8-conectividad y compartida por todos los esquemas. Sobre los puntos restantes asigna bloques cuadrados completos a los pliegues por el criterio del mayor bloque primero al pliegue más pequeño, con un paso de reparación que garantiza que cada pliegue de prueba contenga las cuatro clases. La zona de exclusión, de la mitad del lado del bloque, se aplica en el momento de entrenar y no se escribe en el archivo, para que su ancho quede a la vista. Se emiten tres esquemas en lugar de uno porque los variogramas de las variables no permitieron fijar un solo tamaño de bloque: el NDVI y la primera componente principal de AlphaEarth no alcanzaron meseta, crudos ni sin tendencia, y las componentes menores solo la alcanzaron en algunas series (PC2 a 10,0 km cruda y 7,2 km sin tendencia; PC3 a 9,4 km sin tendencia). Un bloque del tamaño del mayor de esos alcances dejaría demasiado pocos bloques para cinco pliegues, de modo que los esquemas aleatorio, 6 km y 16 km convierten esa elección en un análisis de sensibilidad. Con bloques de 16 km el número de pliegues baja a tres porque el dominio solo produce unos 16 bloques. El alcance de los residuos de clasificación, que es lo que debe ser independiente entre pliegues, se mide después con `run_benchmark.residual_range`.

Salidas. `split/split_assignments.csv`, con el identificador de punto, la marca de reserva y el pliegue de cada esquema.

### `extract_spectral.py`

Propósito. Muestrea las variables espectrales de Sentinel-2 en los puntos.

Entradas. Un ráster de Sentinel-2 (`--raster`) y la asignación de partición (`--split`).

Procedimiento. Divide las reflectancias B2 a B12 por 10.000 y recalcula los índices desde las bandas, con definiciones explícitas: NDVI = (B8 − B4)/(B8 + B4), NDWI = (B3 − B8)/(B3 + B8) según McFeeters y MNDWI = (B3 − B11)/(B3 + B11) según Xu. El recálculo elimina la ambigüedad entre las convenciones de NDWI de McFeeters y de Gao en la banda almacenada. Los índices son invariantes al factor de escala, que solo afecta a las seis reflectancias. Compara el NDVI recalculado con la banda almacenada como comprobación.

Salidas. `spectral_features.csv`.

### `run_benchmark.py`

Propósito. Motor de la comparación: el mismo Random Forest sobre los mismos puntos y pliegues, cambiando solo el conjunto de variables de entrada.

Entradas. Las variables espectrales (`--spectral`), las dimensiones A00 a A63 de AlphaEarth (`--embedding`) y la asignación de partición (`--split`). Comprueba que la clase de cada punto coincide en las tres fuentes.

Procedimiento. Para cada esquema y cada configuración entrena y predice por pliegue, con los hiperparámetros elegidos por validación cruzada interna agrupada por bloque dentro del entrenamiento (número de variables por división entre la raíz cuadrada y la mitad del total; hoja mínima entre 1 y 2), de modo que el ajuste nunca ve el pliegue de prueba ni su zona de exclusión. Calcula exactitud global, kappa, F1 y recall y precisión por clase, y macro-F1. Aplica McNemar emparejado sobre píxeles de prueba idénticos (binomial exacta sobre los pares discordantes). La función `residual_range` reconstruye las coordenadas de cada predicción fuera de pliegue del esquema de 6 km, en el mismo orden en que se concatenaron los pliegues de prueba, y calcula el variograma del indicador de error de clasificación con `_variogram_core` (14 rezagos, ventana del 30 % de la extensión, sin quitar tendencia): el alcance de ese residuo es el que se compara con el lado del bloque.

Salidas. En la carpeta `--out`: métricas por esquema y configuración, métricas por pliegue, predicciones fuera de pliegue y alcances de los residuos.

### `_proj.py`

Propósito. Aísla a GDAL de la instalación de PROJ de PostgreSQL presente en el equipo de trabajo. Debe importarse antes que rasterio.

Procedimiento. La variable de entorno `PROJ_LIB` del sistema apunta a una `proj.db` de PostGIS con `DATABASE.LAYOUT.VERSION.MINOR = 2`, cuando PROJ 9.5 exige 6 o más; con ella GDAL no resuelve códigos EPSG modernos y escribe GeoTIFF sin proyección. El módulo redirige `PROJ_LIB` y `PROJ_DATA` al directorio que trae rasterio, solo dentro del proceso, sin tocar la configuración de la máquina. pyproj no se ve afectado porque usa su propio directorio.

## 2. Línea base espectral multitemporal de 2021

### `linea_base_2021.py`

Propósito. Construye la línea base espectral que resume el año 2021 completo, para que la comparación con AlphaEarth, que también es anual, no confunda el tipo de representación con la amplitud del periodo observado.

Entradas. Los compuestos mensuales del cubo publicados en Earth Engine (`projects/basic-buttress-338101/assets/cgsm_datacube_v1/s2/2021MM`, bandas B2 a B12 por 10.000, 10 m, EPSG:9377) y los 600 puntos con su partición (`PILOTO_BENCHMARK/split_assignments.csv`). Los doce meses se muestrean en una sola llamada.

Procedimiento. Variables: seis reflectancias (B2, B3, B4, B8, B11, B12) más NDVI, NDWI de McFeeters y MNDWI recalculados desde las bandas. Componentes temporales: mediana por banda de enero a marzo (estación seca), mediana de septiembre a noviembre (estación húmeda) e índices recalculados sobre cada mediana; descriptores fenológicos de los doce meses (mínimo, máximo, amplitud y desviación estándar de NDVI, NDWI y MNDWI, doce variables). Configuraciones, con el mismo clasificador, puntos y pliegues: `s2_marzo` (marzo de 2021, 9 variables, la configuración de la comparación inicial reproducida desde el cubo), `s2_seca` (9), `s2_humeda` (9), `s2_seca_humeda` (18) y `s2_multitemporal` (seca + húmeda + fenología, 30). Esquemas de evaluación: los cuatro del motor, con McNemar emparejado en la reserva. La opción `--solo-muestreo` ejecuta solo la extracción.

Salidas. En `linea_base_2021/`: `mensual_2021_crudo.csv`, `variables_puntos.csv`, `cv_metrics.csv`, `cv_per_fold.csv`, `holdout_metrics.csv`, matrices de confusión por esquema y configuración y `resumen.json`. En el repositorio se publican todas salvo las dos tablas por punto.

## 3. Representaciones frente a variables espectrales

### `representaciones_2021.py`

Propósito. Compara la línea base multitemporal con AlphaEarth 2021 y con la combinación de ambas fuentes, en las mismas condiciones.

Entradas. `linea_base_2021/variables_puntos.csv` (30 variables espectrales y partición) y `PILOTO_BENCHMARK/alphaearth2021_etiquetas.csv` (A00 a A63). Comprueba que ambas tablas provienen de la misma partición en el mismo orden antes de unirlas. Motor: `run_benchmark.py`.

Procedimiento. Responde cuatro preguntas: cuánto aporta AlphaEarth frente a una línea base que ya observa todo el año; si la combinación de las dos fuentes supera a AlphaEarth sola; si la ventaja sobrevive al aumentar la separación espacial (aleatorio, 6 km, 16 km, reserva); y si depende del clasificador, para lo cual repite el núcleo de la comparación con un perceptrón multicapa pequeño sin ajuste de hiperparámetros, a raíz de Ball et al. (2026). Configuraciones: `s2_marzo` (9 variables), `s2_seca_humeda` (18), `s2_multitemporal` (30, línea base), `alphaearth` (64) y `s2_multi_mas_alphaearth` (94). Calcula McNemar de cada configuración contra la línea base, las diferencias pareadas por pliegue y las importancias de variables por reducción media de impureza.

Salidas. En `representaciones_2021/`: `cv_metrics.csv`, `cv_per_fold.csv`, `cv_predictions.csv` (no publicada), `holdout_metrics.csv`, `mcnemar_holdout.json`, `importancias.json`, `mlp_sensibilidad.csv`, `pruebas_pareadas.json` y `resumen.json`.

### `permutacion_bloques.py`

Propósito. Repite las comparaciones pareadas con una prueba válida bajo dependencia espacial, porque McNemar trata cada punto como una observación independiente y los puntos vecinos no lo son.

Entradas. Las predicciones fuera de pliegue de `representaciones_2021.py`, `sensibilidad_etiquetas.py` y `diez_pliegues.py`. Para el análisis principal reconstruye la coordenada de cada fila de `cv_predictions.csv` con el mismo orden que usa `run_benchmark.residual_range` y lo comprueba contra `y_true`; las dos pruebas de sensibilidad guardan sus predicciones con coordenadas.

Procedimiento. La unidad que se permuta es el bloque espacial del propio esquema (6 o 16 km). En cada una de 20.000 repeticiones se intercambian al azar, bloque por bloque, las predicciones de la línea base y las de la otra configuración y se recalcula la diferencia de macro-F1 y de exactitud global; el intervalo de la diferencia se obtiene por remuestreo de bloques con reemplazo. No reentrena ningún modelo.

Salidas. `representaciones_2021/permutacion_bloques.json`, con las secciones `principal`, `sensibilidad_etiquetas` y `diez_pliegues`.

## 4. Pruebas de sensibilidad

### `auditoria_etiquetas.py`

Propósito. Mide cuánto ruido de etiqueta hay en las 600 etiquetas débiles, sin reetiquetar ningún punto.

Entradas. Global Mangrove Watch v3.0 de 2020 (vector local) y Dynamic World 2021 (moda anual de la banda `label`, en Earth Engine).

Procedimiento. Compara la clase manglar con la extensión de Global Mangrove Watch y las cuatro clases con la clase modal de Dynamic World, con una correspondencia de clases declarada antes de ver los resultados. Un punto se marca en conflicto cuando contradice a Global Mangrove Watch (manglar fuera de la extensión, o no manglar dentro) o cuando es agua que Dynamic World no ve como agua; Global Mangrove Watch decide para el manglar porque Dynamic World no tiene esa clase.

Salidas. `auditoria_etiquetas/cruce_dw.csv`, `puntos_auditados.csv` (no publicada) y `resumen.json`.

### `sensibilidad_etiquetas.py`

Propósito. Comprueba si las dos conclusiones centrales se sostienen con etiquetas más limpias.

Entradas. La columna `conflicto` de `auditoria_etiquetas.py` y las mismas variables, pliegues, reserva y motor del análisis principal.

Procedimiento. Repite la comparación central (línea base multitemporal, AlphaEarth y combinación) sin los 77 puntos en conflicto, con 523 puntos. Guarda la predicción por punto con coordenadas para la prueba por bloques. No reemplaza los resultados del informe.

Salidas. `sensibilidad_etiquetas/cv_metrics.csv`, matrices de confusión, `cv_predictions.csv` (no publicada) y `resumen.json`.

### `diez_pliegues.py`

Propósito. Repite el esquema principal con los mismos bloques de 6 km repartidos en diez pliegues en lugar de cinco.

Entradas. Las mismas tablas de `representaciones_2021.py` y la función `assign_blocked_folds` de `build_split.py` con k = 10 y semilla 42.

Procedimiento. Con cinco pliegues la prueba de signos no puede bajar de 0,0625, de modo que la evidencia estadística descansa en McNemar agrupado, cuyo supuesto de independencia entre puntos del mismo pliegue solo se cumple de forma aproximada. Con diez pliegues la prueba por pliegue puede alcanzar significancia por sí sola (mínimo 0,00195) y cada punto se predice con un modelo distinto. Se mantienen los puntos, la zona de exclusión de 3 km, los bloques de 6 km y el clasificador; cambian el reparto de bloques en pliegues y, con ello, el tamaño y la composición de cada conjunto de entrenamiento (mediana de 270 a 362 puntos).

Salidas. `diez_pliegues/cv_metrics.csv`, `cv_per_fold.csv`, matrices de confusión, `cv_predictions.csv` (no publicada) y `resumen.json`.

## 5. Aplicación espacial

### `mapa_clasificacion.py`

Propósito. Lleva las tres configuraciones principales a toda el área de estudio y produce el mapa de desacuerdo entre ellas, para mostrar dónde cambia el mapa y no solo cuánto cambia una métrica.

Entradas. El cubo en Earth Engine (`projects/basic-buttress-338101/assets/cgsm_datacube_v1/s2/2021MM`), AlphaEarth 2021 (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), `linea_base_2021/variables_puntos.csv` y el polígono del dominio (`AOI_tesis_CGSM_v1.geojson`).

Procedimiento. Un Random Forest de scikit-learn no puede aplicarse desde el equipo de trabajo a los 43 millones de píxeles de la ventana del cubo (7.150 por 6.022, de los que unos 20 millones caen dentro del dominio analítico de 2.042 km²), de modo que el mapa se produce con `ee.Classifier.smileRandomForest` en Earth Engine, entrenado con los mismos 429 puntos de la partición de la reserva (los 468 de validación cruzada menos los 39 situados a menos de 3 km de ella), las mismas variables y los hiperparámetros elegidos por la validación interna (raíz cuadrada de las variables, hoja mínima 2). No es el mismo modelo sino la misma receta en otro motor; el script mide esa brecha reclasificando los 132 puntos de la reserva con el modelo de Earth Engine y reportando el macro-F1 y el acuerdo punto a punto contra las predicciones de scikit-learn, cifra que va al pie de la figura. Dos diferencias de motor declaradas: la desviación de la fenología es muestral en Earth Engine (ddof = 1) y poblacional en numpy (ddof = 0), un factor de 1,044 sobre doce meses; y la mediana estacional de Earth Engine ignora los meses enmascarados, igual que `nanmedian`.

Salidas. `figuras/fig_mapa_clasificacion.png`, `mapa_clasificacion/validacion_gee_vs_sklearn.json` y las miniaturas crudas `mapa_clasificacion/*.png`, que se reutilizan entre corridas.

### `mapa_desacuerdo_clases.py`

Propósito. Identifica entre qué clases ocurre el desacuerdo entre los mapas de AlphaEarth y de la línea base multitemporal, y el área por clase de cada configuración. Usa el mismo entrenamiento que `mapa_clasificacion.py`.

Salidas. `mapa_clasificacion/desacuerdo_por_clases.json`.

## 6. Figuras e informe

### `figura_representaciones.py`

Propósito. Figura principal del informe: macro-F1 por configuración de entrada y esquema de evaluación, con la desviación entre pliegues como referencia de lo que es una diferencia apreciable.

Procedimiento. El panel (a) muestra el macro-F1 por configuración, con barra de error solo en el esquema principal (desviación entre los cinco pliegues espaciales) y etiquetas siempre por encima de la barra. El panel (b) muestra la diferencia pareada frente a la línea base, pliegue por pliegue, con la desviación de la diferencia y no la de cada configuración, porque los sectores difíciles lo son para las dos entradas y esa parte común no es incertidumbre de la diferencia; el valor p anotado es el de la permutación pareada por bloques y no el de McNemar. Los rótulos de la figura están en inglés.

Salidas. `figuras/fig_representaciones.png` y un resumen numérico para el texto.

### `informe_a_word.py`

Propósito. Convierte `Informe1.md` en `Informe1.docx`: tamaño carta, márgenes de 2,5 cm, Calibri 11, cuerpo justificado, encabezados en negrita sin numerar, tablas de Markdown con fila de encabezado sombreada, figuras a 5,2 pulgadas insertadas desde `![pie](ruta)` y pies y tablas a 8,5 puntos. El bloque de cabecera (líneas antes del primer `## `) va una por línea, con la última, el título, en negrita.
