# Representaciones de modelos fundacionales frente a variables espectrales en el manglar de la Ciénaga Grande de Santa Marta: una comparación con validación espacial

Código, resultados tabulados y figuras del Informe 1 del curso *Percepción Remota Avanzada* de la Maestría en Geomática de la Universidad Nacional de Colombia (2026-2).

**Autora:** Lina María Quintero Fonseca

El estudio mide si las representaciones anuales de AlphaEarth Foundations (2021, 64 dimensiones, 10 m) aportan información predictiva frente a variables espectrales de Sentinel-2 del mismo año para clasificar cuatro coberturas asociadas al manglar de la Ciénaga Grande de Santa Marta (CGSM): manglar, agua, humedal herbáceo y otro. Se comparan siete configuraciones de entrada sobre los mismos 600 puntos, con el mismo Random Forest, los mismos pliegues y las mismas métricas, bajo cuatro esquemas de evaluación espacial: partición aleatoria, bloques de 6 km con zona de exclusión de 3 km, bloques de 16 km con zona de exclusión de 8 km y una reserva contigua de 132 puntos. La referencia son 600 etiquetas débiles de ESA WorldCover 2021, de modo que las métricas expresan acuerdo con ese producto y no exactitud temática independiente.

El informe completo está en `informe/Informe1.md` y en PDF en la misma carpeta.

## Contenido del repositorio

| Carpeta | Contenido |
|---|---|
| `scripts/` | Los diez scripts de Python del informe, en el orden del flujo de trabajo. Cada uno documenta en su cabecera su propósito, sus entradas y sus salidas. |
| `scripts/piloto_benchmark/` | El motor de validación que los anteriores importan sin modificar: `build_split.py` (reserva contigua, pliegues por bloques con zona de exclusión), `run_benchmark.py` (Random Forest con ajuste interno por bloque y métricas), `extract_spectral.py`, `inspect_data.py`, `export_alphaearth_gee.js` (exportación de AlphaEarth 2021 en los puntos) y `BENCHMARK_DESIGN.md`. |
| `resultados/` | Tablas CSV y JSON generadas por los scripts, por experimento: línea base espectral (`linea_base_2021/`), representaciones (`representaciones_2021/`), sensibilidad a las etiquetas (`sensibilidad_etiquetas/`), diez pliegues (`diez_pliegues/`), auditoría de etiquetas (`auditoria_etiquetas/`) y mapas (`mapa_clasificacion/`). Incluyen matrices de confusión por régimen, métricas por pliegue, pruebas pareadas, McNemar, permutación por bloques e importancias de variables. |
| `figuras/` | Las cinco figuras del informe y los mapas clasificados en PNG. |
| `informe/` | Informe en Markdown y PDF. |
| `DOCUMENTACION_scripts.md` | Cabeceras y comentarios de todos los scripts, reunidos por `generar_documentacion.py`. |

No se incluyen los archivos por punto (coordenadas, variables extraídas, predicciones fuera de pliegue ni auditoría punto a punto) ni los rásteres del cubo de datos. Las 600 etiquetas débiles se derivan de ESA WorldCover 2021 con un filtro de pureza de 3 × 3 píxeles y balance de 150 puntos por clase; su construcción se describe en el informe y en `scripts/piloto_benchmark/BENCHMARK_DESIGN.md`.

## Flujo de trabajo

1. `linea_base_2021.py`: muestrea los doce compuestos mensuales de Sentinel-2 de 2021 en los 600 puntos, construye las cinco variantes espectrales (marzo, seca, húmeda, seca + húmeda, seca + húmeda + fenología) y las evalúa en los cuatro esquemas.
2. `representaciones_2021.py`: compara la línea base multitemporal con AlphaEarth 2021 y con la combinación de ambas; pruebas pareadas por pliegue, McNemar fuera de pliegue, importancias y sensibilidad con un perceptrón multicapa.
3. `permutacion_bloques.py`: repite las comparaciones con una prueba de permutación pareada cuya unidad es el bloque espacial, con intervalo por remuestreo de bloques.
4. `auditoria_etiquetas.py` y `sensibilidad_etiquetas.py`: contrastan las etiquetas con Global Mangrove Watch 2020 y Dynamic World 2021 y repiten la comparación sin los 77 puntos en conflicto.
5. `diez_pliegues.py`: repite el régimen principal con los mismos bloques repartidos en diez pliegues.
6. `mapa_clasificacion.py` y `mapa_desacuerdo_clases.py`: llevan las tres configuraciones principales a toda el área de estudio con el Random Forest de Earth Engine y miden el desacuerdo entre mapas.
7. `figura_representaciones.py` e `informe_a_word.py`: figura principal y conversión del informe a Word.

## Requisitos

Python 3.12 o superior y los paquetes de `requirements.txt`. Los scripts que consultan Earth Engine requieren una cuenta autenticada y un proyecto de Google Cloud. Los datos de entrada son productos abiertos: Sentinel-2 L2A (compuestos mensuales del cubo de datos de la tesis), AlphaEarth Foundations (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), ESA WorldCover 2021, Global Mangrove Watch v3.0 y Dynamic World.

## Cómo citar

Quintero Fonseca, L. M. (2026). *Representaciones de modelos fundacionales frente a variables espectrales en el manglar de la Ciénaga Grande de Santa Marta: una comparación con validación espacial* [Código y datos]. GitHub. https://github.com/linaq11/cgsm-alphaearth-vs-espectral

## Licencia

MIT. Ver `LICENSE`.
