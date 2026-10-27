# Representaciones de modelos fundacionales frente a variables espectrales en el manglar de la Ciénaga Grande de Santa Marta: una comparación con validación espacial

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23709966.svg)](https://doi.org/10.5281/zenodo.23709966)

Código, resultados y figuras de un estudio desarrollado en la Maestría en Geomática de la Universidad Nacional de Colombia (2026).

**Autora:** Lina María Quintero Fonseca

## Descripción

El estudio evalúa el aporte predictivo de las representaciones anuales de AlphaEarth Foundations (2021, 64 dimensiones, 10 m) frente a variables espectrales de Sentinel-2 del mismo año para clasificar cuatro coberturas asociadas al manglar de la Ciénaga Grande de Santa Marta (CGSM): manglar, agua, humedal herbáceo y otras coberturas.

Se compararon siete configuraciones de entrada sobre 600 puntos, con el mismo Random Forest, las mismas particiones y las mismas métricas, bajo cuatro esquemas de evaluación: partición aleatoria, bloques de 6 km con zona de exclusión de 3 km, bloques de 16 km con zona de exclusión de 8 km y una reserva contigua de 132 puntos.

Las etiquetas de referencia se derivaron de ESA WorldCover 2021, de modo que las métricas cuantifican acuerdo con ese producto y no exactitud temática independiente.

El informe completo está en `informe/Informe1.md` y en PDF en la misma carpeta.

## Contenido del repositorio

| Carpeta o archivo | Contenido |
|---|---|
| `scripts/` | Los diez scripts de Python del informe, en el orden del flujo de trabajo. |
| `scripts/piloto_benchmark/` | El motor de validación, conservado sin modificaciones: inspección de datos, construcción de particiones espaciales, extracción de variables, clasificación, exportación de AlphaEarth y diseño experimental (`BENCHMARK_DESIGN.md`). |
| `resultados/` | Tablas CSV y JSON por experimento: métricas por esquema y por pliegue, matrices de confusión, comparaciones pareadas, McNemar, permutación por bloques, importancias de variables y pruebas de sensibilidad. |
| `figuras/` | Las cinco figuras del informe y los mapas clasificados en PNG. |
| `informe/` | Informe en Markdown y PDF. |
| `DOCUMENTACION_scripts.md` | Documentación redactada de cada script: propósito, entradas, procedimiento y salidas. |
| `DOCUMENTACION_cabeceras.md` | Cabeceras y comentarios de los scripts tal como están en el código, reunidos por `generar_documentacion.py`; conservan las notas de desarrollo. |

**Alcance de lo publicado.** El repositorio no incluye los archivos por punto (coordenadas, variables extraídas, predicciones fuera de pliegue ni registros de auditoría) ni los rásteres del cubo de datos. Permite examinar el código y los resultados agregados; reproducir íntegramente las pruebas estadísticas exige volver a correr el flujo desde las fuentes de datos, que son abiertas. Las 600 etiquetas se derivan de ESA WorldCover 2021 con un filtro de pureza de 3 × 3 píxeles y balance de 150 puntos por clase; su construcción se describe en el informe y en `scripts/piloto_benchmark/BENCHMARK_DESIGN.md`.

## Flujo de trabajo

1. `linea_base_2021.py`, línea base espectral: muestrea los doce compuestos mensuales de Sentinel-2 de 2021 en los 600 puntos, construye cinco variantes (marzo, estación seca, estación húmeda, seca + húmeda, seca + húmeda + descriptores fenológicos) y las evalúa en los cuatro esquemas.
2. `representaciones_2021.py`, comparación de representaciones: línea base multitemporal, AlphaEarth 2021 y la combinación de ambas; comparaciones pareadas por pliegue, McNemar fuera de pliegue, importancias y sensibilidad con un perceptrón multicapa.
3. `permutacion_bloques.py`, inferencia espacial: permutación pareada con el bloque espacial como unidad de intercambio e intervalos por remuestreo de bloques.
4. `auditoria_etiquetas.py` y `sensibilidad_etiquetas.py`, calidad de la referencia: contraste con Global Mangrove Watch 2020 y Dynamic World 2021, y repetición de la comparación principal sin los 77 puntos en conflicto.
5. `diez_pliegues.py`, sensibilidad a las particiones: el esquema de bloques de 6 km con diez pliegues en lugar de cinco, con los mismos puntos y zona de exclusión.
6. `mapa_clasificacion.py` y `mapa_desacuerdo_clases.py`, aplicación cartográfica: las tres configuraciones principales sobre el dominio con el Random Forest de Earth Engine, comprobación frente a scikit-learn en la reserva y caracterización del desacuerdo entre mapas.
7. `figura_representaciones.py` e `informe_a_word.py`: figura principal y conversión del informe a Word.

## Requisitos y reproducibilidad

Python 3.12 o superior y los paquetes de `requirements.txt`. Los scripts que consultan Google Earth Engine requieren una cuenta autenticada y un proyecto de Google Cloud con Earth Engine habilitado.

El código conserva rutas del entorno original de ejecución (`C:\LINA\UNAL 2626-1\...`) y depende de insumos que no se distribuyen aquí: los compuestos del cubo de datos y los archivos por punto. Para correr el flujo completo hay que disponer de esos insumos y adaptar las rutas. Los mapas se producen con el Random Forest de Earth Engine, una implementación equivalente en su configuración a la de scikit-learn pero no idéntica en todos sus cálculos; la diferencia se mide en la reserva y se declara en el informe.

Fuentes de datos, todas abiertas:

- Sentinel-2 L2A, mediante los compuestos mensuales del cubo de datos de la tesis.
- AlphaEarth Foundations (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`).
- ESA WorldCover 2021.
- Global Mangrove Watch v3.0 (2020).
- Dynamic World 2021.

## Cómo citar

Quintero Fonseca, L. M. (2026). *Representaciones de modelos fundacionales frente a variables espectrales en el manglar de la Ciénaga Grande de Santa Marta: una comparación con validación espacial* [Código y resultados]. GitHub. https://github.com/linaq11/cgsm-alphaearth-vs-espectral

## Licencia

MIT. Ver `LICENSE`.
