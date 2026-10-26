# -*- coding: utf-8 -*-
"""Aisla a GDAL de la instalacion de PROJ de PostgreSQL. IMPORTAR ANTES QUE RASTERIO.

CAUSA RAIZ del CRS roto de la serie. Esta maquina tiene la variable de entorno

    PROJ_LIB = C:\\Program Files\\PostgreSQL\\14\\share\\contrib\\postgis-3.5\\proj

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
"""
import os
import sys


def blindar():
    import importlib.util
    spec = importlib.util.find_spec("rasterio")
    if spec is None or not spec.submodule_search_locations:
        return None
    base = list(spec.submodule_search_locations)[0]
    datos = os.path.join(base, "proj_data")
    if not os.path.exists(os.path.join(datos, "proj.db")):
        return None
    anterior = os.environ.get("PROJ_LIB")
    os.environ["PROJ_LIB"] = datos
    os.environ["PROJ_DATA"] = datos
    return anterior, datos


_r = blindar()
if _r is None:
    print("AVISO: no se pudo localizar el proj_data de rasterio; "
          "si EPSG:9377 falla, revisa PROJ_LIB", file=sys.stderr)
else:
    _antes, _ahora = _r
    if _antes and os.path.normcase(_antes) != os.path.normcase(_ahora):
        print("PROJ_LIB redirigido para este proceso")
        print("   antes : %s" % _antes)
        print("   ahora : %s" % _ahora)
