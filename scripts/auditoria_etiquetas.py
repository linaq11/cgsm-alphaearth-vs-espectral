# -*- coding: utf-8 -*-
"""Auditoria de las 600 etiquetas debiles (WorldCover 2021) contra dos productos
independientes, sin reetiquetar nada:
  - Global Mangrove Watch v3.0, 2020 (vector local): la clase manglar
  - Dynamic World 2021, moda anual de la banda label: las cuatro clases
Identifica discrepancias entre las etiquetas de WorldCover y dos productos cartograficos externos; no corrige la muestra.
Salida: auditoria_etiquetas/cruce_dw.csv, resumen.json
"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

AQUI = Path(__file__).resolve().parent
AVANCE = Path(r"C:\LINA\UNAL 2626-1\AVANCE TESIS")
SAL = AQUI / "auditoria_etiquetas"
SAL.mkdir(exist_ok=True)
DW = ["water", "trees", "grass", "flooded_vegetation", "crops",
      "shrub_and_scrub", "built", "bare", "snow_and_ice"]

df = pd.read_csv(AQUI / "linea_base_2021" / "variables_puntos.csv")[["point_id", "lon", "lat", "clase_nombre"]]
pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)

# 1. GMW 2020
gmw = gpd.read_file("zip://" + str(AVANCE / "PROYECTO" / "gmw_v3_2020_vec.zip"))
gmw = (gmw.set_crs(4326) if gmw.crs is None else gmw.to_crs(4326))
gmw = gmw.cx[pts.total_bounds[0] - .05:pts.total_bounds[2] + .05, pts.total_bounds[1] - .05:pts.total_bounds[3] + .05]
pts["en_gmw"] = pts.sjoin(gmw[["geometry"]], how="left", predicate="within").groupby(level=0).index_right.count().gt(0)
gmw_tab = pts.groupby("clase_nombre").en_gmw.mean().round(3)

# 2. Dynamic World 2021, moda anual
import ee
ee.Initialize(project="basic-buttress-338101")
fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a), float(b)]), {"pid": int(p)})
                           for a, b, p in zip(df.lon, df.lat, df.point_id)])
col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate("2021-01-01", "2022-01-01")
       .filterBounds(fc.geometry()).select("label"))
moda = col.reduce(ee.Reducer.mode()).rename("dw")
n_obs = col.count().rename("n")
got = moda.addBands(n_obs).reduceRegions(fc, ee.Reducer.first(), 10).getInfo()["features"]
dw = pd.DataFrame([f["properties"] for f in got]).rename(columns={"pid": "point_id"})
pts = pts.merge(dw, on="point_id", how="left")
pts["dw_clase"] = pts.dw.map(lambda v: DW[int(v)] if pd.notna(v) else "sin_dato")
cruce = pd.crosstab(pts.clase_nombre, pts.dw_clase)
cruce.to_csv(SAL / "cruce_dw.csv")

# acuerdo con un mapeo declarado de antemano
MAPEO = {"agua": {"water"}, "manglar": {"trees", "flooded_vegetation"},
         "humedal_herbaceo": {"flooded_vegetation", "grass"},
         "otro": {"grass", "crops", "shrub_and_scrub", "built", "bare", "trees"}}
pts["acuerdo_dw"] = [d in MAPEO[c] for c, d in zip(pts.clase_nombre, pts.dw_clase)]
# conflicto declarado antes de ver resultados: contradice a GMW (manglar fuera, o no manglar
# dentro) o es agua que Dynamic World no ve como agua. GMW manda para manglar porque
# Dynamic World no tiene esa clase.
pts["conflicto"] = (((pts.clase_nombre == "manglar") & ~pts.en_gmw)
                    | ((pts.clase_nombre != "manglar") & pts.en_gmw)
                    | ((pts.clase_nombre == "agua") & (pts.dw_clase != "water")))
pts.drop(columns="geometry").to_csv(SAL / "puntos_auditados.csv", index=False)
print("conflictos por clase:", pts.groupby("clase_nombre").conflicto.sum().to_dict(), "total", int(pts.conflicto.sum()))
res = {"gmw_fraccion_dentro_por_clase": gmw_tab.to_dict(),
       "dw_acuerdo_por_clase": pts.groupby("clase_nombre").acuerdo_dw.mean().round(3).to_dict(),
       "dw_observaciones_mediana": float(pts.n.median()),
       "mapeo": {k: sorted(v) for k, v in MAPEO.items()}}
(SAL / "resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print("GMW 2020, fraccion de puntos dentro del manglar por clase WorldCover:\n", gmw_tab.to_string())
print("\nDynamic World 2021 (moda) por clase WorldCover:\n", cruce.to_string())
print("\nacuerdo con el mapeo:", res["dw_acuerdo_por_clase"], "| obs DW mediana:", res["dw_observaciones_mediana"])
