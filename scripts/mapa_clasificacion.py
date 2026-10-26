# -*- coding: utf-8 -*-
r"""Mapa de clasificacion del informe 1: las tres configuraciones de entrada
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
"""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

AQUI = Path(__file__).resolve().parent
AVANCE = Path(r"C:\LINA\UNAL 2626-1\AVANCE TESIS")
SALIDA = AQUI / "mapa_clasificacion"
SALIDA.mkdir(exist_ok=True)
(AQUI / "figuras").mkdir(exist_ok=True)
sys.path.insert(0, str(AVANCE / "PILOTO_BENCHMARK"))

PROYECTO = "basic-buttress-338101"
RAIZ = "projects/basic-buttress-338101/assets/cgsm_datacube_v1/s2/"
BANDAS = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDICES = ["NDVI", "NDWI", "MNDWI"]
MESES = ["2021%02d" % m for m in range(1, 13)]
SECA = ["202101", "202102", "202103"]
HUMEDA = ["202109", "202110", "202111"]

F_SECA = ["seca_" + c for c in BANDAS + INDICES]
F_HUM = ["hum_" + c for c in BANDAS + INDICES]
F_FEN = ["fen_%s_%s" % (c, s) for c in INDICES for s in ("min", "max", "amp", "sd")]
F_S2 = F_SECA + F_HUM + F_FEN          # 30
F_AE = ["A%02d" % i for i in range(64)]

ARMS = {
    "s2_multitemporal": F_S2,
    "alphaearth": F_AE,
    "s2_multi_mas_alphaearth": F_S2 + F_AE,
}
ETQ = {"s2_multitemporal": "(a) S2 multi-temporal (30 vars)",
       "alphaearth": "(b) AlphaEarth (64 vars)",
       "s2_multi_mas_alphaearth": "(c) S2 + AlphaEarth (94 vars)"}
CLASES = ["agua", "humedal_herbaceo", "manglar", "otro"]
PAL = ["1f6f78", "c9b458", "1b7a3e", "c6c1b8"]
ETQ_CLASE = ["Water", "Herbaceous wetland", "Mangrove", "Other"]


# --------------------------------------------------------------------------
# 1. Pila de variables en Earth Engine, con las mismas definiciones del informe
# --------------------------------------------------------------------------
def _mes(ee, m):
    im = ee.Image(RAIZ + m).select(BANDAS).divide(10000)
    return im.addBands([im.normalizedDifference(["B8", "B4"]).rename("NDVI"),
                        im.normalizedDifference(["B3", "B8"]).rename("NDWI"),
                        im.normalizedDifference(["B3", "B11"]).rename("MNDWI")])


def _estacional(ee, meses, pref):
    """Mediana por banda sobre los tres meses; los indices se recalculan DESDE la
    mediana, no se promedian, igual que construir_variables() en linea_base_2021."""
    med = ee.ImageCollection([_mes(ee, m).select(BANDAS) for m in meses]).median()
    med = med.addBands([med.normalizedDifference(["B8", "B4"]).rename("NDVI"),
                        med.normalizedDifference(["B3", "B8"]).rename("NDWI"),
                        med.normalizedDifference(["B3", "B11"]).rename("MNDWI")])
    return med.rename([pref + c for c in BANDAS + INDICES])


def pila(ee):
    col = ee.ImageCollection([_mes(ee, m).select(INDICES) for m in MESES])
    fen = []
    for c in INDICES:
        s = col.select(c)
        mn, mx = s.min().rename("fen_%s_min" % c), s.max().rename("fen_%s_max" % c)
        fen += [mn, mx, mx.subtract(mn).rename("fen_%s_amp" % c),
                s.reduce(ee.Reducer.stdDev()).rename("fen_%s_sd" % c)]
    s2 = _estacional(ee, SECA, "seca_").addBands(_estacional(ee, HUMEDA, "hum_")).addBands(fen)
    ae = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
          .filterDate("2021-01-01", "2022-01-01").mosaic().select(F_AE))
    return s2.addBands(ae).toFloat()


# --------------------------------------------------------------------------
# 2. Los mismos 429 puntos de entrenamiento que usa reserva() en el informe
# --------------------------------------------------------------------------
def particion():
    # misma union verificada (coordenadas y clases) que usa el informe
    sys.path.insert(0, str(AQUI))
    import representaciones_2021 as rep
    df = rep.cargar()
    hold = df["holdout"].to_numpy().astype(bool)
    xy = df[["x", "y"]].to_numpy()
    d = np.sqrt(((xy[~hold][:, None, :] - xy[hold][None, :, :]) ** 2).sum(-1)).min(1)
    tr = np.zeros(len(df), bool)
    tr[np.nonzero(~hold)[0][d > 3000.0]] = True
    print("entrenamiento %d puntos, reserva %d" % (tr.sum(), hold.sum()), flush=True)
    return df, tr, hold


def fc(ee, df, sel):
    s = df[sel]
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(a), float(b)]),
                   {"clase": int(CLASES.index(c)), "pid": int(p)})
        for a, b, c, p in zip(s.lon, s.lat, s.clase_nombre, s.point_id)])


# --------------------------------------------------------------------------
# 3. Predicciones de scikit-learn en la reserva, para medir la brecha de motor
# --------------------------------------------------------------------------
def sklearn_reserva(df, tr, hold):
    import run_benchmark as rb
    from sklearn.metrics import f1_score
    y = df["clase_nombre"].to_numpy()
    xy = df[["x", "y"]].to_numpy()
    out = {}
    for arm, feats in ARMS.items():
        X = df[feats].to_numpy()
        model, _ = rb.tune_and_fit(X[tr], y[tr], xy[tr], rb.BLOCK_L["block_6km"])
        p = model.predict(X[hold])
        out[arm] = p
        print("  sklearn %-24s macro-F1 %.3f" % (arm, f1_score(y[hold], p, average="macro",
                                                               zero_division=0)), flush=True)
    return out


# --------------------------------------------------------------------------
# 4. Miniaturas
# --------------------------------------------------------------------------
def bajar(img, region, destino, anchos=(1100, 850, 650, 500)):
    if destino.exists():
        print("  reutiliza", destino.name, flush=True)
        return destino
    from PIL import Image
    for w in anchos:
        try:
            url = img.getThumbURL({"region": region, "dimensions": w, "format": "png"})
            r = requests.get(url, timeout=600)
            if not r.headers.get("content-type", "").startswith("image"):
                raise RuntimeError(r.text[:200])
            Image.open(io.BytesIO(r.content)).verify()
            destino.write_bytes(r.content)
            print("  %s a %d px" % (destino.name, w), flush=True)
            return destino
        except Exception as e:
            print("  ancho %d falla: %s" % (w, str(e)[:130]), flush=True)
    raise SystemExit("ABORTA: no se pudo descargar " + destino.name)


def main():
    import ee
    ee.Initialize(project=PROYECTO)
    aoi_gj = json.loads((AVANCE / "AOI" / "AOI_tesis_CGSM_v1.geojson").read_text(encoding="utf-8"))
    geom = aoi_gj["features"][0]["geometry"] if "features" in aoi_gj else aoi_gj
    aoi = ee.Geometry(geom, None, False)
    print("area del AOI: %.0f km2" % (aoi.area(1).getInfo() / 1e6), flush=True)

    df, tr, hold = particion()
    st = pila(ee).clip(aoi)
    ent, res = fc(ee, df, tr), fc(ee, df, hold)

    mapas, val = {}, {}
    y_hold = df.loc[hold, "clase_nombre"].to_numpy()
    print("\n== scikit-learn en la reserva (referencia del informe) ==", flush=True)
    sk = sklearn_reserva(df, tr, hold)

    print("\n== Earth Engine ==", flush=True)
    from sklearn.metrics import f1_score
    for arm, feats in ARMS.items():
        muestra = st.select(feats).sampleRegions(collection=ent, properties=["clase"],
                                                 scale=10, tileScale=4)
        clf = (ee.Classifier.smileRandomForest(numberOfTrees=500, minLeafPopulation=2, seed=42)
               .train(features=muestra, classProperty="clase", inputProperties=feats))
        cls = st.select(feats).classify(clf).rename("pred")
        mapas[arm] = cls
        # brecha de motor: reclasificar la reserva y comparar con scikit-learn
        got = cls.sampleRegions(collection=res, properties=["clase", "pid"], scale=10,
                                tileScale=4).getInfo()["features"]
        t = pd.DataFrame([f["properties"] for f in got]).set_index("pid")
        t = t.reindex(df.loc[hold, "point_id"].to_numpy())
        if t["pred"].isna().any():
            raise SystemExit("ABORTA: %d puntos de la reserva sin prediccion en GEE" % t["pred"].isna().sum())
        p_gee = np.array([CLASES[int(v)] for v in t["pred"]])
        val[arm] = {"macro_f1_gee": round(float(f1_score(y_hold, p_gee, average="macro",
                                                         zero_division=0)), 3),
                    "macro_f1_sklearn": round(float(f1_score(y_hold, sk[arm], average="macro",
                                                             zero_division=0)), 3),
                    "acuerdo_gee_sklearn": round(float((p_gee == sk[arm]).mean()), 3),
                    "n": int(hold.sum())}
        print("  %-24s GEE %.3f | sklearn %.3f | acuerdo %.3f"
              % (arm, val[arm]["macro_f1_gee"], val[arm]["macro_f1_sklearn"],
                 val[arm]["acuerdo_gee_sklearn"]), flush=True)

    (SALIDA / "validacion_gee_vs_sklearn.json").write_text(
        json.dumps(val, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n== miniaturas ==", flush=True)
    rutas = {}
    for arm in ARMS:
        rutas[arm] = bajar(mapas[arm].visualize(min=0, max=3, palette=PAL),
                           aoi, SALIDA / ("mapa_%s.png" % arm))
    desac = mapas["alphaearth"].neq(mapas["s2_multitemporal"]).selfMask()
    rutas["desacuerdo"] = bajar(desac.visualize(palette=["d9571e"]), aoi,
                                SALIDA / "mapa_desacuerdo.png")
    rutas["fondo"] = bajar(mapas["s2_multitemporal"].multiply(0).visualize(palette=["e9e6e0"]),
                           aoi, SALIDA / "mapa_fondo.png")

    # fraccion del area donde las dos representaciones discrepan
    fr = (mapas["alphaearth"].neq(mapas["s2_multitemporal"]).rename("d")
          .reduceRegion(ee.Reducer.mean(), aoi, 30, maxPixels=1e10, bestEffort=True).getInfo())
    val["fraccion_desacuerdo_ae_vs_s2"] = round(float(fr["d"]), 4)
    (SALIDA / "validacion_gee_vs_sklearn.json").write_text(
        json.dumps(val, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\ndesacuerdo AlphaEarth vs S2 multitemporal: %.1f %% del area"
          % (100 * val["fraccion_desacuerdo_ae_vs_s2"]), flush=True)

    figura(rutas, val, aoi_gj)
    print("\nFIN", flush=True)


def figura(rutas, val, aoi_gj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image

    xs = [c[0] for c in aoi_gj["features"][0]["geometry"]["coordinates"][0]] \
        if "features" in aoi_gj else []
    lat0 = 10.8
    asp = 1.0 / np.cos(np.radians(lat0))

    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.7), dpi=150)
    for ax, arm in zip(axes[:3], ARMS):
        ax.imshow(Image.open(rutas["fondo"]))
        ax.imshow(Image.open(rutas[arm]))
        ax.set_title(ETQ[arm] + "\nGEE macro-F1 %.3f on the reserve (sklearn %.3f)"
                     % (val[arm]["macro_f1_gee"], val[arm]["macro_f1_sklearn"]), fontsize=9)
    ax = axes[3]
    ax.imshow(Image.open(rutas["fondo"]))
    ax.imshow(Image.open(rutas["desacuerdo"]))
    ax.set_title("(d) Where (a) and (b) disagree\n%.1f %% of the study area"
                 % (100 * val["fraccion_desacuerdo_ae_vs_s2"]), fontsize=9)
    for ax in axes:
        ax.set_aspect(asp)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#9aa7ad")

    leg = [Patch(facecolor="#" + c, label=e) for c, e in zip(PAL, ETQ_CLASE)]
    leg += [Patch(facecolor="#d9571e", label="Disagreement")]
    fig.legend(handles=leg, loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("The same three input configurations applied to the whole study area, "
                 "CGSM 2021", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = AQUI / "figuras" / "fig_mapa_clasificacion.png"
    fig.savefig(out)
    print("\n" + str(out), flush=True)


if __name__ == "__main__":
    main()
