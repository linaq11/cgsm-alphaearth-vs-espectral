# -*- coding: utf-8 -*-
r"""Linea base espectral multitemporal 2021 para el informe 1 de Percepcion Remota
Avanzada.

PARA QUE. La propuesta del curso compara variables espectrales con representaciones
de AlphaEarth (anuales, 2021). El piloto de agosto uso una sola fecha (marzo de
2021), de modo que no podia separar "mejor representacion" de "mas tiempo
observado". Este guion construye la linea base espectral con estacion seca y
humeda de 2021 desde el cubo, con los MISMOS 600 puntos, pliegues y motor de
Random Forest del piloto, para que la comparacion del informe tenga equivalencia temporal.

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
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
AVANCE = r"C:\LINA\UNAL 2626-1\AVANCE TESIS"
PILOTO = os.path.join(AVANCE, "PILOTO_BENCHMARK")
sys.path.insert(0, os.path.join(AVANCE, "PUBLICACION_GEE"))
sys.path.insert(0, PILOTO)
import _proj  # noqa: E402,F401  (aisla PROJ antes de que run_benchmark importe rasterio)
import run_benchmark as rb  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_fscore_support  # noqa: E402

PROYECTO = "basic-buttress-338101"
RAIZ = "projects/basic-buttress-338101/assets/cgsm_datacube_v1/s2/"
SPLIT = os.path.join(PILOTO, "split_assignments.csv")
BANDAS = ["B2", "B3", "B4", "B8", "B11", "B12"]
MESES = ["2021%02d" % m for m in range(1, 13)]
SECA = ["202101", "202102", "202103"]
HUMEDA = ["202109", "202110", "202111"]
INDICES = ["NDVI", "NDWI", "MNDWI"]


def indices(df, pref):
    b3, b4, b8, b11 = (df[pref + b] for b in ("B3", "B4", "B8", "B11"))
    df[pref + "NDVI"] = (b8 - b4) / (b8 + b4)
    df[pref + "NDWI"] = (b3 - b8) / (b3 + b8)
    df[pref + "MNDWI"] = (b3 - b11) / (b3 + b11)


def muestrear_gee(sp):
    import ee
    ee.Initialize(project=PROYECTO)
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a), float(b)]), {"pid": int(p)})
                               for a, b, p in zip(sp["lon"], sp["lat"], sp["point_id"])])
    pila = ee.Image.cat([ee.Image(RAIZ + m).select(BANDAS).rename([m + "_" + b for b in BANDAS])
                         for m in MESES])
    filas = pila.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=10).getInfo()["features"]
    out = pd.DataFrame([dict(point_id=f["properties"]["pid"],
                             **{k: v for k, v in f["properties"].items() if k != "pid"}) for f in filas])
    return out.sort_values("point_id").reset_index(drop=True)


def construir_variables(raw):
    """De 72 columnas mensuales (x10000) a las variables de las cinco configuraciones."""
    v = pd.DataFrame({"point_id": raw["point_id"]})
    mensual = {}
    for m in MESES:
        t = pd.DataFrame({b: raw[m + "_" + b] / 10000.0 for b in BANDAS})
        indices(t, "")
        mensual[m] = t
    # marzo solo, replica del piloto
    for c in BANDAS + INDICES:
        v["mar_" + c] = mensual["202103"][c].to_numpy()
    # medianas estacionales por banda, indices recalculados desde la mediana
    for pref, meses in (("seca_", SECA), ("hum_", HUMEDA)):
        for b in BANDAS:
            v[pref + b] = np.nanmedian(np.vstack([mensual[m][b].to_numpy() for m in meses]), axis=0)
        indices(v, pref)
    # fenologia sobre los 12 meses
    for c in INDICES:
        serie = np.vstack([mensual[m][c].to_numpy() for m in MESES])
        v["fen_" + c + "_min"] = np.nanmin(serie, axis=0)
        v["fen_" + c + "_max"] = np.nanmax(serie, axis=0)
        v["fen_" + c + "_amp"] = v["fen_" + c + "_max"] - v["fen_" + c + "_min"]
        v["fen_" + c + "_sd"] = np.nanstd(serie, axis=0)
    return v


F_MAR = ["mar_" + c for c in BANDAS + INDICES]
F_SECA = ["seca_" + c for c in BANDAS + INDICES]
F_HUM = ["hum_" + c for c in BANDAS + INDICES]
F_FEN = ["fen_%s_%s" % (c, s) for c in INDICES for s in ("min", "max", "amp", "sd")]
CONFIGS = {
    "s2_marzo": F_MAR,
    "s2_seca": F_SECA,
    "s2_humeda": F_HUM,
    "s2_seca_humeda": F_SECA + F_HUM,
    "s2_multitemporal": F_SECA + F_HUM + F_FEN,
}


def reserva(df, arms, classes, buffer_m=3000.0):
    hold = df["holdout"].to_numpy()
    xy = df[["x", "y"]].to_numpy()
    y = df["clase_nombre"].to_numpy()
    d = np.sqrt(((xy[~hold][:, None, :] - xy[hold][None, :, :]) ** 2).sum(-1)).min(1)
    tr = np.zeros(len(df), bool)
    tr[np.nonzero(~hold)[0][d > buffer_m]] = True
    filas, pred, imp = [], {}, {}
    for arm, feats in arms.items():
        X = df[feats].to_numpy()
        model, best = rb.tune_and_fit(X[tr], y[tr], xy[tr], rb.BLOCK_L["block_6km"])
        p = model.predict(X[hold])
        pred[arm] = p
        imp[arm] = dict(zip(feats, model.feature_importances_.round(4)))
        _, _, f1, _ = precision_recall_fscore_support(y[hold], p, labels=classes, zero_division=0)
        filas.append(dict(arm=arm, n=int(hold.sum()), n_train=int(tr.sum()),
                          oa=float((y[hold] == p).mean()),
                          macro_f1=f1_score(y[hold], p, average="macro", zero_division=0),
                          **{"f1_" + c: v for c, v in zip(classes, f1)}))
    return pd.DataFrame(filas), pred, imp, y[hold]


def mcnemar(pa, pb, y):
    ca, cb = pa == y, pb == y
    n_b = int((~ca & cb).sum())
    n_a = int((ca & ~cb).sum())
    p = binomtest(n_a, n_a + n_b, 0.5).pvalue if (n_a + n_b) else None
    return dict(solo_a=n_a, solo_b=n_b, p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(AQUI, "linea_base_2021"))
    ap.add_argument("--solo-muestreo", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sp = pd.read_csv(SPLIT)
    print("puntos del piloto:", len(sp), flush=True)

    crudo = os.path.join(a.out, "mensual_2021_crudo.csv")
    if os.path.exists(crudo):
        raw = pd.read_csv(crudo)
        print("muestreo mensual leido de", crudo, flush=True)
    else:
        print("muestreando 12 meses x 6 bandas en Earth Engine ...", flush=True)
        raw = muestrear_gee(sp)
        raw.to_csv(crudo, index=False)
    for m in MESES:
        print("   %s: %d puntos con dato" % (m, int(raw[m + "_B4"].notna().sum())), flush=True)

    var = construir_variables(raw)
    df = sp.merge(var, on="point_id", validate="1:1")
    todas = sorted(set(sum(CONFIGS.values(), [])))
    ok = df[todas].notna().all(axis=1)
    print("puntos con todas las variables: %d de %d" % (ok.sum(), len(df)), flush=True)
    df = df[ok].reset_index(drop=True)
    df.to_csv(os.path.join(a.out, "variables_puntos.csv"), index=False)
    if a.solo_muestreo:
        return

    classes = sorted(df["clase_nombre"].unique())
    cv, per_fold, preds = rb.run_cv(df, CONFIGS, classes, Path(a.out))
    cv.to_csv(os.path.join(a.out, "cv_metrics.csv"), index=False)
    per_fold.to_csv(os.path.join(a.out, "cv_per_fold.csv"), index=False)
    hold, pred, imp, y_hold = reserva(df, CONFIGS, classes)
    hold.to_csv(os.path.join(a.out, "holdout_metrics.csv"), index=False)
    mc = {"s2_marzo vs s2_seca_humeda": mcnemar(pred["s2_marzo"], pred["s2_seca_humeda"], y_hold),
          "s2_marzo vs s2_multitemporal": mcnemar(pred["s2_marzo"], pred["s2_multitemporal"], y_hold),
          "s2_seca_humeda vs s2_multitemporal": mcnemar(pred["s2_seca_humeda"], pred["s2_multitemporal"], y_hold)}
    json.dump({"mcnemar": mc, "importancias": imp, "n": int(len(df)), "clases": classes},
              open(os.path.join(a.out, "resumen.json"), "w"), indent=2, default=float)

    sd = per_fold.groupby(["regime", "arm"]).macro_f1.std().round(3)
    tabla = cv.pivot(index="arm", columns="regime", values="macro_f1").reindex(list(CONFIGS)).round(3)
    tabla["holdout"] = hold.set_index("arm").macro_f1.round(3)
    print("\n== macro-F1 por regimen y configuracion (n = %d, etiquetas WorldCover 2021) ==" % len(df))
    print(tabla.to_string())
    print("\nsd entre pliegues, bloques 6 km:", sd.loc["block_6km"].to_dict())
    print("\nF1 por clase, bloques 6 km:")
    print(cv[cv.regime == "block_6km"].set_index("arm")[["f1_" + c for c in classes]].reindex(list(CONFIGS)).round(3).to_string())
    print("\nMcNemar en la reserva:", json.dumps(mc))
    print("escrito en", a.out)


if __name__ == "__main__":
    main()
