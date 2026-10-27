# -*- coding: utf-8 -*-
r"""Control del tamaño de entrenamiento a 6 km (preregistrado en PREREGISTRO_control_tamano_6km.md).

Repite el régimen de bloques de 6 km del Informe 1 (mismos pliegues de prueba, misma exclusión de 3 km,
mismo Random Forest y mismo ajuste de hiperparámetros del motor del piloto) submuestreando al azar el
entrenamiento de cada pliegue a n = 95 y n = 142 puntos, los tamaños mínimo y máximo que quedan a 16 km,
estratificado por clase en proporción. 20 repeticiones por tamaño. Sirve para separar el efecto de la
distancia del efecto del tamaño del entrenamiento en la lectura del régimen de 16 km.

SALIDAS (representaciones_2021/)
  control_tamano_6km.csv          macro-F1 fuera de pliegue por repetición, tamaño y brazo
  control_tamano_6km_resumen.json diferencias pareadas frente a la línea base: media e intervalo 2,5-97,5

USO: python -X utf8 control_tamano_6km.py
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

AQUI = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("representaciones_2021", AQUI / "representaciones_2021.py")
rep = importlib.util.module_from_spec(spec); spec.loader.exec_module(rep)
rb = rep.rb
SALIDA = AQUI / "representaciones_2021"

TAMANOS = (95, 142)
REPETICIONES = 20
BRAZOS = {k: rep.ARMS[k] for k in ("s2_multitemporal", "alphaearth", "s2_multi_mas_alphaearth")}
REGIMEN = "block_6km"


def submuestra(idx, y, n, rng):
    """Cuotas por clase proporcionales (mayor resto), al menos 1 por clase; elección al azar dentro de cada clase."""
    clases, cuenta = np.unique(y[idx], return_counts=True)
    exact = cuenta / cuenta.sum() * n
    cuota = np.floor(exact).astype(int); cuota = np.maximum(cuota, 1)
    resto = n - cuota.sum()
    orden = np.argsort(-(exact - np.floor(exact)))
    for k in orden[:max(resto, 0)]:
        cuota[k] += 1
    while cuota.sum() > n:
        cuota[np.argmax(cuota)] -= 1
    sel = []
    for c, q in zip(clases, cuota):
        cand = idx[y[idx] == c]
        sel.append(rng.choice(cand, size=min(q, len(cand)), replace=False))
    return np.sort(np.concatenate(sel))


def main():
    df = rep.cargar()
    pool = ~df["holdout"].to_numpy(); xy = df[["x", "y"]].to_numpy(); y_all = df["clase_nombre"].to_numpy()
    fold = df[f"fold_{REGIMEN}"].to_numpy(); buf = rb.REGIMES[REGIMEN]; L = rb.BLOCK_L[REGIMEN]
    pliegues = sorted(set(fold[fold >= 0]))
    entren = {f: np.nonzero(rb.buffered_train(df, fold, f, buf, pool))[0] for f in pliegues}
    print("entrenamiento completo por pliegue:", {f: len(v) for f, v in entren.items()}, flush=True)
    X = {a: df[feats].to_numpy() for a, feats in BRAZOS.items()}
    filas = []; t0 = time.time()
    for n in TAMANOS:
        for r in range(REPETICIONES):
            rng = np.random.default_rng(r)
            sub = {f: submuestra(entren[f], y_all, n, rng) for f in pliegues}   # misma submuestra para los tres brazos
            for a in BRAZOS:
                yt, yp = [], []
                for f in pliegues:
                    tr = sub[f]; te = pool & (fold == f)
                    model, _ = rb.tune_and_fit(X[a][tr], y_all[tr], xy[tr], L)
                    yt.append(y_all[te]); yp.append(model.predict(X[a][te]))
                yt = np.concatenate(yt); yp = np.concatenate(yp)
                filas.append(dict(n_entren=n, repeticion=r, brazo=a, n_pred=len(yt),
                                  macro_f1=f1_score(yt, yp, average="macro", zero_division=0)))
            print(f"n={n} rep={r:2d} " + " ".join(f"{x['brazo']}={x['macro_f1']:.3f}" for x in filas[-3:]) + f"  ({time.time() - t0:.0f} s)", flush=True)
    res = pd.DataFrame(filas); res.to_csv(SALIDA / "control_tamano_6km.csv", index=False)

    w = res.pivot(index=["n_entren", "repeticion"], columns="brazo", values="macro_f1")
    ref = pd.read_csv(SALIDA / "cv_metrics.csv").pivot(index="arm", columns="regime", values="macro_f1")
    resumen = {"referencia": {reg: {a: round(float(ref.loc[a, reg]), 3) for a in BRAZOS} for reg in ("block_6km", "block_16km")},
               "referencia_dif": {reg: {"alphaearth": round(float(ref.loc["alphaearth", reg] - ref.loc["s2_multitemporal", reg]), 3),
                                        "combinacion": round(float(ref.loc["s2_multi_mas_alphaearth", reg] - ref.loc["s2_multitemporal", reg]), 3)}
                                  for reg in ("block_6km", "block_16km")}}
    for n in TAMANOS:
        g = w.loc[n]
        d_ae = g["alphaearth"] - g["s2_multitemporal"]; d_cb = g["s2_multi_mas_alphaearth"] - g["s2_multitemporal"]
        resumen[f"n_{n}"] = {
            "macro_f1_media": {a: round(float(g[a].mean()), 3) for a in BRAZOS},
            "macro_f1_sd": {a: round(float(g[a].std()), 3) for a in BRAZOS},
            "dif_alphaearth": {"media": round(float(d_ae.mean()), 3), "ic_2.5": round(float(d_ae.quantile(0.025)), 3), "ic_97.5": round(float(d_ae.quantile(0.975)), 3), "rep_positivas": int((d_ae > 0).sum())},
            "dif_combinacion": {"media": round(float(d_cb.mean()), 3), "ic_2.5": round(float(d_cb.quantile(0.025)), 3), "ic_97.5": round(float(d_cb.quantile(0.975)), 3), "rep_positivas": int((d_cb > 0).sum())},
        }
    (SALIDA / "control_tamano_6km_resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
