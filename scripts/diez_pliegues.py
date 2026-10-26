# -*- coding: utf-8 -*-
"""Informe 1: repetir el regimen principal con DIEZ pliegues de 6 km en vez de cinco.

PARA QUE. Con cinco pliegues la prueba de signos no puede bajar de 0,0625, de modo que
la evidencia estadistica del informe descansa en el McNemar agrupado, cuyo supuesto de
independencia entre puntos de un mismo pliegue solo se cumple de forma aproximada. Con
diez pliegues la prueba por pliegue puede alcanzar significancia por si sola (minimo
0,00195) y, ademas, cada punto se predice con un modelo distinto, lo que reduce la
dependencia entre las predicciones que entran al McNemar.

Mismos puntos, mismo buffer de 3 km, mismos bloques de 6 km, mismo Random Forest: se modifica la asignacion de los bloques a diez pliegues, conservando su tamano y la zona de exclusion (assign_blocked_folds de
build_split.py con k=10, semilla 42).

Salida: diez_pliegues/ (cv_metrics.csv, resumen.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import representaciones_2021 as rep  # noqa: E402
rb = rep.rb
sys.path.insert(0, str(AQUI.parent.parent / "AVANCE TESIS" / "PILOTO_BENCHMARK"))
import build_split as bs  # noqa: E402

SAL = AQUI / "diez_pliegues"
SAL.mkdir(exist_ok=True)
BASE = "s2_multitemporal"
OTROS = ["alphaearth", "s2_multi_mas_alphaearth"]
rep.ARMS = {k: rep.ARMS[k] for k in [BASE] + OTROS}


def main():
    df = rep.cargar()
    pool = ~df["holdout"].to_numpy()
    xy = df.loc[pool, ["x", "y"]].to_numpy()
    n_bloques = len(np.unique(np.floor(xy / 6000.0).astype(np.int64), axis=0))
    print("bloques de 6 km con puntos en el conjunto de validacion:", n_bloques, flush=True)
    if n_bloques < 15:
        sys.exit("ABORTA: %d bloques no alcanzan para diez pliegues" % n_bloques)

    df["fold_block_6km"] = bs.assign_blocked_folds(df, pool, L=6000.0, k=10, seed=42)
    t = df.loc[pool].groupby("fold_block_6km").size()
    print("puntos por pliegue:", t.to_dict(), flush=True)

    rb.REGIMES = {"block_6km": 3000.0}
    rb.BLOCK_L = {"block_6km": 6000.0}
    classes = sorted(df["clase_nombre"].unique())
    cv, per_fold, preds = rb.run_cv(df, rep.ARMS, classes, SAL)
    cv.to_csv(SAL / "cv_metrics.csv", index=False)
    per_fold.to_csv(SAL / "cv_per_fold.csv", index=False)
    from permutacion_bloques import con_xy     # prediccion por punto, para la prueba por bloques
    con_xy(df, preds).to_csv(SAL / "cv_predictions.csv", index=False)

    tab =per_fold.pivot(index="fold", columns="arm", values="macro_f1")
    y = preds[preds.arm == BASE].y_true.to_numpy()
    okb = preds[preds.arm == BASE].y_pred.to_numpy() == y
    res = {"n_bloques": int(n_bloques), "n_pliegues": int(tab.shape[0]),
           "macro_f1": cv.set_index("arm").macro_f1.round(3).to_dict(), "pareadas": {}}
    for arm in OTROS:
        d = (tab[arm] - tab[BASE]).to_numpy()
        gana = int((d > 0).sum())
        oka = preds[preds.arm == arm].y_pred.to_numpy() == y
        b, c = int((okb & ~oka).sum()), int((~okb & oka).sum())
        res["pareadas"][arm] = {
            "d_media": round(float(d.mean()), 3), "sd_pareada": round(float(d.std(ddof=1)), 3),
            "gana": "%d/%d" % (gana, len(d)),
            "p_signos": round(float(binomtest(gana, len(d), 0.5).pvalue), 5),
            "mcnemar": {"n": int(len(y)), "solo_base": b, "solo_otro": c,
                        "p": round(float(binomtest(b, b + c, 0.5).pvalue), 5) if b + c else 1.0},
            "d_por_pliegue": [round(float(v), 3) for v in d]}
        print("  %-24s d = %+.3f +/- %.3f | gana %s | signos p=%.4f | McNemar %d/%d p=%.4f"
              % (arm, d.mean(), d.std(ddof=1), res["pareadas"][arm]["gana"],
                 res["pareadas"][arm]["p_signos"], b, c, res["pareadas"][arm]["mcnemar"]["p"]), flush=True)
    (SAL / "resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nmacro-F1:", res["macro_f1"], "\nFIN", flush=True)


if __name__ == "__main__":
    main()
