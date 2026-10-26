# -*- coding: utf-8 -*-
"""Prueba de sensibilidad del informe 1 a la calidad de las etiquetas.

Repite la comparacion central (linea base multitemporal, AlphaEarth, union) sin los
puntos cuya etiqueta WorldCover contradice a Global Mangrove Watch 2020 o, para agua,
a Dynamic World 2021 (auditoria_etiquetas.py, columna 'conflicto'). Mismos pliegues,
misma reserva, mismo motor. No reemplaza los resultados del informe: comprueba si
las dos conclusiones centrales se sostienen con etiquetas mas limpias.
Salida: sensibilidad_etiquetas/
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
SAL = AQUI / "sensibilidad_etiquetas"
SAL.mkdir(exist_ok=True)
BASE = "s2_multitemporal"
rep.ARMS = {k: rep.ARMS[k] for k in (BASE, "alphaearth", "s2_multi_mas_alphaearth")}


def pareadas(per_fold, preds):
    out = {}
    for reg in ("block_6km", "block_16km"):
        t = per_fold[per_fold.regime == reg].pivot(index="fold", columns="arm", values="macro_f1")
        sub = preds[preds.regime == reg]
        y = sub[sub.arm == BASE].y_true.to_numpy()
        okb = sub[sub.arm == BASE].y_pred.to_numpy() == y
        for arm in ("alphaearth", "s2_multi_mas_alphaearth"):
            d = (t[arm] - t[BASE]).to_numpy()
            oka = sub[sub.arm == arm].y_pred.to_numpy() == y
            b, c = int((okb & ~oka).sum()), int((~okb & oka).sum())
            out[f"{reg}|{arm}"] = dict(d_media=round(float(d.mean()), 3), sd=round(float(d.std(ddof=1)), 3),
                                      gana=f"{int((d > 0).sum())}/{len(d)}", n=int(len(y)),
                                      solo_base=b, solo_otro=c,
                                      p=round(float(binomtest(b, b + c, .5).pvalue), 5) if b + c else 1.0)
    return out


def main():
    df = rep.cargar()
    aud = pd.read_csv(AQUI / "auditoria_etiquetas" / "puntos_auditados.csv")[["point_id", "conflicto"]]
    df = df.merge(aud, on="point_id", validate="1:1")
    limpio = df[~df.conflicto].reset_index(drop=True)
    print("puntos: %d -> %d (quitados %d)" % (len(df), len(limpio), int(df.conflicto.sum())), flush=True)
    print("por clase:", limpio.clase_nombre.value_counts().to_dict(),
          "| reserva:", int(limpio.holdout.sum()), flush=True)
    classes = sorted(limpio.clase_nombre.unique())
    cv, per_fold, preds = rb.run_cv(limpio, rep.ARMS, classes, SAL)
    cv.to_csv(SAL / "cv_metrics.csv", index=False)
    from permutacion_bloques import con_xy     # prediccion por punto, para la prueba por bloques
    con_xy(limpio, preds).to_csv(SAL / "cv_predictions.csv", index=False)
    hold, mc, _, _ = rep.reserva(limpio, classes)
    tabla = cv.pivot(index="arm", columns="regime", values="macro_f1")
    tabla["reserva"] = hold.set_index("arm").macro_f1
    tabla = tabla.reindex(list(rep.ARMS))[["random", "block_6km", "block_16km", "reserva"]].round(3)
    f1c = cv[cv.regime == "block_6km"].set_index("arm")[[f"f1_{c}" for c in classes]].round(3)
    res = {"n": len(limpio), "quitados": int(df.conflicto.sum()),
           "macro_f1": json.loads(tabla.to_json(orient="index")),
           "f1_clase_6km": json.loads(f1c.to_json(orient="index")),
           "pareadas": pareadas(per_fold, preds), "mcnemar_reserva": mc}
    (SAL / "resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nmacro-F1 con etiquetas depuradas:\n" + tabla.to_string())
    print("\nF1 por clase, 6 km:\n" + f1c.to_string())
    print("\npareadas:\n" + json.dumps(res["pareadas"], indent=1))
    print("\nFIN")


if __name__ == "__main__":
    main()
