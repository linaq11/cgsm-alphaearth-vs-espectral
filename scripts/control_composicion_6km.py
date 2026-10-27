# -*- coding: utf-8 -*-
r"""Control B: tamaño y composición de clases del entrenamiento a 6 km (preregistrado en PREREGISTRO_control_tamano_6km.md, sección B).

Igual que control_tamano_6km.py, pero el entrenamiento de cada pliegue de 6 km se submuestrea a las cuentas exactas por
clase con que entrena cada pliegue de 16 km (tres composiciones, una por pliegue de 16 km). Si una clase pide más puntos
de los disponibles en el pliegue de 6 km, se toman todos y se registra el tamaño real. 20 repeticiones por composición.
Separa el efecto de la distancia del efecto conjunto de tamaño y composición en la lectura del régimen de 16 km.

SALIDAS (representaciones_2021/)
  control_composicion_6km.csv          macro-F1 por repetición, composición y brazo, con el tamaño real
  control_composicion_6km_resumen.json diferencias pareadas frente a la línea base por composición y en conjunto

USO: python -X utf8 control_composicion_6km.py
"""
import importlib.util
import json
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
REPETICIONES = 20
BRAZOS = {k: rep.ARMS[k] for k in ("s2_multitemporal", "alphaearth", "s2_multi_mas_alphaearth")}


def main():
    df = rep.cargar()
    pool = ~df["holdout"].to_numpy(); xy = df[["x", "y"]].to_numpy(); y_all = df["clase_nombre"].to_numpy()
    f6 = df["fold_block_6km"].to_numpy(); f16 = df["fold_block_16km"].to_numpy()
    pl6 = sorted(set(f6[f6 >= 0])); pl16 = sorted(set(f16[f16 >= 0]))
    entren6 = {f: np.nonzero(rb.buffered_train(df, f6, f, rb.REGIMES["block_6km"], pool))[0] for f in pl6}
    comps = {}
    for f in pl16:
        tr = rb.buffered_train(df, f16, f, rb.REGIMES["block_16km"], pool)
        comps[int(f)] = {c: int(n) for c, n in pd.Series(y_all[tr]).value_counts().sort_index().items()}
    print("composiciones de 16 km:", comps, flush=True)
    X = {a: df[feats].to_numpy() for a, feats in BRAZOS.items()}
    filas = []; t0 = time.time()
    for comp_id, cuotas in comps.items():
        for r in range(REPETICIONES):
            rng = np.random.default_rng(1000 * comp_id + r)
            sub = {}
            for f in pl6:
                idx = entren6[f]; sel = []
                for c, q in cuotas.items():
                    cand = idx[y_all[idx] == c]
                    sel.append(rng.choice(cand, size=min(q, len(cand)), replace=False))
                sub[f] = np.sort(np.concatenate(sel))
            n_real = int(np.mean([len(v) for v in sub.values()]))
            for a in BRAZOS:
                yt, yp = [], []
                for f in pl6:
                    tr = sub[f]; te = pool & (f6 == f)
                    model, _ = rb.tune_and_fit(X[a][tr], y_all[tr], xy[tr], rb.BLOCK_L["block_6km"])
                    yt.append(y_all[te]); yp.append(model.predict(X[a][te]))
                yt = np.concatenate(yt); yp = np.concatenate(yp)
                filas.append(dict(composicion=comp_id, n_pedido=sum(cuotas.values()), n_real_medio=n_real, repeticion=r, brazo=a,
                                  macro_f1=f1_score(yt, yp, average="macro", zero_division=0)))
            print(f"comp={comp_id} rep={r:2d} n~{n_real} " + " ".join(f"{x['brazo']}={x['macro_f1']:.3f}" for x in filas[-3:]) + f"  ({time.time() - t0:.0f} s)", flush=True)
    res = pd.DataFrame(filas); res.to_csv(SALIDA / "control_composicion_6km.csv", index=False)

    w = res.pivot(index=["composicion", "repeticion"], columns="brazo", values="macro_f1")
    resumen = {"composiciones_16km": comps, "n_real_medio": {int(k): int(v) for k, v in res.groupby("composicion").n_real_medio.first().items()}}
    def bloque(g):
        d_ae = g["alphaearth"] - g["s2_multitemporal"]; d_cb = g["s2_multi_mas_alphaearth"] - g["s2_multitemporal"]
        return {"macro_f1_media": {a: round(float(g[a].mean()), 3) for a in BRAZOS},
                "dif_alphaearth": {"media": round(float(d_ae.mean()), 3), "ic_2.5": round(float(d_ae.quantile(0.025)), 3), "ic_97.5": round(float(d_ae.quantile(0.975)), 3), "rep_positivas": int((d_ae > 0).sum()), "n": int(len(d_ae))},
                "dif_combinacion": {"media": round(float(d_cb.mean()), 3), "ic_2.5": round(float(d_cb.quantile(0.025)), 3), "ic_97.5": round(float(d_cb.quantile(0.975)), 3), "rep_positivas": int((d_cb > 0).sum()), "n": int(len(d_cb))}}
    for comp_id in comps:
        resumen[f"comp_{comp_id}"] = bloque(w.loc[comp_id])
    resumen["conjunto"] = bloque(w)
    (SALIDA / "control_composicion_6km_resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
