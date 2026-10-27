# -*- coding: utf-8 -*-
r"""Control C: régimen de 16 km con exclusión de 3 km y varias mallas (preregistrado en PREREGISTRO_control_tamano_6km.md, sección C).

El régimen de 16 km del Informe 1 usó una sola malla de bloques, tres pliegues y una zona de exclusión de 8 km, tres veces
el alcance máximo de los variogramas de los residuos (2.627 m). Con esa exclusión cada pliegue entrena con 95 a 142 puntos y
pierde casi por completo alguna clase. Aquí se repite el régimen de 16 km con la malla original y con 9 desplazamientos
aleatorios de la malla, con exclusión de 3 km (la misma del régimen de 6 km, justificada por el variograma) y, como
comparación, con la exclusión original de 8 km. Mismo motor de asignación de bloques a pliegues (build_split), mismo
Random Forest y mismo ajuste de hiperparámetros.

SALIDAS (representaciones_2021/)
  regimen_16km_corregido.csv          macro-F1 fuera de pliegue por malla, exclusión y brazo; tamaño y clase mínima del entrenamiento
  regimen_16km_corregido_resumen.json medias e intervalos 2,5-97,5 entre mallas de las diferencias frente a la línea base

USO: python -X utf8 regimen_16km_corregido.py
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
sys.path.insert(0, str(rep.BENCH)); import build_split as bs  # noqa: E402
SALIDA = AQUI / "representaciones_2021"

L = 16000.0; K = 3; N_MALLAS = 10; EXCLUSIONES = {"3km": 3000.0, "8km": 8000.0}
BRAZOS = rep.ARMS


def main():
    df = rep.cargar()
    pool = ~df["holdout"].to_numpy(); xy = df[["x", "y"]].to_numpy(); y_all = df["clase_nombre"].to_numpy()
    rng = np.random.default_rng(16)
    desplaz = [(0.0, 0.0)] + [tuple(rng.uniform(0, L, 2)) for _ in range(N_MALLAS - 1)]
    # comprobación: la malla 0 reproduce los pliegues de 16 km del informe
    f_orig = df["fold_block_16km"].to_numpy()
    d0 = df.copy(); f0 = bs.assign_blocked_folds(d0, pool, L, k=K)
    iguales = (f0[pool] == f_orig[pool]).mean()
    print(f"malla 0 frente a fold_block_16km del informe: {iguales:.3f} de coincidencia", flush=True)
    X = {a: df[feats].to_numpy() for a, feats in BRAZOS.items()}
    filas = []; t0 = time.time()
    for m, (dx, dy) in enumerate(desplaz):
        d = df.copy(); d["x"] = d["x"] - dx; d["y"] = d["y"] - dy      # desplazar la malla equivale a desplazar los puntos
        fold = bs.assign_blocked_folds(d, pool, L, k=K)
        for nombre, buf in EXCLUSIONES.items():
            entren = {f: rb.buffered_train(df, fold, f, buf, pool) for f in range(K)}
            n_tr = [int(entren[f].sum()) for f in range(K)]
            min_clase = [int(pd.Series(y_all[entren[f]]).value_counts().reindex(np.unique(y_all)).fillna(0).min()) for f in range(K)]
            for a in BRAZOS:
                yt, yp = [], []
                for f in range(K):
                    tr = entren[f]; te = pool & (fold == f)
                    if tr.sum() < 20 or te.sum() == 0: continue
                    model, _ = rb.tune_and_fit(X[a][tr], y_all[tr], xy[tr], L)
                    yt.append(y_all[te]); yp.append(model.predict(X[a][te]))
                yt = np.concatenate(yt); yp = np.concatenate(yp)
                filas.append(dict(malla=m, dx=round(dx), dy=round(dy), exclusion=nombre, brazo=a, n_pred=len(yt),
                                  n_entren=";".join(map(str, n_tr)), n_entren_medio=int(np.mean(n_tr)), min_clase_entren=min(min_clase),
                                  macro_f1=f1_score(yt, yp, average="macro", zero_division=0)))
            print(f"malla {m} excl {nombre} n_entren {n_tr} min_clase {min_clase} " + " ".join(f"{x['brazo']}={x['macro_f1']:.3f}" for x in filas[-len(BRAZOS):]) + f"  ({time.time() - t0:.0f} s)", flush=True)
    res = pd.DataFrame(filas); res.to_csv(SALIDA / "regimen_16km_corregido.csv", index=False)
    resumen = {"coincidencia_malla0_con_informe": round(float(iguales), 3)}
    for nombre in EXCLUSIONES:
        w = res[res.exclusion == nombre].pivot(index="malla", columns="brazo", values="macro_f1")
        d_ae = w["alphaearth"] - w["s2_multitemporal"]; d_cb = w["s2_multi_mas_alphaearth"] - w["s2_multitemporal"]
        resumen[nombre] = {"macro_f1_media": {a: round(float(w[a].mean()), 3) for a in BRAZOS},
                           "macro_f1_malla0": {a: round(float(w.loc[0, a]), 3) for a in BRAZOS},
                           "n_entren_medio": round(float(res[res.exclusion == nombre].n_entren_medio.mean()), 1),
                           "min_clase_entren_medio": round(float(res[res.exclusion == nombre].min_clase_entren.mean()), 1),
                           "dif_alphaearth": {"media": round(float(d_ae.mean()), 3), "ic_2.5": round(float(d_ae.quantile(0.025)), 3), "ic_97.5": round(float(d_ae.quantile(0.975)), 3), "mallas_positivas": int((d_ae > 0).sum()), "malla0": round(float(d_ae.loc[0]), 3)},
                           "dif_combinacion": {"media": round(float(d_cb.mean()), 3), "ic_2.5": round(float(d_cb.quantile(0.025)), 3), "ic_97.5": round(float(d_cb.quantile(0.975)), 3), "mallas_positivas": int((d_cb > 0).sum()), "malla0": round(float(d_cb.loc[0]), 3)}}
    (SALIDA / "regimen_16km_corregido_resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
