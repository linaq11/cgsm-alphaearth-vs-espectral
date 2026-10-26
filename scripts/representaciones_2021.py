# -*- coding: utf-8 -*-
r"""Comparacion de representaciones para el informe 1 de Percepcion Remota Avanzada.

PARA QUE. El piloto de agosto comparo variables espectrales de UNA fecha (marzo de
2021, 9 variables) contra las representaciones anuales de AlphaEarth (64), de modo
que no separaba "mejor representacion" de "mas tiempo observado". La linea base
multitemporal (linea_base_2021.py) arreglo el lado espectral pero no volvio a
correr AlphaEarth contra ella. Este script compara ambas fuentes en condiciones equivalentes: mismos
600 puntos, mismos pliegues, mismo Random Forest, y solo cambia la entrada.

PREGUNTAS QUE RESPONDE
  1. Ventaja: cuanto aporta AlphaEarth frente a la linea base espectral cuando la
     espectral ya observa todo el ano.
  2. Complementariedad: la union de las dos supera a AlphaEarth sola?
  3. Generalizacion: esa ventaja sobrevive al aumentar la separacion espacial
     (aleatorio -> bloques de 6 km -> bloques de 16 km -> reserva contigua)?
  4. Sensibilidad al clasificador: Ball et al. (2026) muestran que la ventaja de
     las representaciones exige un clasificador no lineal. Se repite el nucleo de
     la comparacion con un perceptron multicapa pequeno.

ENTRADAS
  informe1/linea_base_2021/variables_puntos.csv   30 variables espectrales + split
  PILOTO_BENCHMARK/alphaearth2021_etiquetas.csv   A00-A63 (Satellite Embedding 2021)
  Motor de validacion: PILOTO_BENCHMARK/run_benchmark.py (pliegues con buffer,
  ajuste de hiperparametros por CV interna agrupada, semilla 42).

SALIDAS (informe1/representaciones_2021/)
  cv_metrics.csv, cv_per_fold.csv, cv_predictions.csv, holdout_metrics.csv,
  mcnemar_holdout.json, importancias.json, mlp_sensibilidad.csv, resumen.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).resolve().parent
BENCH = AQUI.parent.parent / "AVANCE TESIS" / "PILOTO_BENCHMARK"
SALIDA = AQUI / "representaciones_2021"
SALIDA.mkdir(exist_ok=True)
sys.path.insert(0, str(BENCH))
import run_benchmark as rb          # noqa: E402  motor del piloto, sin modificar

BANDAS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "MNDWI"]
F_MAR = ["mar_" + b for b in BANDAS]
F_SECA = ["seca_" + b for b in BANDAS]
F_HUM = ["hum_" + b for b in BANDAS]
F_FEN = [f"fen_{i}_{e}" for i in ("NDVI", "NDWI", "MNDWI") for e in ("min", "max", "amp", "sd")]
F_S2 = F_SECA + F_HUM + F_FEN          # 30, la linea base del informe
F_AE = [f"A{i:02d}" for i in range(64)]

ARMS = {
    "s2_marzo": F_MAR,                       # 9   replica del piloto
    "s2_seca_humeda": F_SECA + F_HUM,        # 18
    "s2_multitemporal": F_S2,                # 30  linea base
    "alphaearth": F_AE,                      # 64
    "s2_multi_mas_alphaearth": F_S2 + F_AE,  # 94  complementariedad
}


def cargar():
    vp = pd.read_csv(AQUI / "linea_base_2021" / "variables_puntos.csv")
    ae = pd.read_csv(BENCH / "alphaearth2021_etiquetas.csv").reset_index(drop=True)
    if len(vp) != len(ae):
        sys.exit(f"ABORTA: {len(vp)} puntos espectrales frente a {len(ae)} de AlphaEarth")
    ae["point_id"] = np.arange(len(ae))
    # las dos tablas vienen del mismo split en el mismo orden; se verifica antes de unir
    d = np.hypot(vp.lon.to_numpy() - ae.lon.to_numpy(), vp.lat.to_numpy() - ae.lat.to_numpy())
    if d.max() > 1e-6:
        sys.exit(f"ABORTA: las coordenadas no coinciden (maximo {d.max():.2e} grados)")
    if (vp.clase_nombre.to_numpy() != ae.clase_nombre.to_numpy()).any():
        sys.exit("ABORTA: las clases no coinciden entre las dos tablas")
    df = vp.merge(ae[["point_id"] + F_AE], on="point_id", validate="1:1")
    faltan = [c for c in sum(ARMS.values(), []) if c not in df.columns]
    if faltan:
        sys.exit(f"ABORTA: faltan columnas {faltan[:6]}")
    assert df[sorted(set(sum(ARMS.values(), [])))].notna().all().all()
    print(f"tabla alineada: {len(df)} puntos, {len(F_S2)} espectrales + {len(F_AE)} de AlphaEarth", flush=True)
    return df


def reserva(df, classes, buffer_m=3000.0):
    """Entrena en la reserva de puntos con buffer y evalua una sola vez en la reserva contigua."""
    hold = df["holdout"].to_numpy()
    xy = df[["x", "y"]].to_numpy()
    y = df["clase_nombre"].to_numpy()
    d = np.sqrt(((xy[~hold][:, None, :] - xy[hold][None, :, :]) ** 2).sum(-1)).min(1)
    tr = np.zeros(len(df), bool)
    tr[np.nonzero(~hold)[0][d > buffer_m]] = True
    print(f"reserva: entrena con {tr.sum()} puntos, evalua en {hold.sum()}", flush=True)
    filas, pred, imp = [], {}, {}
    for arm, feats in ARMS.items():
        X = df[feats].to_numpy()
        model, best = rb.tune_and_fit(X[tr], y[tr], xy[tr], rb.BLOCK_L["block_6km"])
        p = model.predict(X[hold])
        pred[arm] = p
        imp[arm] = dict(zip(feats, model.feature_importances_.round(4)))
        _, _, f1, _ = precision_recall_fscore_support(y[hold], p, labels=classes, zero_division=0)
        filas.append(dict(arm=arm, n_vars=len(feats), n=int(hold.sum()), n_train=int(tr.sum()),
                          oa=float((y[hold] == p).mean()),
                          kappa=cohen_kappa_score(y[hold], p),
                          macro_f1=f1_score(y[hold], p, average="macro", zero_division=0),
                          **{f"f1_{c}": v for c, v in zip(classes, f1)},
                          **{"hp_" + k: v for k, v in best.items()}))
        print(f"  {arm:24s} macro-F1 {filas[-1]['macro_f1']:.3f}", flush=True)
    # McNemar pareado de cada configuracion contra la linea base espectral
    from scipy.stats import binomtest
    mc, base = {}, "s2_multitemporal"
    ok_base = pred[base] == y[hold]
    for arm in ARMS:
        if arm == base:
            continue
        ok = pred[arm] == y[hold]
        b, c = int((ok_base & ~ok).sum()), int((~ok_base & ok).sum())
        mc[f"{base} vs {arm}"] = {"solo_base": b, "solo_otro": c,
                                  "p": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0}
    return pd.DataFrame(filas), mc, imp, pred


def mlp_sensibilidad(df, classes, arms_mlp):
    """Ball et al. (2026): la ventaja de las representaciones puede exigir un
    clasificador no lineal. Se repite el regimen de bloques de 6 km con un
    perceptron pequeno, estandarizando las entradas."""
    pool = ~df["holdout"].to_numpy()
    fold = df["fold_block_6km"].to_numpy()
    xy = df[["x", "y"]].to_numpy()
    y = df["clase_nombre"].to_numpy()
    filas = []
    for arm in arms_mlp:
        X = df[ARMS[arm]].to_numpy()
        yt, yp = [], []
        for f in sorted(set(fold[fold >= 0])):
            tr = rb.buffered_train(df, fold, f, rb.REGIMES["block_6km"], pool)
            te = pool & (fold == f)
            if tr.sum() < 20 or te.sum() == 0:
                continue
            m = make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                                            early_stopping=True, random_state=rb.SEED))
            m.fit(X[tr], y[tr])
            yt.append(y[te]); yp.append(m.predict(X[te]))
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        filas.append(dict(arm=arm, n_vars=len(ARMS[arm]), n=len(yt),
                          oa=float((yt == yp).mean()),
                          macro_f1=f1_score(yt, yp, average="macro", zero_division=0)))
        print(f"  MLP {arm:24s} macro-F1 {filas[-1]['macro_f1']:.3f}", flush=True)
    return pd.DataFrame(filas)


def main():
    df = cargar()
    classes = sorted(df["clase_nombre"].unique())

    print("\n== validacion cruzada ==", flush=True)
    cv, per_fold, preds = rb.run_cv(df, ARMS, classes, SALIDA)
    cv.to_csv(SALIDA / "cv_metrics.csv", index=False)
    per_fold.to_csv(SALIDA / "cv_per_fold.csv", index=False)
    preds.to_csv(SALIDA / "cv_predictions.csv", index=False)

    print("\n== reserva espacial contigua ==", flush=True)
    hold, mc, imp, _ = reserva(df, classes)
    hold.to_csv(SALIDA / "holdout_metrics.csv", index=False)
    (SALIDA / "mcnemar_holdout.json").write_text(json.dumps(mc, indent=1), encoding="utf-8")
    (SALIDA / "importancias.json").write_text(json.dumps(imp, indent=1), encoding="utf-8")

    print("\n== sensibilidad al clasificador (MLP, bloques de 6 km) ==", flush=True)
    mlp = mlp_sensibilidad(df, classes, ["s2_multitemporal", "alphaearth", "s2_multi_mas_alphaearth"])
    mlp.to_csv(SALIDA / "mlp_sensibilidad.csv", index=False)

    tabla = (cv.pivot(index="arm", columns="regime", values="macro_f1")
             .join(hold.set_index("arm")[["macro_f1", "n_vars"]].rename(columns={"macro_f1": "reserva"})))
    tabla = tabla[["n_vars", "random", "block_6km", "block_16km", "reserva"]].round(3)
    tabla = tabla.reindex(list(ARMS))
    print("\n== macro-F1 por configuracion y regimen ==\n" + tabla.to_string(), flush=True)
    sd = (per_fold[per_fold.regime == "block_6km"].groupby("arm").macro_f1.std().round(3))
    print("\ndesviacion entre pliegues (6 km):\n" + sd.to_string(), flush=True)
    print("\nMcNemar en la reserva, contra s2_multitemporal:\n" + json.dumps(mc, indent=1), flush=True)

    json.dump({"tabla_macro_f1": json.loads(tabla.to_json(orient="index")),
               "sd_pliegues_6km": json.loads(sd.to_json()),
               "mcnemar_reserva": mc,
               "mlp": json.loads(mlp.set_index("arm").to_json(orient="index"))},
              open(SALIDA / "resumen.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nFIN", flush=True)


if __name__ == "__main__":
    main()
