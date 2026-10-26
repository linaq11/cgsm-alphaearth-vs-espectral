# -*- coding: utf-8 -*-
"""Variograma del indicador de error de clasificacion (residuo, Roberts et al. 2017) para los modelos FINALES del Informe 1
en el regimen de bloques de 6 km: linea base multitemporal (30 variables), AlphaEarth (64) y la combinacion (94). Repite
exactamente el procedimiento de run_benchmark.residual_range (14 rezagos, ventana del 30 % de la extension, sin quitar
tendencia) sobre las predicciones fuera de pliegue de representaciones_2021/cv_predictions.csv, para comprobar que el
alcance de los residuos sigue por debajo del lado del bloque y de la zona de exclusion de 3 km.
SALIDA: representaciones_2021/variograma_residuos.json
USO: python -X utf8 variograma_residuos_finales.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent; BENCH = AQUI.parent.parent / "AVANCE TESIS" / "PILOTO_BENCHMARK"
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(BENCH)); sys.path.insert(0, str(BENCH.parent / "PUBLICACION_GEE")); os.chdir(BENCH)
import _proj  # noqa: E402,F401
import representaciones_2021 as rep  # noqa: E402
from inspect_data import _variogram_core  # noqa: E402

BRAZOS = ["s2_marzo", "s2_multitemporal", "alphaearth", "s2_multi_mas_alphaearth"]


def main():
    df = rep.cargar(); preds = pd.read_csv(AQUI / "representaciones_2021" / "cv_predictions.csv")
    pool = ~df["holdout"].to_numpy(); fold = df["fold_block_6km"].to_numpy()
    idx = np.concatenate([np.nonzero(pool & (fold == f))[0] for f in sorted(set(fold[fold >= 0]))]); xy = df[["x", "y"]].to_numpy()[idx]
    res = {}
    for arm in BRAZOS:
        s = preds[(preds.regime == "block_6km") & (preds.arm == arm)]
        assert len(idx) == len(s), "%s: %d coordenadas frente a %d predicciones" % (arm, len(idx), len(s))
        assert (df["clase_nombre"].to_numpy()[idx] == s.y_true.to_numpy()).all(), arm
        err = (s.y_true.to_numpy() != s.y_pred.to_numpy()).astype(float)
        vg = _variogram_core(err, xy, n_lags=14, max_lag_pct=30.0, detrend=False)
        res[arm] = {k: v for k, v in vg.items() if k != "n_pairs"}
        print("%-24s error medio %.3f | meseta %s | alcance practico %s m | codo %s m" % (arm, err.mean(), vg.get("plateaued"), vg.get("practical_range_m"), vg.get("knee_m")), flush=True)
    json.dump(res, open(AQUI / "representaciones_2021" / "variograma_residuos.json", "w", encoding="utf-8"), indent=1, default=float)
    print("escrito representaciones_2021/variograma_residuos.json", flush=True)


if __name__ == "__main__":
    main()
