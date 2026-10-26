# -*- coding: utf-8 -*-
r"""Rehace las comparaciones pareadas del Informe 1 con una prueba valida bajo dependencia espacial.

McNemar sobre las predicciones fuera de pliegue reunidas trata cada punto como una prueba
independiente, y los puntos vecinos no lo son: su p sale optimista. Aqui la unidad que se
permuta es el bloque espacial del propio regimen (6 km o 16 km):

  - permutacion pareada: en cada repeticion se intercambian al azar, bloque por bloque, las
    predicciones de la linea base y las del otro brazo, y se recalcula la diferencia;
  - intervalo: remuestreo de bloques con reemplazo.

No reentrena nada. Para el analisis principal reconstruye a que punto corresponde cada fila
de cv_predictions.csv con el mismo orden que usa run_benchmark.residual_range, y lo comprueba
con y_true. Las dos pruebas de sensibilidad guardan ya sus predicciones con coordenadas.

USO: python permutacion_bloques.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import representaciones_2021 as rep

AQUI = Path(__file__).resolve().parent
SAL = AQUI / "representaciones_2021"
BASE, N_PERM, N_BOOT, SEMILLA = "s2_multitemporal", 20000, 5000, 2021
LADO = {"block_6km": 6000.0, "block_16km": 16000.0}


def macro(yt, yp, C):
    """macro-F1 con etiquetas enteras 0..C-1; equivale a f1_score(average="macro", zero_division=0)
    y es unas cien veces mas rapido, que es lo que hace viable la permutacion."""
    cm = np.bincount(yt * C + yp, minlength=C * C).reshape(C, C)
    den = cm.sum(0) + cm.sum(1)
    return float(np.where(den > 0, 2 * np.diag(cm) / np.maximum(den, 1), 0.0).mean())


def con_xy(df, preds):
    """Devuelve preds con point_id, x, y de cada fila. run_cv concatena los pliegues de prueba
    en orden ascendente, asi que ese mismo orden recupera el punto; se comprueba con y_true."""
    pool = ~df["holdout"].to_numpy()
    out = []
    for (reg, arm), s in preds.groupby(["regime", "arm"], sort=False):
        fold = df[f"fold_{reg}"].to_numpy()
        idx = np.concatenate([np.nonzero(pool & (fold == f))[0] for f in sorted(set(s.fold))])
        assert len(idx) == len(s) and (df["clase_nombre"].to_numpy()[idx] == s.y_true.to_numpy()).all(), (reg, arm)
        out.append(s.assign(point_id=df["point_id"].to_numpy()[idx], x=df["x"].to_numpy()[idx],
                            y=df["y"].to_numpy()[idx]))
    return pd.concat(out, ignore_index=True)


def comparar(p, rng, etiqueta):
    """p: predicciones con x, y de un solo analisis. Devuelve {regimen|brazo: resultado}."""
    nombres = sorted(p.y_true.unique())
    cod = {c: i for i, c in enumerate(nombres)}
    C, res = len(nombres), {}
    for reg in [r for r in LADO if r in set(p.regime)]:
        s = p[p.regime == reg]
        b = s[s.arm == BASE]
        g = np.floor(b[["x", "y"]].to_numpy() / LADO[reg]).astype(np.int64)
        blo = pd.factorize(g[:, 0] * 100000 + g[:, 1])[0]
        nb = blo.max() + 1
        por_bloque = [np.nonzero(blo == k)[0] for k in range(nb)]
        yt, pb = b.y_true.map(cod).to_numpy(), b.y_pred.map(cod).to_numpy()
        assert abs(macro(yt, pb, C) - f1_score(b.y_true, b.y_pred, labels=nombres, average="macro",
                                               zero_division=0)) < 1e-12
        print("\n== %s | %s | %d puntos en %d bloques de %.0f km"
              % (etiqueta, reg, len(b), nb, LADO[reg] / 1000), flush=True)
        for arm in [a for a in s.arm.unique() if a != BASE]:
            o = s[s.arm == arm]
            assert (o.point_id.to_numpy() == b.point_id.to_numpy()).all()
            po = o.y_pred.map(cod).to_numpy()
            d_f1 = macro(yt, po, C) - macro(yt, pb, C)
            d_oa = (po == yt).mean() - (pb == yt).mean()
            solo_b, solo_o = int(((pb == yt) & (po != yt)).sum()), int(((pb != yt) & (po == yt)).sum())
            pf, pa = np.empty(N_PERM), np.empty(N_PERM)
            for i in range(N_PERM):
                sw = rng.random(nb)[blo] < 0.5
                a, c = np.where(sw, po, pb), np.where(sw, pb, po)      # a hace de base, c de otro
                pf[i] = macro(yt, c, C) - macro(yt, a, C)
                pa[i] = (c == yt).mean() - (a == yt).mean()
            p_f1 = (1 + (np.abs(pf) >= abs(d_f1) - 1e-12).sum()) / (1 + N_PERM)
            p_oa = (1 + (np.abs(pa) >= abs(d_oa) - 1e-12).sum()) / (1 + N_PERM)
            bf = np.empty(N_BOOT)
            for i in range(N_BOOT):
                ii = np.concatenate([por_bloque[k] for k in rng.integers(0, nb, nb)])
                bf[i] = macro(yt[ii], po[ii], C) - macro(yt[ii], pb[ii], C)
            lo, hi = np.percentile(bf, [2.5, 97.5])
            res[f"{reg}|{arm}"] = dict(n=int(len(b)), bloques=int(nb), solo_base=solo_b, solo_otro=solo_o,
                                       d_macro_f1=round(float(d_f1), 4),
                                       ic95=[round(float(lo), 4), round(float(hi), 4)],
                                       p_perm_macro_f1=round(float(p_f1), 4), d_oa=round(float(d_oa), 4),
                                       p_perm_oa=round(float(p_oa), 4))
            print("  %-26s discordantes %2d/%2d | d macro-F1 %+.3f [%+.3f a %+.3f] p perm %.4f | d OA %+.3f p perm %.4f"
                  % (arm, solo_b, solo_o, d_f1, lo, hi, p_f1, d_oa, p_oa), flush=True)
    return res


def main():
    rng = np.random.default_rng(SEMILLA)
    principal = con_xy(rep.cargar(), pd.read_csv(SAL / "cv_predictions.csv"))
    res = {"principal": comparar(principal, rng, "analisis principal")}
    for carpeta in ("sensibilidad_etiquetas", "diez_pliegues"):
        f = AQUI / carpeta / "cv_predictions.csv"
        if f.exists() and "x" in pd.read_csv(f, nrows=1).columns:
            res[carpeta] = comparar(pd.read_csv(f), rng, carpeta)
        else:
            print("\n(%s sin predicciones por punto: no se prueba)" % carpeta)
    (SAL / "permutacion_bloques.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
