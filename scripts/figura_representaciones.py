# -*- coding: utf-8 -*-
"""Figura principal del informe 1: macro-F1 por configuracion de entrada y regimen de
validacion, con la desviacion entre pliegues como referencia de lo que es una diferencia
apreciable. Figuras en ingles. Salida: figuras/fig_representaciones.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "representaciones_2021")
cv = pd.read_csv(os.path.join(RES, "cv_metrics.csv"))
hold = pd.read_csv(os.path.join(RES, "holdout_metrics.csv")).set_index("arm")
pf = pd.read_csv(os.path.join(RES, "cv_per_fold.csv"))
mlp = pd.read_csv(os.path.join(RES, "mlp_sensibilidad.csv")).set_index("arm")

ARMS = ["s2_marzo", "s2_seca_humeda", "s2_multitemporal", "alphaearth", "s2_multi_mas_alphaearth"]
ETQ = {"s2_marzo": "S2, single date\n(March, 9 vars)",
       "s2_seca_humeda": "S2, dry + wet\n(18 vars)",
       "s2_multitemporal": "S2, multi-temporal\n(30 vars)",
       "alphaearth": "AlphaEarth\n(64 vars)",
       "s2_multi_mas_alphaearth": "S2 multi-temporal\n+ AlphaEarth (94)"}
REG = [("random", "Random folds\n(leaky control)"), ("block_6km", "6 km blocks\n(primary)"),
       ("block_16km", "16 km blocks\n(conservative)"), ("reserva", "Contiguous reserve\n(single evaluation)")]
COL = {"s2_marzo": "#c6c1b8", "s2_seca_humeda": "#9aa7ad", "s2_multitemporal": "#d9571e",
       "alphaearth": "#1f6f78", "s2_multi_mas_alphaearth": "#6b3fa0"}

tabla = cv.pivot(index="arm", columns="regime", values="macro_f1")
tabla["reserva"] = hold["macro_f1"]
tabla = tabla.reindex(ARMS)
sd6 = pf[pf.regime == "block_6km"].groupby("arm").macro_f1.std()

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=150, gridspec_kw={"width_ratios": [2.05, 1]})

x = np.arange(len(REG)); w = 0.16
for k, arm in enumerate(ARMS):
    v = [tabla.loc[arm, r] for r, _ in REG]
    ax.bar(x + (k - 2) * w, v, w, color=COL[arm], label=ETQ[arm].replace("\n", " "))
# barra de error solo en el regimen principal: desviacion entre los cinco pliegues espaciales
for k, arm in enumerate(ARMS):
    ax.errorbar(x[1] + (k - 2) * w, tabla.loc[arm, "block_6km"], yerr=sd6[arm],
                color="#14212a", capsize=2.5, lw=1.1, zorder=5)
# etiquetas horizontales, siempre por encima de la barra y de su barra de error
for k, arm in enumerate(ARMS):
    for j, (r, _) in enumerate(REG):
        v = tabla.loc[arm, r]
        alto = v + (sd6[arm] if r == "block_6km" else 0) + 0.008
        ax.annotate(f"{v:.3f}", (x[j] + (k - 2) * w, alto), ha="center", va="bottom",
                    fontsize=6.6, color="#14212a")
ax.set_xticks(x); ax.set_xticklabels([t for _, t in REG], fontsize=8.5)
ax.set_ylim(0.6, 0.98); ax.set_ylabel("Macro-F1")
ax.set_title("(a) Same 600 points, same folds, same Random Forest: only the input changes\n"
             "Error bars on the primary regime: standard deviation across the five spatial folds", fontsize=9.5)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=7.4, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)

# panel b: diferencia PAREADA frente a la linea base, pliegue por pliegue. La barra es la
# sd de la diferencia, no la sd de cada configuracion: los sectores dificiles lo son para
# las dos entradas y esa parte comun no es incertidumbre de la diferencia.
pb = json.load(open(os.path.join(RES, "permutacion_bloques.json"), encoding="utf-8"))["principal"]
REG_CV = [("random", "Random"), ("block_6km", "6 km blocks"), ("block_16km", "16 km blocks")]
ax2.axhline(0, color="#14212a", lw=1)
for k, (arm, mk, off) in enumerate([("alphaearth", "o", -0.12), ("s2_multi_mas_alphaearth", "s", 0.12)]):
    for j, (r, _) in enumerate(REG_CV):
        t = pf[pf.regime == r].pivot(index="fold", columns="arm", values="macro_f1")
        d = (t[arm] - t["s2_multitemporal"]).to_numpy()
        ax2.scatter(np.full(len(d), j + off), d, s=11, color=COL[arm], alpha=0.35, lw=0)
        ax2.errorbar(j + off, d.mean(), yerr=d.std(ddof=1), fmt=mk, color=COL[arm], capsize=3, ms=6,
                     label=None if j else ("AlphaEarth" if arm == "alphaearth" else "S2 + AlphaEarth")
                     + " − S2 multi-temporal")
        # p de la permutacion pareada por bloques (permutacion_bloques.py), no el de McNemar,
        # que trata como independientes puntos vecinos
        mc = pb.get(f"{r}|{arm}")
        if mc:
            ax2.annotate(f"p={mc['p_perm_macro_f1']:.3f}" if mc["p_perm_macro_f1"] >= 0.001 else "p<0.001",
                         (j + off, d.mean() + d.std(ddof=1)), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=6.8, color=COL[arm])
ax2.set_xticks(range(len(REG_CV))); ax2.set_xticklabels([t for _, t in REG_CV], fontsize=8)
ax2.set_xlim(-0.55, len(REG_CV) - 0.45)
ax2.set_ylabel("Δ Macro-F1 within each fold"); ax2.grid(alpha=0.3)
ax2.set_title("(b) Paired gain over the multi-temporal baseline\n"
              "Dots: folds · bars: mean ± paired sd\np: paired permutation of spatial blocks (53 and 16 blocks)",
              fontsize=9)
ax2.legend(fontsize=7.2, loc="lower left")
fig.suptitle("Foundation-model embeddings against multi-temporal spectral features, CGSM 2021, four mangrove-related classes", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(AQUI, "figuras", "fig_representaciones.png"))
print("figuras/fig_representaciones.png")

# resumen numerico para el texto
print("\nsd marginal entre pliegues 6 km:", sd6.round(3).to_dict())
print("MLP (6 km):", mlp.macro_f1.round(3).to_dict())
