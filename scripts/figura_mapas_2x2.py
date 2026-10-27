# -*- coding: utf-8 -*-
"""Vuelve a componer fig_mapa_clasificacion.png en dos filas por dos columnas, a partir de las teselas ya descargadas por
mapa_clasificacion.py (carpeta mapa_clasificacion/), para que los mapas se lean en el ancho de página del informe.
No recalcula nada: mismas teselas, mismos valores de validacion_gee_vs_sklearn.json, misma leyenda.

USO: python -X utf8 figura_mapas_2x2.py
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

AQUI = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mapa_clasificacion", AQUI / "mapa_clasificacion.py")
mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
T = AQUI / "mapa_clasificacion"
val = json.loads((T / "validacion_gee_vs_sklearn.json").read_text(encoding="utf-8"))
rutas = {arm: T / ("mapa_%s.png" % arm) for arm in mc.ARMS}; rutas["desacuerdo"] = T / "mapa_desacuerdo.png"; rutas["fondo"] = T / "mapa_fondo.png"
asp = 1.0 / np.cos(np.radians(10.8))

fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.3), dpi=150)
axes = axes.ravel()
for ax, arm in zip(axes[:3], mc.ARMS):
    ax.imshow(Image.open(rutas["fondo"])); ax.imshow(Image.open(rutas[arm]))
    ax.set_title(mc.ETQ[arm] + "\nGEE macro-F1 %.3f on the reserve (sklearn %.3f)" % (val[arm]["macro_f1_gee"], val[arm]["macro_f1_sklearn"]), fontsize=10)
ax = axes[3]
ax.imshow(Image.open(rutas["fondo"])); ax.imshow(Image.open(rutas["desacuerdo"]))
ax.set_title("(d) Where (a) and (b) disagree\n%.1f %% of the study area" % (100 * val["fraccion_desacuerdo_ae_vs_s2"]), fontsize=10)
for ax in axes:
    ax.set_aspect(asp); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9aa7ad")
leg = [Patch(facecolor="#" + c, label=e) for c, e in zip(mc.PAL, mc.ETQ_CLASE)] + [Patch(facecolor="#d9571e", label="Disagreement")]
fig.legend(handles=leg, loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("The same three input configurations applied to the whole study area, CGSM 2021", fontsize=12)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = AQUI / "figuras" / "fig_mapa_clasificacion.png"
fig.savefig(out); print("escrito", out, Image.open(out).size)
