# -*- coding: utf-8 -*-
"""Reúne en un solo Markdown las cabeceras (docstring) y los comentarios de cada script, en el orden del
flujo de trabajo del Informe 1, tal como están en el código. Escribe DOCUMENTACION_cabeceras.md; la
documentación redactada por script está en DOCUMENTACION_scripts.md.

USO: python generar_documentacion.py
"""
import ast
import io
import os
import tokenize

AQUI = os.path.dirname(os.path.abspath(__file__))
ORDEN = [
    ("1. Motor de validación (piloto, sin modificar)", ["piloto_benchmark/inspect_data.py", "piloto_benchmark/build_split.py", "piloto_benchmark/extract_spectral.py", "piloto_benchmark/run_benchmark.py", "piloto_benchmark/_proj.py"]),
    ("2. Línea base espectral multitemporal de 2021", ["linea_base_2021.py"]),
    ("3. Representaciones frente a variables espectrales", ["representaciones_2021.py", "permutacion_bloques.py", "variograma_residuos_finales.py"]),
    ("4. Pruebas de sensibilidad", ["auditoria_etiquetas.py", "sensibilidad_etiquetas.py", "diez_pliegues.py", "control_tamano_6km.py", "control_composicion_6km.py", "regimen_16km_corregido.py"]),
    ("5. Aplicación espacial", ["mapa_clasificacion.py", "mapa_desacuerdo_clases.py"]),
    ("6. Figuras e informe", ["figura_representaciones.py", "figura_mapas_2x2.py", "informe_a_word.py"]),
]


def cabecera_y_comentarios(ruta):
    src = open(ruta, encoding="utf-8").read()
    doc = ast.get_docstring(ast.parse(src)) or "(sin cabecera)"
    comentarios = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            t = tok.string.lstrip("#").strip()
            if t and not t.startswith("-*-") and not t.startswith("noqa"): comentarios.append("- línea %d: %s" % (tok.start[0], t))
    return doc, comentarios


def main():
    out = ["# Cabeceras y comentarios de los scripts\n", "Texto tal como está en el código, en el orden del flujo de trabajo; conserva las notas de desarrollo. Generado por `generar_documentacion.py`. La documentación redactada está en `DOCUMENTACION_scripts.md`.\n"]
    for titulo, archivos in ORDEN:
        out.append("\n## %s\n" % titulo)
        for a in archivos:
            ruta = os.path.join(AQUI, "scripts", a)
            if not os.path.exists(ruta): out.append("\n### `%s`\n\n(no está en el repositorio)\n" % a); continue
            doc, com = cabecera_y_comentarios(ruta)
            out.append("\n### `%s`\n\n```\n%s\n```\n" % (a, doc))
            if com: out.append("\nComentarios en el código:\n\n" + "\n".join(com) + "\n")
    open(os.path.join(AQUI, "DOCUMENTACION_cabeceras.md"), "w", encoding="utf-8").write("\n".join(out)); print("escrito DOCUMENTACION_cabeceras.md")


if __name__ == "__main__":
    main()
