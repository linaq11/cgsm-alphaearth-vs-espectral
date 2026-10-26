# -*- coding: utf-8 -*-
"""Entre que clases ocurre el desacuerdo AlphaEarth vs S2 multitemporal del mapa, y
area por clase de cada configuracion. Mismo entrenamiento que mapa_clasificacion.py.
Salida: mapa_clasificacion/desacuerdo_por_clases.json"""
import json

import mapa_clasificacion as mc


def main():
    import ee
    ee.Initialize(project=mc.PROYECTO)
    gj = json.loads((mc.AVANCE / "AOI" / "AOI_tesis_CGSM_v1.geojson").read_text(encoding="utf-8"))
    aoi = ee.Geometry(gj["features"][0]["geometry"] if "features" in gj else gj, None, False)
    df, tr, _ = mc.particion()
    st = mc.pila(ee).clip(aoi)
    ent = mc.fc(ee, df, tr)
    cls = {}
    for arm in ("s2_multitemporal", "alphaearth"):
        f = mc.ARMS[arm]
        m = st.select(f).sampleRegions(collection=ent, properties=["clase"], scale=10, tileScale=4)
        clf = (ee.Classifier.smileRandomForest(numberOfTrees=500, minLeafPopulation=2, seed=42)
               .train(features=m, classProperty="clase", inputProperties=f))
        cls[arm] = st.select(f).classify(clf)
    par = cls["s2_multitemporal"].multiply(4).add(cls["alphaearth"]).rename("par")
    h = par.reduceRegion(ee.Reducer.frequencyHistogram(), aoi, 30, maxPixels=1e10,
                         tileScale=4).getInfo()["par"]
    tot = sum(h.values())
    out = {"pares_pct_del_area": {}, "area_pct_s2": {}, "area_pct_ae": {}}
    for k, n in sorted(h.items(), key=lambda kv: -kv[1]):
        a, b = divmod(int(float(k)), 4)
        out["pares_pct_del_area"]["%s -> %s" % (mc.CLASES[a], mc.CLASES[b])] = round(100 * n / tot, 2)
        out["area_pct_s2"][mc.CLASES[a]] = out["area_pct_s2"].get(mc.CLASES[a], 0) + 100 * n / tot
        out["area_pct_ae"][mc.CLASES[b]] = out["area_pct_ae"].get(mc.CLASES[b], 0) + 100 * n / tot
    for k in ("area_pct_s2", "area_pct_ae"):
        out[k] = {c: round(v, 2) for c, v in out[k].items()}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    (mc.SALIDA / "desacuerdo_por_clases.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
