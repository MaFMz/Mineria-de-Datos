#!/usr/bin/env python3
"""
pruebas_gw2.py

Pruebas estadisticas para comprobar si existen diferencias significativas
entre grupos etiquetados (rarity, type, level) en las variables de mercado
del dataset de GW2.

Flujo automatico por cada prueba definida en PRUEBAS:

  1. Preparar la muestra (limpieza + unidad de analisis).
  2. Verificar supuestos: normalidad por grupo (Shapiro-Wilk) y
     homogeneidad de varianzas (Levene, centrado en la mediana).
  3. Elegir la prueba:
        2 grupos  -> t de Welch          (si hay normalidad)
                     U de Mann-Whitney   (si no)
        3+ grupos -> ANOVA + Tukey HSD   (si hay normalidad y varianzas homogeneas)
                     Kruskal-Wallis + Mann-Whitney por pares con Bonferroni (si no)
  4. Calcular el TAMANO DEL EFECTO (eta^2, epsilon^2, d de Cohen o r
     biserial de rangos). Con muchos datos casi cualquier diferencia sale
     "significativa"; el tamano del efecto dice si importa en la practica.

Ademas se incluye una prueba pareada (buy_price_avg vs sell_price_avg) para
contrastar si el precio de venta es mayor que el de compra.

UNIDAD DE ANALISIS: los registros diarios de un mismo item no son
independientes (el precio de hoy depende del de ayer). Por eso, por
defecto cada item aporta UNA observacion (la mediana de su historico), lo
que hace validos los supuestos de independencia de las pruebas. Cambia
UNIDAD_ANALISIS a "registro" para usar todas las filas diarias (los
p-values saldran mucho mas pequenos, pero inflados por la dependencia).
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

CSV_PATH = "dataset.csv"
OUTPUT_DIR = Path("pruebas_estadisticas")

ALPHA = 0.05            # nivel de significancia
MIN_GRUPO = 3           # grupos con menos observaciones se excluyen de la prueba
UNIDAD_ANALISIS = "item"    # "item" (1 obs. por item) o "registro" (todas las filas)
AGREGACION = "median"       # como resumir el historico de cada item: "median" o "mean"

# Cada prueba: variable de agrupacion (etiqueta), variable respuesta y
#   cero: columnas donde 0 significa "sin actividad" y se excluye
#         (en precios). En volumen (sold) el 0 es un valor real.
#   log : True aplica log10 a la respuesta (los precios abarcan varios
#         ordenes de magnitud y son muy asimetricos).
PRUEBAS = [
    {"grupo": "rarity", "respuesta": "sell_price_avg", "cero": ["sell_price_avg"], "log": True},
    {"grupo": "type",   "respuesta": "sell_price_avg", "cero": ["sell_price_avg"], "log": True},
    {"grupo": "level",  "respuesta": "sell_price_avg", "cero": ["sell_price_avg"], "log": True},
    {"grupo": "rarity", "respuesta": "sell_sold",      "cero": [],                 "log": False},
    {"grupo": "type",   "respuesta": "sell_sold",      "cero": [],                 "log": False},
]


# ---------------------------------------------------------------------------
# PREPARACION DE LA MUESTRA
# ---------------------------------------------------------------------------

def preparar_muestra(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Devuelve un DataFrame con columnas [grupo, valor] listo para probar."""
    g, y = spec["grupo"], spec["respuesta"]
    d = df.dropna(subset=[g, y])
    for col in spec.get("cero", []):
        d = d[d[col] != 0]
    if spec.get("log"):
        d = d[d[y] > 0].assign(**{y: np.log10(d.loc[d[y] > 0, y])})

    if UNIDAD_ANALISIS == "item":
        d = d.groupby("itemID").agg(grupo=(g, "first"), valor=(y, AGREGACION)).reset_index(drop=True)
    else:
        d = d.rename(columns={g: "grupo", y: "valor"})[["grupo", "valor"]]
    return d


def separar_grupos(muestra: pd.DataFrame):
    """Devuelve ({etiqueta: array}, [etiquetas excluidas por tamano])."""
    grupos, excluidos = {}, []
    for etiqueta, sub in muestra.groupby("grupo"):
        if len(sub) >= MIN_GRUPO:
            grupos[etiqueta] = sub["valor"].to_numpy()
        else:
            excluidos.append(f"{etiqueta} (n={len(sub)})")
    return grupos, excluidos


# ---------------------------------------------------------------------------
# SUPUESTOS Y TAMANOS DE EFECTO
# ---------------------------------------------------------------------------

def verificar_supuestos(grupos: dict) -> dict:
    """Shapiro-Wilk por grupo (submuestra de 5000 si es mas grande) y Levene."""
    rng = np.random.default_rng(42)
    p_normal = {}
    for etiqueta, x in grupos.items():
        muestra = rng.choice(x, 5000, replace=False) if len(x) > 5000 else x
        # Shapiro no es valido si todos los valores son iguales
        p_normal[etiqueta] = stats.shapiro(muestra).pvalue if np.ptp(muestra) > 0 else 0.0
    _, p_levene = stats.levene(*grupos.values(), center="median")
    return {
        "normalidad_ok": all(p > ALPHA for p in p_normal.values()),
        "p_normalidad_min": min(p_normal.values()),
        "p_levene": p_levene,
        "varianzas_ok": p_levene > ALPHA,
    }


def magnitud_eta(v: float) -> str:      # eta^2 y epsilon^2
    return "insignificante" if v < 0.01 else "pequeno" if v < 0.06 else "mediano" if v < 0.14 else "grande"


def magnitud_d(v: float) -> str:        # d de Cohen
    v = abs(v)
    return "insignificante" if v < 0.2 else "pequeno" if v < 0.5 else "mediano" if v < 0.8 else "grande"


def magnitud_r(v: float) -> str:        # r biserial de rangos
    v = abs(v)
    return "insignificante" if v < 0.1 else "pequeno" if v < 0.3 else "mediano" if v < 0.5 else "grande"


def eta_cuadrado(grupos: dict) -> float:
    todos = np.concatenate(list(grupos.values()))
    ss_total = ((todos - todos.mean()) ** 2).sum()
    ss_entre = sum(len(x) * (x.mean() - todos.mean()) ** 2 for x in grupos.values())
    return ss_entre / ss_total if ss_total > 0 else np.nan


# ---------------------------------------------------------------------------
# PRUEBAS
# ---------------------------------------------------------------------------

def comparar_dos_grupos(grupos: dict, parametrica: bool) -> dict:
    (n1, a), (n2, b) = grupos.items()
    if parametrica:
        res = stats.ttest_ind(a, b, equal_var=False)   # t de Welch
        d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return {"prueba": "t de Welch", "estadistico": res.statistic, "p": res.pvalue,
                "efecto_nombre": "d de Cohen", "efecto": d, "magnitud": magnitud_d(d), "posthoc": None}
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * res.statistic / (len(a) * len(b)) - 1
    return {"prueba": "U de Mann-Whitney", "estadistico": res.statistic, "p": res.pvalue,
            "efecto_nombre": "r biserial", "efecto": r, "magnitud": magnitud_r(r), "posthoc": None}


def comparar_varios_grupos(grupos: dict, parametrica: bool) -> dict:
    etiquetas = list(grupos)
    pares = list(itertools.combinations(etiquetas, 2))

    if parametrica:
        res = stats.f_oneway(*grupos.values())
        eta = eta_cuadrado(grupos)
        tukey = stats.tukey_hsd(*grupos.values())
        filas = [{"grupo_a": etiquetas[i], "grupo_b": etiquetas[j],
                  "dif_medias": grupos[etiquetas[i]].mean() - grupos[etiquetas[j]].mean(),
                  "p_ajustado": tukey.pvalue[i, j]}
                 for i, j in itertools.combinations(range(len(etiquetas)), 2)]
        return {"prueba": "ANOVA + Tukey HSD", "estadistico": res.statistic, "p": res.pvalue,
                "efecto_nombre": "eta^2", "efecto": eta, "magnitud": magnitud_eta(eta),
                "posthoc": pd.DataFrame(filas)}

    res = stats.kruskal(*grupos.values())
    n = sum(len(x) for x in grupos.values())
    epsilon = res.statistic / (n - 1)     # epsilon^2 = H / (n - 1)
    filas = []
    for a, b in pares:
        p = stats.mannwhitneyu(grupos[a], grupos[b], alternative="two-sided").pvalue
        filas.append({"grupo_a": a, "grupo_b": b,
                      "dif_medianas": np.median(grupos[a]) - np.median(grupos[b]),
                      "p_ajustado": min(p * len(pares), 1.0)})   # Bonferroni
    return {"prueba": "Kruskal-Wallis + Mann-Whitney (Bonferroni)", "estadistico": res.statistic,
            "p": res.pvalue, "efecto_nombre": "epsilon^2", "efecto": epsilon,
            "magnitud": magnitud_eta(epsilon), "posthoc": pd.DataFrame(filas)}


def ejecutar_prueba(df: pd.DataFrame, spec: dict) -> dict:
    muestra = preparar_muestra(df, spec)
    grupos, excluidos = separar_grupos(muestra)
    resultado = {"grupo": spec["grupo"], "respuesta": spec["respuesta"],
                 "escala": "log10" if spec.get("log") else "original",
                 "excluidos": excluidos, "n_total": sum(len(x) for x in grupos.values())}
    if len(grupos) < 2:
        resultado["error"] = "menos de 2 grupos con suficientes observaciones"
        return resultado

    supuestos = verificar_supuestos(grupos)
    # 2 grupos: el t de Welch tolera varianzas distintas, basta la normalidad.
    # 3+ grupos: ANOVA exige normalidad y varianzas homogeneas.
    parametrica = supuestos["normalidad_ok"] and (len(grupos) == 2 or supuestos["varianzas_ok"])

    prueba = comparar_dos_grupos(grupos, parametrica) if len(grupos) == 2 \
        else comparar_varios_grupos(grupos, parametrica)

    descriptivo = pd.DataFrame({
        "n": {k: len(v) for k, v in grupos.items()},
        "media": {k: v.mean() for k, v in grupos.items()},
        "mediana": {k: np.median(v) for k, v in grupos.items()},
        "desv_est": {k: v.std(ddof=1) for k, v in grupos.items()},
    })
    resultado.update(supuestos)
    resultado.update(prueba)
    resultado["descriptivo"] = descriptivo
    resultado["k"] = len(grupos)
    return resultado


def prueba_pareada(df: pd.DataFrame, col_a: str = "buy_price_avg", col_b: str = "sell_price_avg") -> dict:
    """Compara precio de compra vs venta sobre el MISMO item (muestras pareadas)."""
    d = df.dropna(subset=[col_a, col_b])
    d = d[(d[col_a] > 0) & (d[col_b] > 0)]
    if UNIDAD_ANALISIS == "item":
        d = d.groupby("itemID")[[col_a, col_b]].agg(AGREGACION)
    a, b = np.log10(d[col_a].to_numpy()), np.log10(d[col_b].to_numpy())
    dif = b - a
    normal = stats.shapiro(dif if len(dif) <= 5000 else np.random.default_rng(42).choice(dif, 5000, replace=False)).pvalue > ALPHA
    if normal:
        res = stats.ttest_rel(b, a)
        efecto = dif.mean() / dif.std(ddof=1)
        nombre_p, nombre_e, mag = "t pareada", "d de Cohen (pareada)", magnitud_d(efecto)
    else:
        res = stats.wilcoxon(b, a)
        z = (dif > 0).mean() - (dif < 0).mean()          # r biserial pareado
        efecto, nombre_p, nombre_e, mag = z, "Wilcoxon", "r biserial", magnitud_r(z)
    return {"prueba": nombre_p, "n": len(dif), "estadistico": res.statistic, "p": res.pvalue,
            "efecto_nombre": nombre_e, "efecto": efecto, "magnitud": mag,
            "venta_mayor_pct": (dif > 0).mean() * 100,
            "razon_mediana": 10 ** np.median(dif)}


# ---------------------------------------------------------------------------
# REPORTE
# ---------------------------------------------------------------------------

def formatear_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def imprimir_resultado(r: dict):
    print("=" * 78)
    print(f"{r['respuesta']} por {r['grupo']}  (escala {r['escala']})")
    if "error" in r:
        print(f"  No se pudo probar: {r['error']}")
        return
    if r["excluidos"]:
        print(f"  Grupos excluidos por n < {MIN_GRUPO}: {', '.join(r['excluidos'])}")
    print(f"  Supuestos -> normalidad: {'OK' if r['normalidad_ok'] else 'NO'} "
          f"(p min Shapiro = {formatear_p(r['p_normalidad_min'])}), "
          f"varianzas homogeneas: {'OK' if r['varianzas_ok'] else 'NO'} (Levene p = {formatear_p(r['p_levene'])})")
    print(f"  Prueba: {r['prueba']}  |  estadistico = {r['estadistico']:.3f}  |  p = {formatear_p(r['p'])}")
    print(f"  Tamano del efecto: {r['efecto_nombre']} = {r['efecto']:.3f} ({r['magnitud']})")

    if r["p"] < ALPHA:
        conclusion = f"Hay diferencias significativas (alfa = {ALPHA})"
        if r["magnitud"] == "insignificante":
            conclusion += ", pero el efecto es insignificante en la practica"
    else:
        conclusion = f"No hay evidencia de diferencias significativas (alfa = {ALPHA})"
    print(f"  Conclusion: {conclusion}.")

    desc = r["descriptivo"].sort_values("mediana", ascending=False)
    print(f"  Mediana mas alta: {desc.index[0]} ({desc['mediana'].iloc[0]:.3f}); "
          f"mas baja: {desc.index[-1]} ({desc['mediana'].iloc[-1]:.3f})")
    if r["posthoc"] is not None:
        sig = r["posthoc"][r["posthoc"]["p_ajustado"] < ALPHA]
        print(f"  Post-hoc: {len(sig)} de {len(r['posthoc'])} pares difieren significativamente")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    print(f"Unidad de analisis: {UNIDAD_ANALISIS} ({AGREGACION} por item)" if UNIDAD_ANALISIS == "item"
          else "Unidad de analisis: registro diario")

    resumen = []
    for spec in PRUEBAS:
        r = ejecutar_prueba(df, spec)
        imprimir_resultado(r)
        if "error" in r:
            continue
        nombre = f"{spec['respuesta']}_por_{spec['grupo']}"
        r["descriptivo"].round(4).to_csv(OUTPUT_DIR / f"descriptivo_{nombre}.csv")
        if r["posthoc"] is not None:
            r["posthoc"].round(6).to_csv(OUTPUT_DIR / f"posthoc_{nombre}.csv", index=False)
        resumen.append({k: r[k] for k in ["respuesta", "grupo", "escala", "k", "n_total", "prueba",
                                          "estadistico", "p", "efecto_nombre", "efecto", "magnitud"]})

    print("=" * 78)
    print("Prueba pareada: buy_price_avg vs sell_price_avg (log10)")
    pp = prueba_pareada(df)
    print(f"  {pp['prueba']} (n = {pp['n']}) | estadistico = {pp['estadistico']:.3f} | p = {formatear_p(pp['p'])}")
    print(f"  {pp['efecto_nombre']} = {pp['efecto']:.3f} ({pp['magnitud']})")
    print(f"  El precio de venta supera al de compra en {pp['venta_mayor_pct']:.1f}% de los casos; "
          f"razon mediana venta/compra = {pp['razon_mediana']:.3f}")

    pd.DataFrame(resumen).to_csv(OUTPUT_DIR / "resumen_pruebas.csv", index=False)
    print(f"\nResultados guardados en '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
