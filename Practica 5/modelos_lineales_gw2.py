#!/usr/bin/env python3
"""
modelos_lineales_gw2.py

Modelos lineales y correlacion sobre el dataset de mercado de GW2.

Contenido, generado automaticamente mediante ciclos sobre listas de
especificaciones (mismo patron que graficas_gw2.py y pruebas_gw2.py):

  1. Matriz de correlacion (Pearson y Spearman) entre las variables
     numericas de interes, con su mapa de calor.
  2. Regresiones lineales SIMPLES (una por cada par x/y en MODELOS):
     grafica de dispersion + recta ajustada, grafica de residuales, y la
     metrica R^2 de cada una.
  3. Una regresion lineal MULTIPLE (varios predictores a la vez) como
     ejemplo de como combinar variables, con su propia R^2 y graficas de
     valores reales vs. predichos y de residuales.

UNIDAD DE ANALISIS: igual que en pruebas_gw2.py, cada item aporta una
sola observacion (la mediana de su historico), porque los registros
diarios de un mismo item no son independientes entre si. Esto evita que
el R^2 se vea artificialmente alto solo por tener miles de filas
repetidas del mismo item.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

CSV_PATH = "dataset.csv"
OUTPUT_DIR = Path("modelos_lineales")

UNIDAD_ANALISIS = "item"   # "item" (1 obs. por item, recomendado) o "registro"
AGREGACION = "median"

plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["figure.dpi"] = 120

# Variables numericas para la matriz de correlacion. "log" = True aplica
# log10 (precios y volumenes abarcan varios ordenes de magnitud).
VARIABLES_CORRELACION = [
    {"col": "buy_price_avg", "log": True,  "etiqueta": "log10(buy_price_avg)"},
    {"col": "sell_price_avg", "log": True,  "etiqueta": "log10(sell_price_avg)"},
    {"col": "buy_sold",       "log": True,  "etiqueta": "log10(buy_sold)"},
    {"col": "sell_sold",      "log": True,  "etiqueta": "log10(sell_sold)"},
    {"col": "level",          "log": False, "etiqueta": "level"},
    {"col": "vendor_value",   "log": True,  "etiqueta": "log10(vendor_value)"},
]

# Regresiones lineales simples: una recta por cada especificacion.
MODELOS = [
    {
        "archivo": "modelo_01_buy_vs_sell.png",
        "x": "buy_price_avg", "log_x": True,
        "y": "sell_price_avg", "log_y": True,
        "titulo": "Precio de venta en funcion del precio de compra",
    },
    {
        "archivo": "modelo_02_volumen_vs_precio.png",
        "x": "sell_sold", "log_x": True,
        "y": "sell_price_avg", "log_y": True,
        "titulo": "Precio de venta en funcion del volumen vendido",
    },
]

# Regresion lineal multiple: una sola, con varios predictores.
MODELO_MULTIPLE = {
    "y": "sell_price_avg", "log_y": True,
    "predictores": [
        {"col": "sell_sold", "log": True},
        {"col": "level", "log": False},
    ],
    "archivo_base": "modelo_multiple",
    "titulo": "log10(sell_price_avg) ~ log10(sell_sold) + level",
}


# ---------------------------------------------------------------------------
# PREPARACION DE DATOS
# ---------------------------------------------------------------------------

def preparar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce el dataset a una fila por item (mediana del historico) y
    conserva los atributos propios del item (type, rarity, level, ...)."""
    columnas_numericas = ["buy_price_avg", "sell_price_avg", "buy_sold", "sell_sold",
                           "level", "vendor_value"]
    columnas_numericas = [c for c in columnas_numericas if c in df.columns]

    if UNIDAD_ANALISIS == "item":
        agg = df.groupby("itemID")[columnas_numericas].agg(AGREGACION)
        atributos = df.groupby("itemID")[["type", "rarity"]].first() if "type" in df.columns else None
        base = agg.join(atributos) if atributos is not None else agg
        return base.reset_index()
    return df[["itemID"] + columnas_numericas + [c for c in ["type", "rarity"] if c in df.columns]].copy()


def transformar(serie: pd.Series, usar_log: bool) -> pd.Series:
    if not usar_log:
        return serie
    serie = serie.where(serie > 0)   # log10 de 0 o negativos no esta definido
    return np.log10(serie)


# ---------------------------------------------------------------------------
# 1. CORRELACION
# ---------------------------------------------------------------------------

def calcular_correlaciones(base: pd.DataFrame):
    datos = pd.DataFrame({
        v["etiqueta"]: transformar(base[v["col"]], v["log"])
        for v in VARIABLES_CORRELACION if v["col"] in base.columns
    }).dropna()

    pearson = datos.corr(method="pearson")
    spearman = datos.corr(method="spearman")
    return datos, pearson, spearman


def graficar_correlacion(matriz: pd.DataFrame, titulo: str, archivo: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(matriz.columns)))
    ax.set_xticklabels(matriz.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matriz.index)))
    ax.set_yticklabels(matriz.index)
    for i in range(len(matriz.index)):
        for j in range(len(matriz.columns)):
            ax.text(j, i, f"{matriz.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, label="coeficiente de correlacion")
    fig.tight_layout()
    fig.savefig(archivo, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. REGRESION LINEAL SIMPLE (ciclo sobre MODELOS)
# ---------------------------------------------------------------------------

def ajustar_modelo_simple(base: pd.DataFrame, spec: dict) -> dict:
    x = transformar(base[spec["x"]], spec["log_x"])
    y = transformar(base[spec["y"]], spec["log_y"])
    datos = pd.DataFrame({"x": x, "y": y}).dropna()

    resultado = stats.linregress(datos["x"], datos["y"])
    r2 = resultado.rvalue ** 2
    predicho = resultado.intercept + resultado.slope * datos["x"]
    residuales = datos["y"] - predicho

    return {
        "spec": spec, "datos": datos, "predicho": predicho, "residuales": residuales,
        "pendiente": resultado.slope, "intercepto": resultado.intercept,
        "r": resultado.rvalue, "r2": r2, "p": resultado.pvalue, "n": len(datos),
    }


def graficar_modelo_simple(ajuste: dict, ruta: Path):
    spec, datos, predicho, residuales = ajuste["spec"], ajuste["datos"], ajuste["predicho"], ajuste["residuales"]
    etiqueta_x = f"log10({spec['x']})" if spec["log_x"] else spec["x"]
    etiqueta_y = f"log10({spec['y']})" if spec["log_y"] else spec["y"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    orden = datos["x"].sort_values().index
    ax1.scatter(datos["x"], datos["y"], alpha=0.5, s=18, label="items")
    ax1.plot(datos["x"].loc[orden], predicho.loc[orden], color="firebrick", linewidth=2,
              label=f"y = {ajuste['pendiente']:.3f}x + {ajuste['intercepto']:.3f}")
    ax1.set_xlabel(etiqueta_x)
    ax1.set_ylabel(etiqueta_y)
    ax1.set_title(f"{spec['titulo']}\nR² = {ajuste['r2']:.3f}   n = {ajuste['n']}")
    ax1.legend()

    ax2.scatter(predicho, residuales, alpha=0.5, s=18)
    ax2.axhline(0, color="firebrick", linewidth=1.5)
    ax2.set_xlabel(f"{etiqueta_y} predicho")
    ax2.set_ylabel("residual (real - predicho)")
    ax2.set_title("Residuales vs. valores predichos")

    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. REGRESION LINEAL MULTIPLE
# ---------------------------------------------------------------------------

def ajustar_modelo_multiple(base: pd.DataFrame, spec: dict) -> dict:
    y = transformar(base[spec["y"]], spec["log_y"])
    predictores = {p["col"]: transformar(base[p["col"]], p["log"]) for p in spec["predictores"]}
    datos = pd.DataFrame({"y": y, **predictores}).dropna()

    nombres = list(predictores.keys())
    X = np.column_stack([np.ones(len(datos))] + [datos[c].to_numpy() for c in nombres])
    y_vec = datos["y"].to_numpy()

    coeficientes, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)
    predicho = X @ coeficientes
    residuales = y_vec - predicho

    ss_res = (residuales ** 2).sum()
    ss_tot = ((y_vec - y_vec.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    n, k = len(datos), len(nombres)
    r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    return {
        "spec": spec, "datos": datos, "nombres": nombres,
        "intercepto": coeficientes[0], "coeficientes": dict(zip(nombres, coeficientes[1:])),
        "predicho": predicho, "residuales": residuales, "r2": r2, "r2_ajustado": r2_ajustado,
        "n": n, "k": k,
    }


def graficar_modelo_multiple(ajuste: dict, ruta: Path):
    spec, predicho, residuales = ajuste["spec"], ajuste["predicho"], ajuste["residuales"]
    y_real = ajuste["datos"]["y"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    lim_min, lim_max = min(y_real.min(), predicho.min()), max(y_real.max(), predicho.max())
    ax1.scatter(y_real, predicho, alpha=0.5, s=18)
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], color="firebrick", linewidth=1.5, label="y = x (ajuste perfecto)")
    ax1.set_xlabel("valor real")
    ax1.set_ylabel("valor predicho")
    ax1.set_title(f"{spec['titulo']}\nR² = {ajuste['r2']:.3f}   R² ajustado = {ajuste['r2_ajustado']:.3f}")
    ax1.legend()

    ax2.scatter(predicho, residuales, alpha=0.5, s=18)
    ax2.axhline(0, color="firebrick", linewidth=1.5)
    ax2.set_xlabel("valor predicho")
    ax2.set_ylabel("residual (real - predicho)")
    ax2.set_title("Residuales vs. valores predichos")

    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    base = preparar_dataset(df)
    print(f"Dataset preparado: {len(base)} filas (unidad de analisis: {UNIDAD_ANALISIS})")

    # --- 1. Correlacion ---
    print("\n" + "=" * 78)
    print("Matriz de correlacion")
    _, pearson, spearman = calcular_correlaciones(base)
    print("\nPearson:")
    print(pearson.round(3))
    print("\nSpearman:")
    print(spearman.round(3))
    graficar_correlacion(pearson, "Correlacion de Pearson entre variables de mercado",
                          OUTPUT_DIR / "correlacion_pearson.png")
    graficar_correlacion(spearman, "Correlacion de Spearman entre variables de mercado",
                          OUTPUT_DIR / "correlacion_spearman.png")
    pearson.round(4).to_csv(OUTPUT_DIR / "correlacion_pearson.csv")
    spearman.round(4).to_csv(OUTPUT_DIR / "correlacion_spearman.csv")

    # --- 2. Regresiones simples (ciclo sobre MODELOS) ---
    resumen_simples = []
    for spec in MODELOS:
        print("\n" + "=" * 78)
        print(spec["titulo"])
        ajuste = ajustar_modelo_simple(base, spec)
        print(f"  y = {ajuste['pendiente']:.4f} x + {ajuste['intercepto']:.4f}")
        print(f"  R^2 = {ajuste['r2']:.4f}  |  r = {ajuste['r']:.4f}  |  p = {ajuste['p']:.2e}  |  n = {ajuste['n']}")
        graficar_modelo_simple(ajuste, OUTPUT_DIR / spec["archivo"])
        resumen_simples.append({
            "modelo": spec["titulo"], "x": spec["x"], "y": spec["y"],
            "pendiente": ajuste["pendiente"], "intercepto": ajuste["intercepto"],
            "r": ajuste["r"], "r2": ajuste["r2"], "p": ajuste["p"], "n": ajuste["n"],
        })
    pd.DataFrame(resumen_simples).to_csv(OUTPUT_DIR / "resumen_modelos_simples.csv", index=False)

    # --- 3. Regresion multiple ---
    print("\n" + "=" * 78)
    print(MODELO_MULTIPLE["titulo"])
    ajuste_m = ajustar_modelo_multiple(base, MODELO_MULTIPLE)
    print(f"  intercepto = {ajuste_m['intercepto']:.4f}")
    for nombre, coef in ajuste_m["coeficientes"].items():
        print(f"  coeficiente [{nombre}] = {coef:.4f}")
    print(f"  R^2 = {ajuste_m['r2']:.4f}  |  R^2 ajustado = {ajuste_m['r2_ajustado']:.4f}  |  n = {ajuste_m['n']}")
    graficar_modelo_multiple(ajuste_m, OUTPUT_DIR / f"{MODELO_MULTIPLE['archivo_base']}.png")

    pd.DataFrame([{"predictor": "intercepto", "coeficiente": ajuste_m["intercepto"]},
                  *[{"predictor": n, "coeficiente": c} for n, c in ajuste_m["coeficientes"].items()],
                  {"predictor": "R2", "coeficiente": ajuste_m["r2"]},
                  {"predictor": "R2_ajustado", "coeficiente": ajuste_m["r2_ajustado"]}]
                 ).to_csv(OUTPUT_DIR / "resumen_modelo_multiple.csv", index=False)

    print(f"\nResultados y graficas guardados en '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
