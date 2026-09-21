"""
graficas_gw2.py

Genera automaticamente, mediante un ciclo sobre una lista de
especificaciones, los 5 tipos de graficas pedidas para el dataset de
mercado de GW2:

    1. Linea      -> date vs buy_price_avg / sell_price_avg
    2. Dispersion -> buy_price_avg vs sell_price_avg
    3. Barras     -> type vs buy_sold / sell_sold
    4. Boxplot    -> rarity vs sell_price_avg
    5. Dispersion -> level vs sell_price_avg

Cada tipo de grafica tiene su propia funcion de dibujo; el ciclo
principal (GRAFICAS) solo decide, por cada especificacion, que funcion
llamar y con que datos. Agregar una sexta grafica es solo
anadir un diccionario mas a la lista GRAFICAS.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

CSV_PATH = "dataset.csv"
OUTPUT_DIR = Path("graficas_gw2")

# Si se deja en None, la grafica de linea promedia buy/sell_price_avg de
# TODOS los items por fecha. Si se pone un itemID (ej. 19701), la linea
# muestra el historico de ese item en particular.
ITEM_ID_FILTER = None

# Para las dispersiones, tomar una muestra si el dataset es muy grande
# (140 items x ~5 anos de datos diarios puede ser >200,000 puntos, lo
# cual hace el scatter lento e ilegible). None = usar todos los puntos.
SCATTER_SAMPLE_SIZE = None

plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["figure.dpi"] = 120


# ---------------------------------------------------------------------------
# LIMPIEZA COMUN
# ---------------------------------------------------------------------------

def limpiar_datos(df: pd.DataFrame, columnas_dropna: list, columnas_cero: list = None) -> pd.DataFrame:
    """Descarta filas con NaN en `columnas_dropna` (todas las columnas que
    va a usar la grafica). Ademas, SOLO en las columnas listadas en
    `columnas_cero` (tipicamente columnas de precio), descarta tambien las
    filas en 0: un precio en 0 significa "sin actividad de mercado ese
    dia", pero un 0 en una columna categorica/numerica como `level` es un
    valor real y no debe descartarse."""
    df_limpio = df.dropna(subset=columnas_dropna)
    for col in columnas_cero or []:
        df_limpio = df_limpio[df_limpio[col] != 0]
    return df_limpio


# ---------------------------------------------------------------------------
# UNA FUNCION DE DIBUJO POR TIPO DE GRAFICA
# ---------------------------------------------------------------------------

def grafica_lineas(df: pd.DataFrame, spec: dict):
    x, ys = spec["x"], spec["y"]
    datos = limpiar_datos(df, columnas_dropna=ys, columnas_cero=ys)

    if ITEM_ID_FILTER is not None:
        datos = datos[datos["itemID"] == ITEM_ID_FILTER]
        subtitulo = f"item {ITEM_ID_FILTER}"
    else:
        datos = datos.groupby(x)[ys].mean().reset_index()
        subtitulo = "promedio de todos los items"

    datos = datos.sort_values(x)

    fig, ax = plt.subplots()
    for columna in ys:
        ax.plot(datos[x], datos[columna], label=columna, linewidth=1)
    ax.set_xlabel(x)
    ax.set_ylabel("precio (cobre)")
    ax.set_title(f"{spec['titulo']} ({subtitulo})")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def grafica_dispersion(df: pd.DataFrame, spec: dict):
    x, y = spec["x"], spec["y"]
    
    columnas_cero = spec.get("cero", [y])
    datos = limpiar_datos(df, columnas_dropna=[x, y], columnas_cero=columnas_cero)

    if SCATTER_SAMPLE_SIZE and len(datos) > SCATTER_SAMPLE_SIZE:
        datos = datos.sample(n=SCATTER_SAMPLE_SIZE, random_state=42)

    fig, ax = plt.subplots()
    ax.scatter(datos[x], datos[y], alpha=0.4, s=12)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(spec["titulo"])
    return fig


def grafica_barras(df: pd.DataFrame, spec: dict):
    categoria, ys = spec["x"], spec["y"]
    datos = df.dropna(subset=[categoria]).groupby(categoria)[ys].sum()
    datos = datos.sort_values(ys[0], ascending=False)

    fig, ax = plt.subplots()
    datos.plot(kind="bar", ax=ax)
    ax.set_xlabel(categoria)
    ax.set_ylabel("cantidad total transada")
    ax.set_title(spec["titulo"])
    ax.legend(title=None)
    fig.tight_layout()
    return fig


def grafica_boxplot(df: pd.DataFrame, spec: dict):
    categoria, y = spec["x"], spec["y"]
    datos = limpiar_datos(df, columnas_dropna=[categoria, y], columnas_cero=[y])

    grupos = datos.groupby(categoria)[y]
    etiquetas = list(grupos.groups.keys())
    valores = [grupos.get_group(g).values for g in etiquetas]

    fig, ax = plt.subplots()
    ax.boxplot(valores, tick_labels=etiquetas, showfliers=False)
    ax.set_xlabel(categoria)
    ax.set_ylabel(y)
    ax.set_title(spec["titulo"])
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


DISPATCH = {
    "linea": grafica_lineas,
    "dispersion": grafica_dispersion,
    "barras": grafica_barras,
    "boxplot": grafica_boxplot,
}


# ---------------------------------------------------------------------------
# ESPECIFICACION DE LAS 5 GRAFICAS PEDIDAS
# ---------------------------------------------------------------------------

GRAFICAS = [
    {
        "archivo": "01_lineas_precio_promedio.png",
        "tipo": "linea",
        "x": "date",
        "y": ["buy_price_avg", "sell_price_avg"],
        "titulo": "Precio promedio de compra/venta a lo largo del tiempo",
    },
    {
        "archivo": "02_dispersion_buy_vs_sell.png",
        "tipo": "dispersion",
        "x": "buy_price_avg",
        "y": "sell_price_avg",
        "cero": ["buy_price_avg", "sell_price_avg"],
        "titulo": "Relacion entre precio de compra y precio de venta",
    },
    {
        "archivo": "03_barras_sold_por_tipo.png",
        "tipo": "barras",
        "x": "type",
        "y": ["buy_sold", "sell_sold"],
        "titulo": "Unidades compradas y vendidas por tipo de item",
    },
    {
        "archivo": "04_boxplot_precio_por_rareza.png",
        "tipo": "boxplot",
        "x": "rarity",
        "y": "sell_price_avg",
        "titulo": "Distribucion del precio de venta por rareza",
    },
    {
        "archivo": "05_dispersion_nivel_vs_precio.png",
        "tipo": "dispersion",
        "x": "level",
        "y": "sell_price_avg",
        "cero": ["sell_price_avg"],
        "titulo": "Relacion entre nivel requerido y precio de venta",
    },
]


# ---------------------------------------------------------------------------
# MAIN: ciclo de automatizacion sobre todas las graficas
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    OUTPUT_DIR.mkdir(exist_ok=True)

    for spec in GRAFICAS:
        funcion = DISPATCH[spec["tipo"]]
        print(f"Generando [{spec['tipo']}] -> {spec['archivo']} ...")
        fig = funcion(df, spec)
        ruta = OUTPUT_DIR / spec["archivo"]
        fig.savefig(ruta, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardada en {ruta}")

    print(f"\n{len(GRAFICAS)} graficas generadas en '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
