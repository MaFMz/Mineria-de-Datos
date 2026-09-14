import pandas as pd
import numpy as np
from scipy import stats

CSV_PATH = "dataset.csv"

ITEM_COLUMNS = ["itemID", "name", "description", "type", "rarity", "level",
                "vendor_value", "flags"]

MERCADO_COLUMNS = [
    "itemID", "date", "count",
    "buy_delisted", "buy_listed", "buy_sold", "buy_value",
    "buy_price_avg", "buy_price_max", "buy_price_min", "buy_price_stdev",
    "buy_quantity_avg", "buy_quantity_max", "buy_quantity_min", "buy_quantity_stdev",
    "sell_delisted", "sell_listed", "sell_sold", "sell_value",
    "sell_price_avg", "sell_price_max", "sell_price_min", "sell_price_stdev",
    "sell_quantity_avg", "sell_quantity_max", "sell_quantity_min", "sell_quantity_stdev",
]

METRICAS = ["buy_price_avg", "sell_price_avg", "buy_quantity_avg", "sell_quantity_avg"]


def cargar_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def construir_entidades(df: pd.DataFrame):
    item_df = df[ITEM_COLUMNS].drop_duplicates(subset="itemID").reset_index(drop=True)

    fecha_df = df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)

    mercado_df = df[MERCADO_COLUMNS].copy()

    return item_df, fecha_df, mercado_df


def verificar_cardinalidad(item_df, fecha_df, mercado_df):
    items_referenciados = mercado_df["itemID"].nunique()
    fechas_referenciadas = mercado_df["date"].nunique()
    print(f"ITEM: {len(item_df)} entidades unicas")
    print(f"FECHA: {len(fecha_df)} fechas unicas")
    print(f"MERCADO_DIARIO: {len(mercado_df)} registros (hechos)")
    print(f"  -> ITEM (1) --- (N) MERCADO_DIARIO: {items_referenciados} items con historico")
    print(f"  -> FECHA (1) --- (N) MERCADO_DIARIO: {fechas_referenciadas} fechas con registros")


def ejemplos_algebra_relacional(item_df: pd.DataFrame, mercado_df: pd.DataFrame):
    resultados = {}
    
    resultados["seleccion_por_rareza"] = item_df[item_df["rarity"] == "Exotic"]

    resultados["proyeccion_precios"] = mercado_df[["itemID", "date", "buy_price_avg", "sell_price_avg"]]

    resultados["join_completo"] = mercado_df.merge(item_df, on="itemID", how="inner")

    armas = item_df[item_df["type"] == "Weapon"]
    consumibles = item_df[item_df["type"] == "Consumable"]
    resultados["union_tipos"] = pd.concat([armas, consumibles]).drop_duplicates()

    mercado_con_tipo = mercado_df.merge(item_df[["itemID", "type"]], on="itemID", how="left")
    resultados["mercado_con_tipo"] = mercado_con_tipo

    return resultados


def resumen_estadistico(serie: pd.Series) -> pd.Series:
    datos = serie.dropna()
    datos = datos[datos != 0]

    if datos.empty:
        return pd.Series({
            "conteo": 0, "min": np.nan, "max": np.nan, "moda": np.nan,
            "sumatoria": np.nan, "media": np.nan, "varianza": np.nan,
            "desv_estandar": np.nan, "asimetria": np.nan, "kurtosis": np.nan,
        })

    moda_resultado = stats.mode(datos, keepdims=True)
    moda = moda_resultado.mode[0] if len(moda_resultado.mode) > 0 else np.nan
    asimetria = round(stats.skew(datos), 3) if len(datos) >= 2 else np.nan
    kurtosis = round(stats.kurtosis(datos), 3) if len(datos) >= 4 else np.nan

    return pd.Series({
        "conteo": datos.count(),
        "min": datos.min(),
        "max": datos.max(),
        "moda": moda,
        "sumatoria": datos.sum(),
        "media": round(datos.mean(), 2),
        "varianza": round(datos.var(), 2),
        "desv_estandar": round(datos.std(), 2),
        "asimetria": asimetria,
        "kurtosis": kurtosis,
    })


def estadisticas_agrupadas(df_hechos: pd.DataFrame, columna_grupo: str, columnas_metricas: list) -> pd.DataFrame:
    resultados = {}
    for columna in columnas_metricas:
        resultados[columna] = df_hechos.groupby(columna_grupo)[columna].apply(resumen_estadistico).unstack()
    return pd.concat(resultados, axis=1)


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df = cargar_dataset(CSV_PATH)
    item_df, fecha_df, mercado_df = construir_entidades(df)

    print("=== Entidades del modelo E-R ===")
    verificar_cardinalidad(item_df, fecha_df, mercado_df)

    print("\n=== Ejemplos de algebra relacional ===")
    ejemplos = ejemplos_algebra_relacional(item_df, mercado_df)
    for nombre, resultado in ejemplos.items():
        print(f"\n-- {nombre} ({len(resultado)} filas) --")
        print(resultado.head(3))

    mercado_con_tipo = ejemplos["mercado_con_tipo"]

    stats_por_tipo = estadisticas_agrupadas(mercado_con_tipo, "type", METRICAS)

    stats_por_item = estadisticas_agrupadas(mercado_df, "itemID", METRICAS)

    stats_por_tipo_transpuesta = stats_por_tipo.T

    print("\n=== Estadistica descriptiva agrupada por TIPO de item ===")
    print(stats_por_tipo.round(2))

    print("\n=== Estadistica descriptiva agrupada por TIPO (transpuesta) ===")
    print(stats_por_tipo_transpuesta.round(2))

    print("\n=== Estadistica descriptiva agrupada por ITEM (primeras filas) ===")
    print(stats_por_item.round(2).head())

    stats_por_tipo.to_csv("stats_por_tipo.csv")
    stats_por_item.to_csv("stats_por_item.csv")
    print("\nArchivos generados: stats_por_tipo.csv, stats_por_item.csv")


if __name__ == "__main__":
    main()
