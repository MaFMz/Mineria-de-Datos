#!/usr/bin/env python3
"""
pronostico_gw2.py

Pronostico (forecasting) de series de tiempo del mercado de GW2: predice los
proximos H dias del precio de venta y del volumen vendido de items
representativos, con intervalos de prediccion y una evaluacion honesta.

IDEAS CLAVE

  * VALIDACION TEMPORAL, nunca aleatoria. En una serie de tiempo mezclar dias
    al azar entre entrenamiento y prueba deja al modelo "ver el futuro".
    Aqui se usa ORIGEN MOVIL (rolling origin): se entrena con todo hasta un
    dia de corte, se pronostican los H dias siguientes y se compara con lo
    que paso; luego el corte avanza H dias y se repite (N_ORIGENES veces).

  * CLASIFICACION POR LINEAS BASE. Los precios de un mercado se parecen a un
    "paseo aleatorio": el mejor pronostico de manana suele ser el valor de
    hoy. Por eso cada modelo se compara con modelos ingenuos:
        naive            -> repite el ultimo valor
        estacional_7     -> repite la semana anterior (hay ciclo semanal)
        media_movil_7    -> promedio de los ultimos 7 dias
    Un modelo "sofisticado" que no supera a estos no aporta nada.

  * MODELOS DE PRONOSTICO
        ridge_ar         -> REGRESION lineal regularizada (Ridge) sobre
                            rezagos de los cambios diarios, desviacion
                            respecto a la media de 28 dias y dia de la
                            semana; pronostica de forma recursiva.
        holt_winters     -> suavizado exponencial con tendencia amortiguada
                            y estacionalidad semanal (serie de tiempo
                            clasica). Opcional: requiere `statsmodels`.

  * ESCALA LOGARITMICA. Se modela log10 del precio (y log10(1 + ventas)):
    los precios varian en porcentaje, no en cobre fijo, y el log hace los
    errores comparables entre items baratos y caros. Un error medio (MAE) de
    0.04 en log10 equivale a equivocarse ~10% (10^0.04 = 1.10).

  * INTERVALOS. Se calculan con los errores reales del backtest del mejor
    modelo (por semana de horizonte), no con supuestos teoricos.

  * HUECOS. En precio, 0 significa "sin actividad": se trata como dato
    faltante y se rellena SOLO hacia adelante (con el ultimo valor conocido)
    para no usar informacion futura. Las metricas se calculan unicamente
    sobre dias con dato real observado.

LIMITES: el pronostico asume que el futuro se parece al pasado reciente. Los
parches, eventos y cambios de balance del juego rompen las series y NO son
predecibles con estos datos.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventanas: solo se guardan archivos (evita fallos de Tk en Windows)
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAY_STATSMODELS = True
except ImportError:
    HAY_STATSMODELS = False

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

CSV_PATH = "dataset.csv"
CLUSTERS_CSV = "items_con_cluster.csv"   # salida de kmeans_gw2.py (opcional)
OUTPUT_DIR = Path("pronostico_resultados")

H = 30                    # dias a pronosticar
N_ORIGENES = 12           # cortes del backtest (12 x 30 dias = ultimo ano)
MIN_ENTRENAMIENTO = 365   # dias minimos de historia para entrenar en cada corte
VENTANA = 730             # maximo de dias recientes usados para entrenar
NIVEL_INTERVALO = 0.80    # cobertura del intervalo de prediccion
LAGS = 14                 # rezagos de la regresion
RIDGE_ALPHA = 10.0

# Items a pronosticar. Vacio = eleccion automatica: un item representativo por
# cada grupo de K-Means (si existe CLUSTERS_CSV) o, si no, items repartidos
# por nivel de precio. Tambien puedes poner tus propios itemID: [19701, 19721]
ITEM_IDS = []
N_ITEMS_AUTO = 3
MAX_FALTANTES = 0.10      # un item es elegible si al menos 90% de sus dias tienen precio

FEATURES_CLUSTER = ["log_precio_venta", "log_volumen_venta", "log_spread",
                    "volatilidad_precio", "frac_dias_sin_ventas"]

# Variables a pronosticar
VARIABLES = [
    {"variable": "sell_price_avg", "cero_es_faltante": True, "transformacion": "log10",
     "etiqueta": "precio de venta (cobre)", "escala_y": "log"},
    {"variable": "sell_sold", "cero_es_faltante": False, "transformacion": "log1p10",
     "etiqueta": "unidades vendidas por dia", "escala_y": "linear"},
]

plt.rcParams["figure.dpi"] = 120


# ---------------------------------------------------------------------------
# TRANSFORMACIONES
# ---------------------------------------------------------------------------

def a_escala_modelo(s: pd.Series, transformacion: str) -> pd.Series:
    return np.log10(s) if transformacion == "log10" else np.log10(1 + s)


def a_escala_original(y, transformacion: str):
    return 10 ** np.asarray(y, dtype=float) if transformacion == "log10" else 10 ** np.asarray(y, dtype=float) - 1


# ---------------------------------------------------------------------------
# PREPARACION DE DATOS
# ---------------------------------------------------------------------------

def cargar(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normaliza a fecha sin zona horaria ni hora (la API entrega "2020-05-12T00:00:01.000Z")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df


def construir_serie(df_item: pd.DataFrame, spec: dict):
    """Devuelve (y, observado): y = serie diaria en escala del modelo con huecos
    rellenados solo hacia adelante; observado = True donde hubo dato real."""
    s = df_item.drop_duplicates("date").set_index("date")[spec["variable"]].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    if spec["cero_es_faltante"]:
        s = s.where(s > 0)
    observado = s.notna()
    y = a_escala_modelo(s, spec["transformacion"]).ffill()   # ffill: solo usa el pasado
    primero = y.first_valid_index()
    return y.loc[primero:], observado.loc[primero:]


def elegir_items(df: pd.DataFrame) -> list:
    """Elige items con serie completa: uno por grupo de K-Means, o repartidos por precio."""
    span = df.groupby("itemID")["date"].agg(lambda s: (s.max() - s.min()).days + 1)
    con_precio = df[df["sell_price_avg"] > 0].groupby("itemID").size()
    cobertura = (con_precio / span).reindex(span.index).fillna(0)
    elegibles = cobertura[(cobertura >= 1 - MAX_FALTANTES) &
                          (span >= MIN_ENTRENAMIENTO + H * N_ORIGENES)].index
    if len(elegibles) == 0:
        return []

    ruta = Path(CLUSTERS_CSV)
    if ruta.exists():
        c = pd.read_csv(ruta)
        cols = [f for f in FEATURES_CLUSTER if f in c.columns]
        c = c[c["itemID"].isin(elegibles)].dropna(subset=cols)
        if len(c) and cols:
            z = (c[cols] - c[cols].mean()) / c[cols].std().replace(0, 1)
            elegidos = []
            for _, idx in z.groupby(c["cluster"]).groups.items():
                sub = z.loc[idx]
                distancia = ((sub - sub.mean()) ** 2).sum(axis=1)
                elegidos.append(int(c.loc[distancia.idxmin(), "itemID"]))
            print(f"Items elegidos: el mas representativo de cada grupo de K-Means -> {elegidos}")
            return elegidos

    precio = df[df["itemID"].isin(elegibles) & (df["sell_price_avg"] > 0)] \
        .groupby("itemID")["sell_price_avg"].median().sort_values()
    posiciones = np.linspace(0.1, 0.9, N_ITEMS_AUTO) * (len(precio) - 1)
    elegidos = sorted({int(precio.index[int(round(p))]) for p in posiciones})
    print(f"Items elegidos: repartidos por nivel de precio -> {elegidos}")
    return elegidos


# ---------------------------------------------------------------------------
# MODELOS: cada uno recibe la serie de entrenamiento y las fechas futuras
# ---------------------------------------------------------------------------

def pron_naive(y: pd.Series, fechas_fut: pd.DatetimeIndex) -> np.ndarray:
    return np.repeat(y.iloc[-1], len(fechas_fut))


def pron_estacional_7(y: pd.Series, fechas_fut: pd.DatetimeIndex) -> np.ndarray:
    ultimos = y.iloc[-7:].to_numpy()
    return np.array([ultimos[i % 7] for i in range(len(fechas_fut))])


def pron_media_movil_7(y: pd.Series, fechas_fut: pd.DatetimeIndex) -> np.ndarray:
    return np.repeat(y.iloc[-7:].mean(), len(fechas_fut))


def _fila_features(y: np.ndarray, fechas: pd.DatetimeIndex, t: int) -> np.ndarray:
    """Variables para predecir el cambio diario en la posicion t usando solo y[:t]."""
    difs = (y[t - LAGS:t] - y[t - LAGS - 1:t - 1])[::-1]     # cambios de ayer, antier, ...
    desviacion = y[t - 1] - y[max(0, t - 28):t].mean()        # que tan lejos esta de su media de 28 dias
    dia_semana = np.zeros(7)
    dia_semana[fechas[t].dayofweek] = 1
    return np.concatenate([difs, [desviacion], dia_semana])


def pron_ridge_ar(y: pd.Series, fechas_fut: pd.DatetimeIndex) -> np.ndarray:
    yv = y.to_numpy(dtype=float)
    n = len(yv)
    fechas = y.index.append(fechas_fut)
    t0 = max(LAGS + 1, 28)
    X = np.array([_fila_features(yv, fechas, t) for t in range(t0, n)])
    objetivo = yv[t0:n] - yv[t0 - 1:n - 1]
    modelo = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(X, objetivo)

    historia = list(yv)
    for h in range(len(fechas_fut)):
        x = _fila_features(np.array(historia), fechas, n + h)
        historia.append(historia[-1] + modelo.predict(x.reshape(1, -1))[0])
    return np.array(historia[n:])


def pron_holt_winters(y: pd.Series, fechas_fut: pd.DatetimeIndex) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            y.to_numpy(dtype=float), trend="add", damped_trend=True,
            seasonal="add", seasonal_periods=7, initialization_method="estimated").fit()
        return np.asarray(modelo.forecast(len(fechas_fut)))


MODELOS = {
    "naive": pron_naive,
    "estacional_7": pron_estacional_7,
    "media_movil_7": pron_media_movil_7,
    "ridge_ar": pron_ridge_ar,
}
if HAY_STATSMODELS:
    MODELOS["holt_winters"] = pron_holt_winters


def pronosticar_seguro(nombre: str, y: pd.Series, fechas_fut: pd.DatetimeIndex, avisados: set):
    """Un modelo que falla no debe tumbar todo el script: se omite y se avisa una vez."""
    try:
        pred = MODELOS[nombre](y, fechas_fut)
        return pred if np.all(np.isfinite(pred)) else None
    except Exception as e:                                   # noqa: BLE001
        if nombre not in avisados:
            print(f"    Aviso: '{nombre}' fallo ({type(e).__name__}: {e}); se omite en estos datos.")
            avisados.add(nombre)
        return None


# ---------------------------------------------------------------------------
# BACKTEST (origen movil)
# ---------------------------------------------------------------------------

def backtest(y: pd.Series, observado: pd.Series):
    filas, errores = [], {m: [] for m in MODELOS}
    avisados = set()
    for k in range(N_ORIGENES, 0, -1):
        corte = len(y) - H * k
        if corte < MIN_ENTRENAMIENTO:
            continue
        entreno = y.iloc[max(0, corte - VENTANA):corte]
        prueba = y.iloc[corte:corte + H]
        obs = observado.loc[prueba.index].to_numpy()
        if len(prueba) < H or obs.sum() < 5:
            continue
        for nombre in MODELOS:
            pred = pronosticar_seguro(nombre, entreno, prueba.index, avisados)
            if pred is None:
                continue
            err = prueba.to_numpy() - pred
            filas.append({"origen": entreno.index[-1], "modelo": nombre,
                          "mae_log": np.abs(err[obs]).mean(),
                          "rmse_log": np.sqrt((err[obs] ** 2).mean()),
                          "sesgo_log": err[obs].mean()})
            errores[nombre].append(np.where(obs, err, np.nan))
    return pd.DataFrame(filas), {m: np.array(v) for m, v in errores.items() if len(v)}


def resumir_modelos(metricas: pd.DataFrame) -> pd.DataFrame:
    r = metricas.groupby("modelo").agg(
        origenes=("mae_log", "size"), mae_log=("mae_log", "mean"), mae_std=("mae_log", "std"),
        rmse_log=("rmse_log", "mean"), sesgo_log=("sesgo_log", "mean"))
    r["factor_error"] = 10 ** r["mae_log"]                      # error tipico como multiplicador (1.10 = ~10%)
    r["vs_naive"] = r["mae_log"] / r.loc["naive", "mae_log"]    # < 1 mejora al ingenuo
    return r.sort_values("mae_log")


def intervalos(errores_mejor: np.ndarray):
    """Cuantiles de los errores del backtest, por semana de horizonte (1-7, 8-14, ...)."""
    bajo, alto = (1 - NIVEL_INTERVALO) / 2, 1 - (1 - NIVEL_INTERVALO) / 2
    todos = errores_mejor[np.isfinite(errores_mejor)]
    lo, hi = np.zeros(H), np.zeros(H)
    for b in range((H + 6) // 7):
        cols = np.arange(7 * b, min(7 * b + 7, H))
        v = errores_mejor[:, cols].ravel()
        v = v[np.isfinite(v)]
        v = v if len(v) >= 20 else todos
        lo[cols], hi[cols] = np.quantile(v, bajo), np.quantile(v, alto)
    return lo, hi


# ---------------------------------------------------------------------------
# GRAFICAS
# ---------------------------------------------------------------------------

def graficar(y, observado, fut, resumen, mejor, spec, titulo, ruta: Path):
    tr = spec["transformacion"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.7, 1]})

    hist = y.iloc[-365:]
    obs = observado.loc[hist.index]
    ax1.plot(hist.index[obs], a_escala_original(hist[obs], tr), lw=0.9, color="tab:blue", label="observado")
    ax1.plot(fut["fecha"], fut["pronostico"], color="firebrick", lw=2, label=f"pronostico ({mejor})")
    ax1.fill_between(fut["fecha"], fut["inferior"], fut["superior"], color="firebrick", alpha=0.22,
                     label=f"intervalo {NIVEL_INTERVALO:.0%}")
    ax1.axvline(y.index[-1], color="gray", linestyle=":", linewidth=1)
    ax1.set_yscale(spec["escala_y"])
    ax1.set_ylabel(spec["etiqueta"])
    ax1.set_title(f"{titulo}: ultimo ano y proximos {H} dias")
    ax1.legend(loc="best", fontsize=8)
    ax1.tick_params(axis="x", rotation=30)

    colores = ["firebrick" if m == mejor else "steelblue" for m in resumen.index]
    ax2.bar(resumen.index, resumen["mae_log"], yerr=resumen["mae_std"].fillna(0), capsize=3, color=colores)
    for i, (m, fila) in enumerate(resumen.iterrows()):
        ax2.text(i, fila["mae_log"], f"x{fila['vs_naive']:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel(f"MAE en log10 (promedio de {int(resumen['origenes'].max())} cortes)")
    ax2.set_title("Error del backtest por modelo\n(x = proporcion del error del naive; menor es mejor)", fontsize=10)
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def procesar(df_item: pd.DataFrame, item_id: int, nombre_item: str, spec: dict):
    y, observado = construir_serie(df_item, spec)
    faltantes = 1 - observado.mean()
    print(f"\n{'=' * 78}\n{nombre_item} (itemID {item_id}) | {spec['variable']} | "
          f"{len(y)} dias, {faltantes:.1%} sin dato real")
    if len(y) < MIN_ENTRENAMIENTO + H * 3:
        print("  Serie demasiado corta para el backtest; se omite.")
        return None

    metricas, errores = backtest(y, observado)
    if metricas.empty or "naive" not in set(metricas["modelo"]) or metricas["origen"].nunique() < 3:
        print("  No hubo suficientes cortes validos; se omite.")
        return None

    resumen = resumir_modelos(metricas)
    print(f"  Backtest: {metricas['origen'].nunique()} cortes de {H} dias")
    print(resumen.round(4).to_string())
    mejor = resumen.index[0]
    print(f"  -> Mejor modelo: {mejor} (error tipico x{resumen.loc[mejor, 'factor_error']:.3f}; "
          f"{'SUPERA' if resumen.loc[mejor, 'vs_naive'] < 1 else 'no supera'} al naive: "
          f"{resumen.loc[mejor, 'vs_naive']:.2f} veces su error)")

    # Pronostico futuro con el mejor modelo entrenado con toda la historia reciente
    fechas_fut = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    pred = pronosticar_seguro(mejor, y.iloc[-VENTANA:], fechas_fut, set())
    if pred is None:
        print("  El mejor modelo fallo al ajustar con toda la historia; se omite.")
        return None
    lo, hi = intervalos(errores[mejor])
    tr = spec["transformacion"]
    fut = pd.DataFrame({
        "itemID": item_id, "variable": spec["variable"], "modelo": mejor, "fecha": fechas_fut,
        "pronostico": a_escala_original(pred, tr),
        "inferior": a_escala_original(pred + lo, tr),
        "superior": a_escala_original(pred + hi, tr)})

    graficar(y, observado, fut, resumen, mejor, spec, nombre_item,
             OUTPUT_DIR / f"pronostico_{item_id}_{spec['variable']}.png")

    detalle = metricas.assign(itemID=item_id, variable=spec["variable"])
    resumen_fila = resumen.reset_index().assign(itemID=item_id, variable=spec["variable"], mejor=lambda d: d["modelo"] == mejor)
    return detalle, resumen_fila, fut


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = cargar(CSV_PATH)
    print(f"Modelos: {list(MODELOS)}" + ("" if HAY_STATSMODELS else
          "  (holt_winters desactivado: instala 'statsmodels' para incluirlo)"))
    print(f"Datos hasta {df['date'].max().date()}; se pronostican los {H} dias siguientes.")

    ids = ITEM_IDS or elegir_items(df)
    if not ids:
        print("Ningun item cumple los requisitos de longitud y cobertura.")
        return

    nombres = df.groupby("itemID")["name"].first().to_dict() if "name" in df.columns else {}
    detalles, resumenes, futuros = [], [], []
    for item_id in ids:
        df_item = df[df["itemID"] == item_id]
        nombre = nombres.get(item_id, f"item {item_id}")
        for spec in VARIABLES:
            salida = procesar(df_item, item_id, str(nombre), spec)
            if salida:
                detalles.append(salida[0]); resumenes.append(salida[1]); futuros.append(salida[2])

    if futuros:
        pd.concat(detalles).round(5).to_csv(OUTPUT_DIR / "backtest_detalle.csv", index=False)
        pd.concat(resumenes).round(5).to_csv(OUTPUT_DIR / "resumen_modelos.csv", index=False)
        pd.concat(futuros).round(3).to_csv(OUTPUT_DIR / "pronostico_futuro.csv", index=False)
    print(f"\nResultados guardados en '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
