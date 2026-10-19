#!/usr/bin/env python3
"""
knn_gw2.py

Modelo de clasificacion con K Vecinos Mas Cercanos (KNN) sobre el dataset
de mercado de GW2.

PREGUNTA: dado el comportamiento de mercado de un item (precio, volumen,
margen, volatilidad), ¿se puede predecir a que categoria pertenece
(rareza, tipo o nivel)? Es la continuacion natural de las pruebas
estadisticas: ahi se vio que esas categorias difieren en precio y volumen;
aqui se mide cuanto de esa diferencia basta para CLASIFICAR.

Para cada objetivo en OBJETIVOS (ciclo, mismo patron que los scripts
anteriores):

  1. Construye una fila por item con variables de mercado derivadas.
  2. Separa un conjunto de prueba estratificado (nunca se toca al elegir
     hiperparametros).
  3. Busca los mejores hiperparametros (k, ponderacion, distancia) con
     validacion cruzada estratificada SOLO sobre el conjunto de
     entrenamiento. El escalado (StandardScaler) va dentro del Pipeline,
     asi que se ajusta en cada pliegue y no hay fuga de datos. El escalado
     es obligatorio en KNN: sin el, la variable de mayor magnitud domina
     la distancia.
  4. Evalua en el conjunto de prueba: accuracy, accuracy balanceada,
     F1 macro, reporte por clase y matriz de confusion.
  5. Como con ~140 items un unico conjunto de prueba es ruidoso, repite la
     evaluacion con VALIDACION CRUZADA ANIDADA (la busqueda de
     hiperparametros se rehace dentro de cada pliegue externo) y la
     compara contra un clasificador base que siempre predice la clase
     mayoritaria. Si KNN no supera ese base, el modelo no aporta nada.

Metrica principal: accuracy balanceada, porque las clases estan
desbalanceadas y la accuracy simple premia predecir siempre la clase
grande.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             balanced_accuracy_score, classification_report, f1_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

CSV_PATH = "dataset.csv"
OUTPUT_DIR = Path("knn_resultados")

RANDOM_STATE = 42
TEST_SIZE = 0.25        # fraccion de items reservada para la prueba final
MIN_CLASE = 5           # clases con menos items se excluyen (no se pueden validar)
CV_SPLITS = 5           # pliegues de la validacion cruzada
CV_REPEATS = 3          # repeticiones de la validacion cruzada anidada
K_MAX = 15              # se prueban k = 1, 3, 5, ... hasta K_MAX

# Variables de mercado calculadas por item. Se evita meter a la vez
# buy_price y sell_price (correlacion ~0.999) o buy_sold y sell_sold
# (~0.96): serian la misma informacion contada dos veces en la distancia.
FEATURES = [
    "log_precio_venta",       # mediana de log10(sell_price_avg)
    "log_volumen_venta",      # log10(1 + mediana de sell_sold); el +1 conserva items con mediana 0
    "log_spread",             # mediana de log10(sell_price_avg / buy_price_avg): margen compra-venta
    "volatilidad_precio",     # desviacion estandar de log10(sell_price_avg) en el tiempo
    "frac_dias_sin_ventas",   # proporcion de dias sin ninguna venta (0 a 1)
]

# Cada objetivo es una etiqueta de item a predecir. level es 0 u 80 en
# la practica, asi que se trata como clase ("nivel_0", "nivel_80").
OBJETIVOS = [
    {"objetivo": "rarity", "features": FEATURES},
    {"objetivo": "type", "features": FEATURES},
    {"objetivo": "level", "features": FEATURES},
]

plt.rcParams["figure.dpi"] = 120


# ---------------------------------------------------------------------------
# PREPARACION: una fila por item
# ---------------------------------------------------------------------------

def construir_dataset_items(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["sell_price_avg"] > 0].copy()    # 0 = sin actividad, no un precio real
    d["log_precio_venta"] = np.log10(d["sell_price_avg"])
    if "buy_price_avg" in d.columns:
        d["log_spread"] = np.log10(d["sell_price_avg"] / d["buy_price_avg"].where(d["buy_price_avg"] > 0))
    else:
        d["log_spread"] = np.nan
    d["sin_ventas"] = (d["sell_sold"] == 0).astype(float)

    g = d.groupby("itemID")
    items = pd.DataFrame({
        "log_precio_venta": g["log_precio_venta"].median(),
        "log_volumen_venta": np.log10(1 + g["sell_sold"].median()),
        "log_spread": g["log_spread"].median(),
        "volatilidad_precio": g["log_precio_venta"].std(),
        "frac_dias_sin_ventas": g["sin_ventas"].mean(),
    })

    etiquetas = df.groupby("itemID")[[c for c in ["name", "type", "rarity", "level"] if c in df.columns]].first()
    if "level" in etiquetas.columns:
        etiquetas["level"] = etiquetas["level"].map(lambda v: f"nivel_{int(v)}" if pd.notna(v) else np.nan)
    return items.join(etiquetas).reset_index()


# ---------------------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------------------

def crear_pipeline() -> Pipeline:
    return Pipeline([("escala", StandardScaler()), ("knn", KNeighborsClassifier())])


def crear_busqueda(n_splits: int, k_max: int) -> GridSearchCV:
    rejilla = {
        "knn__n_neighbors": list(range(1, k_max + 1, 2)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }
    return GridSearchCV(
        crear_pipeline(), rejilla,
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE),
        scoring="balanced_accuracy", refit=True,
    )


def preparar_objetivo(items: pd.DataFrame, spec: dict):
    """Devuelve (X, y, clases_excluidas) o None si no se puede modelar."""
    obj, feats = spec["objetivo"], spec["features"]
    if obj not in items.columns:
        print(f"  Se omite: no existe la columna '{obj}'.")
        return None
    datos = items.dropna(subset=[obj] + feats)
    conteos = datos[obj].value_counts()
    excluidas = [f"{c} (n={n})" for c, n in conteos.items() if n < MIN_CLASE]
    datos = datos[datos[obj].isin(conteos[conteos >= MIN_CLASE].index)]
    if datos[obj].nunique() < 2:
        print("  Se omite: menos de 2 clases con suficientes items.")
        return None
    return datos[feats], datos[obj], excluidas


# ---------------------------------------------------------------------------
# EVALUACION
# ---------------------------------------------------------------------------

def graficar_resultados(busqueda: GridSearchCV, y_test, y_pred, clases, spec, ruta: Path):
    res = pd.DataFrame(busqueda.cv_results_)
    mejor = busqueda.best_params_
    # Curva de k, fijando la distancia elegida (una linea por tipo de ponderacion)
    sub = res[res["param_knn__metric"] == mejor["knn__metric"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for peso, grupo in sub.groupby("param_knn__weights"):
        grupo = grupo.sort_values("param_knn__n_neighbors")
        ax1.errorbar(grupo["param_knn__n_neighbors"], grupo["mean_test_score"],
                     yerr=grupo["std_test_score"], marker="o", capsize=3, label=f"weights = {peso}")
    ax1.axvline(mejor["knn__n_neighbors"], color="firebrick", linestyle="--", linewidth=1,
                label=f"mejor k = {mejor['knn__n_neighbors']}")
    ax1.set_xlabel("k (numero de vecinos)")
    ax1.set_ylabel("accuracy balanceada (validacion cruzada)")
    ax1.set_title(f"Eleccion de k para '{spec['objetivo']}' (distancia {mejor['knn__metric']})")
    ax1.legend()

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=clases, ax=ax2,
                                            xticks_rotation=45, colorbar=False, cmap="Blues")
    ax2.set_title("Matriz de confusion (conjunto de prueba)")

    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def clasificar_fuera_de_muestra(items: pd.DataFrame, X, y, spec: dict,
                                k_splits_ext: int, n_splits_int: int, k_max_int: int) -> pd.DataFrame:
    """Predice CADA item con un modelo que no lo vio al entrenar (validacion
    cruzada anidada, una pasada). Se usa en vez del conjunto de prueba porque
    este ultimo solo cubre ~25% de los items; asi se revisan los 140.
    Devuelve la tabla completa (aciertos y errores) de este objetivo."""
    cv = StratifiedKFold(k_splits_ext, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(crear_busqueda(n_splits_int, k_max_int), X, y, cv=cv)

    columnas_id = [c for c in ["itemID", "name"] if c in items.columns]
    tabla = items.loc[X.index, columnas_id].copy()
    tabla.insert(0, "objetivo", spec["objetivo"])
    tabla["clase_real"] = y.to_numpy()
    tabla["clase_predicha"] = pred
    tabla["acierto"] = tabla["clase_real"] == tabla["clase_predicha"]
    for col in ["log_precio_venta", "log_volumen_venta"]:   # contexto para explicar el error
        tabla[col] = X[col].round(3).to_numpy()
    return tabla


def evaluar_objetivo(items: pd.DataFrame, spec: dict, tablas_errores: list) -> dict | None:
    preparado = preparar_objetivo(items, spec)
    if preparado is None:
        return None
    X, y, excluidas = preparado
    obj = spec["objetivo"]
    conteo_min = int(y.value_counts().min())
    print(f"  {len(X)} items, {y.nunique()} clases. Distribucion: {y.value_counts().to_dict()}")
    if excluidas:
        print(f"  Clases excluidas por n < {MIN_CLASE}: {', '.join(excluidas)}")

    # --- Hold-out estratificado ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    n_splits = max(2, min(CV_SPLITS, int(y_tr.value_counts().min())))
    k_max = min(K_MAX, int(len(X_tr) * (n_splits - 1) / n_splits) - 1)

    busqueda = crear_busqueda(n_splits, k_max).fit(X_tr, y_tr)
    y_pred = busqueda.predict(X_te)
    mejor = {k.replace("knn__", ""): v for k, v in busqueda.best_params_.items()}
    print(f"  Mejores hiperparametros: {mejor} (balanced acc. CV = {busqueda.best_score_:.3f})")

    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    y_base = base.predict(X_te)

    acc, bacc = accuracy_score(y_te, y_pred), balanced_accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"  CONJUNTO DE PRUEBA (n = {len(y_te)}): accuracy = {acc:.3f} | "
          f"accuracy balanceada = {bacc:.3f} | F1 macro = {f1:.3f}")
    print(f"    Clasificador base (clase mayoritaria): accuracy = {accuracy_score(y_te, y_base):.3f} | "
          f"accuracy balanceada = {balanced_accuracy_score(y_te, y_base):.3f}")
    print(classification_report(y_te, y_pred, zero_division=0))

    # --- Validacion cruzada anidada (estimacion mas estable) ---
    k_splits_ext = max(2, min(CV_SPLITS, conteo_min))
    n_splits_int = max(2, min(CV_SPLITS, conteo_min * (k_splits_ext - 1) // k_splits_ext))
    k_max_int = min(K_MAX, int(len(X) * (k_splits_ext - 1) / k_splits_ext * (n_splits_int - 1) / n_splits_int) - 1)
    externo = RepeatedStratifiedKFold(n_splits=k_splits_ext, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    metricas = ["accuracy", "balanced_accuracy", "f1_macro"]
    knn_cv = cross_validate(crear_busqueda(n_splits_int, k_max_int), X, y, cv=externo, scoring=metricas)
    base_cv = cross_validate(DummyClassifier(strategy="most_frequent"), X, y, cv=externo, scoring=metricas)

    def resumen(cv, m):
        v = cv[f"test_{m}"]
        return v.mean(), v.std()

    print(f"  VALIDACION CRUZADA ANIDADA ({k_splits_ext} pliegues x {CV_REPEATS} repeticiones):")
    for m in metricas:
        (mk, sk), (mb, sb) = resumen(knn_cv, m), resumen(base_cv, m)
        print(f"    {m:18s} KNN = {mk:.3f} ± {sk:.3f}   |   base = {mb:.3f} ± {sb:.3f}")

    # --- Items mal clasificados (todos los items, predichos sin haberlos visto) ---
    tabla = clasificar_fuera_de_muestra(items, X, y, spec, k_splits_ext, n_splits_int, k_max_int)
    tablas_errores.append(tabla)
    errores = tabla[~tabla["acierto"]]
    print(f"  ITEMS MAL CLASIFICADOS (prediccion fuera de muestra): {len(errores)} de {len(tabla)}")
    recall = tabla.groupby("clase_real")["acierto"].agg(["sum", "count"])
    recall["recall"] = (recall["sum"] / recall["count"]).round(3)
    print("    Recall por clase:  " + " | ".join(
        f"{c}: {int(r['sum'])}/{int(r['count'])} ({r['recall']:.2f})" for c, r in recall.iterrows()))
    pares = errores.groupby(["clase_real", "clase_predicha"]).size().sort_values(ascending=False)
    if len(pares):
        print("    Confusiones mas frecuentes (real -> predicha): " + "; ".join(
            f"{a} -> {b} ({n})" for (a, b), n in pares.head(5).items()))

    graficar_resultados(busqueda, y_te, y_pred, sorted(y.unique()), spec, OUTPUT_DIR / f"knn_{obj}.png")

    fila = {"objetivo": obj, "n_items": len(X), "n_clases": y.nunique(),
            "k": mejor["n_neighbors"], "weights": mejor["weights"], "metric": mejor["metric"],
            "test_accuracy": acc, "test_balanced_accuracy": bacc, "test_f1_macro": f1,
            "test_base_balanced_accuracy": balanced_accuracy_score(y_te, y_base)}
    for m in metricas:
        fila[f"cv_{m}"], fila[f"cv_{m}_std"] = resumen(knn_cv, m)
        fila[f"cv_base_{m}"], _ = resumen(base_cv, m)
    return fila


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    items = construir_dataset_items(df)
    print(f"Dataset por item: {len(items)} filas, variables: {FEATURES}")
    items.to_csv(OUTPUT_DIR / "dataset_items_knn.csv", index=False)

    resumen, tablas_errores = [], []
    for spec in OBJETIVOS:
        print("\n" + "=" * 78)
        print(f"Clasificacion de '{spec['objetivo']}' con KNN")
        fila = evaluar_objetivo(items, spec, tablas_errores)
        if fila:
            resumen.append(fila)

    if resumen:
        pd.DataFrame(resumen).round(4).to_csv(OUTPUT_DIR / "resumen_knn.csv", index=False)
    if tablas_errores:
        todas = pd.concat(tablas_errores, ignore_index=True)
        todas.to_csv(OUTPUT_DIR / "predicciones_knn_todas.csv", index=False)
        todas[~todas["acierto"]].drop(columns="acierto").to_csv(
            OUTPUT_DIR / "items_mal_clasificados_knn.csv", index=False)
        print(f"\nItems mal clasificados: {(~todas['acierto']).sum()} errores en total "
              f"-> '{OUTPUT_DIR}/items_mal_clasificados_knn.csv'")
    print(f"\nResultados guardados en '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
