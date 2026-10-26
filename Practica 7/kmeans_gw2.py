#!/usr/bin/env python3
"""
kmeans_gw2.py

Agrupamiento (clustering) con K-Means sobre el dataset de mercado de GW2.

PREGUNTA: si se ignoran por completo las etiquetas (rarity, type, level),
¿los items se agrupan solos segun su comportamiento de mercado? Y si lo
hacen, ¿esos grupos coinciden con las categorias del juego?

A diferencia del KNN (supervisado), aqui el modelo NUNCA ve las etiquetas.
Se usan despues, solo para validar los grupos desde fuera.

Flujo:

  1. Una fila por item con las mismas variables de mercado que el KNN
     (precio, volumen, margen, volatilidad, dias sin ventas).
  2. Estandarizar (StandardScaler). Obligatorio: K-Means usa distancia
     euclidiana y sin escalar domina la variable de mayor magnitud.
  3. ELEGIR k probando k = 2..K_MAX con cuatro criterios:
        - Inercia (codo): suma de distancias al centroide; siempre baja
          al subir k, se busca el "codo".
        - Silueta (mayor es mejor, de -1 a 1): que tan separado esta cada
          item de los otros grupos frente al suyo.
        - Calinski-Harabasz (mayor es mejor): separacion entre grupos
          sobre dispersion interna.
        - Davies-Bouldin (menor es mejor): similitud media entre cada
          grupo y su vecino mas parecido.
     Por defecto se toma el k de mayor silueta; se puede forzar con
     K_ELEGIDO. Ningun criterio es definitivo: revisa la tabla completa.
  4. EVALUAR el modelo final: silueta global y por grupo, estabilidad
     (re-ajustes sobre submuestras del 80% comparados con ARI) y
     validacion externa contra rarity / type / level (ARI, informacion
     mutua normalizada, homogeneidad) mas tablas cruzadas.
  5. INTERPRETAR: perfil de cada grupo en unidades reales y proyeccion PCA.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventanas: solo se guardan archivos (evita fallos de Tk en Windows)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             completeness_score, davies_bouldin_score, homogeneity_score,
                             normalized_mutual_info_score, silhouette_samples, silhouette_score)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

CSV_PATH = "dataset.csv"
OUTPUT_DIR = Path("kmeans_resultados")

RANDOM_STATE = 42
K_MIN, K_MAX = 2, 10
K_ELEGIDO = 3         # None = el k de mayor silueta; o fuerza un entero (ej. 3)
N_INIT = 20              # reinicios aleatorios de K-Means (se conserva el mejor)
N_SUBMUESTRAS = 30       # re-ajustes para medir la estabilidad
FRAC_SUBMUESTRA = 0.8

FEATURES = [
    "log_precio_venta",       # mediana de log10(sell_price_avg)
    "log_volumen_venta",      # log10(1 + mediana de sell_sold)
    "log_spread",             # mediana de log10(sell_price_avg / buy_price_avg)
    "volatilidad_precio",     # desviacion estandar de log10(sell_price_avg) en el tiempo
    "frac_dias_sin_ventas",   # proporcion de dias sin ninguna venta
]

# Etiquetas SOLO para validar los grupos despues (K-Means no las usa)
ETIQUETAS_VALIDACION = ["rarity", "type", "level"]

plt.rcParams["figure.dpi"] = 120


# ---------------------------------------------------------------------------
# PREPARACION: una fila por item (mismas variables que knn_gw2.py)
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

    columnas_meta = [c for c in ["name", "type", "rarity", "level"] if c in df.columns]
    meta = df.groupby("itemID")[columnas_meta].first()
    if "level" in meta.columns:
        meta["level"] = meta["level"].map(lambda v: f"nivel_{int(v)}" if pd.notna(v) else np.nan)
    return items.join(meta).reset_index()


# ---------------------------------------------------------------------------
# ELECCION DE k
# ---------------------------------------------------------------------------

def evaluar_rango_k(X: np.ndarray) -> pd.DataFrame:
    filas = []
    for k in range(K_MIN, min(K_MAX, len(X) - 1) + 1):
        modelo = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
        filas.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silueta": silhouette_score(X, modelo.labels_),
            "calinski_harabasz": calinski_harabasz_score(X, modelo.labels_),
            "davies_bouldin": davies_bouldin_score(X, modelo.labels_),
            "tam_grupo_menor": int(np.bincount(modelo.labels_).min()),
        })
    return pd.DataFrame(filas)


def graficar_seleccion_k(metricas: pd.DataFrame, k_elegido: int, ruta: Path):
    paneles = [("inercia", "Inercia (metodo del codo)", "menor al subir k; buscar el codo"),
               ("silueta", "Coeficiente de silueta", "mayor es mejor"),
               ("calinski_harabasz", "Calinski-Harabasz", "mayor es mejor"),
               ("davies_bouldin", "Davies-Bouldin", "menor es mejor")]
    fig, ejes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (col, titulo, nota) in zip(ejes.ravel(), paneles):
        ax.plot(metricas["k"], metricas[col], marker="o")
        ax.axvline(k_elegido, color="firebrick", linestyle="--", linewidth=1, label=f"k elegido = {k_elegido}")
        ax.set_title(f"{titulo}\n({nota})", fontsize=10)
        ax.set_xlabel("k (numero de grupos)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# EVALUACION DEL MODELO FINAL
# ---------------------------------------------------------------------------

def estabilidad(X: np.ndarray, k: int, etiquetas_ref: np.ndarray) -> np.ndarray:
    """Ajusta K-Means sobre submuestras del 80%, asigna TODOS los items al
    modelo resultante y compara con la agrupacion de referencia usando el
    Adjusted Rand Index (1 = identica, ~0 = al azar). El ARI no depende de
    como se numeren los grupos."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_sub = int(len(X) * FRAC_SUBMUESTRA)
    resultados = []
    for _ in range(N_SUBMUESTRAS):
        idx = rng.choice(len(X), n_sub, replace=False)
        modelo = KMeans(n_clusters=k, n_init=N_INIT, random_state=int(rng.integers(1e6))).fit(X[idx])
        resultados.append(adjusted_rand_score(etiquetas_ref, modelo.predict(X)))
    return np.array(resultados)


def validacion_externa(items: pd.DataFrame, columna_grupo: str = "cluster") -> pd.DataFrame:
    filas = []
    for etiqueta in ETIQUETAS_VALIDACION:
        if etiqueta not in items.columns:
            continue
        d = items.dropna(subset=[etiqueta])
        filas.append({
            "etiqueta": etiqueta, "n_clases": d[etiqueta].nunique(), "n_items": len(d),
            "ARI": adjusted_rand_score(d[etiqueta], d[columna_grupo]),
            "NMI": normalized_mutual_info_score(d[etiqueta], d[columna_grupo]),
            "homogeneidad": homogeneity_score(d[etiqueta], d[columna_grupo]),
            "completitud": completeness_score(d[etiqueta], d[columna_grupo]),
        })
    return pd.DataFrame(filas)


def graficar_silueta(X: np.ndarray, etiquetas: np.ndarray, k: int, ruta: Path):
    valores = silhouette_samples(X, etiquetas)
    promedio = valores.mean()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y_inf = 10
    for c in range(k):
        v = np.sort(valores[etiquetas == c])
        y_sup = y_inf + len(v)
        ax.fill_betweenx(np.arange(y_inf, y_sup), 0, v, alpha=0.75)
        ax.text(-0.05, y_inf + len(v) / 2, f"grupo {c}", va="center", ha="right", fontsize=9)
        y_inf = y_sup + 10
    ax.axvline(promedio, color="firebrick", linestyle="--", label=f"silueta media = {promedio:.3f}")
    ax.set_xlabel("coeficiente de silueta por item")
    ax.set_yticks([])
    ax.set_title(f"Grafica de silueta (k = {k})\nbarras que cruzan a la izquierda de 0 = items probablemente mal asignados",
                 fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# INTERPRETACION
# ---------------------------------------------------------------------------

def perfil_grupos(items: pd.DataFrame) -> pd.DataFrame:
    """Resume cada grupo en unidades reales (precio en cobre; 10,000 cobre = 1 oro)."""
    filas = []
    for c, sub in items.groupby("cluster"):
        fila = {
            "cluster": c, "n_items": len(sub),
            "precio_venta_mediano_cobre": 10 ** sub["log_precio_venta"].median(),
            "ventas_diarias_medianas": 10 ** sub["log_volumen_venta"].median() - 1,
            "razon_venta_compra": 10 ** sub["log_spread"].median(),
            "volatilidad_precio_log10": sub["volatilidad_precio"].median(),
            "frac_dias_sin_ventas": sub["frac_dias_sin_ventas"].median(),
            "silueta_media": sub["silueta"].mean(),
        }
        for etiqueta in ["rarity", "type", "level"]:
            if etiqueta in sub.columns and sub[etiqueta].notna().any():
                conteo = sub[etiqueta].value_counts()
                fila[f"{etiqueta}_dominante"] = f"{conteo.index[0]} ({conteo.iloc[0]}/{len(sub)})"
        filas.append(fila)
    return pd.DataFrame(filas)


def graficar_perfiles(centroides_std: pd.DataFrame, ruta: Path):
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(centroides_std) + 2.5))
    lim = np.abs(centroides_std.to_numpy()).max()
    im = ax.imshow(centroides_std, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(centroides_std.columns)))
    ax.set_xticklabels(centroides_std.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(centroides_std)))
    ax.set_yticklabels([f"grupo {i}" for i in centroides_std.index])
    for i in range(centroides_std.shape[0]):
        for j in range(centroides_std.shape[1]):
            ax.text(j, i, f"{centroides_std.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Centroides (desviaciones estandar respecto a la media global)")
    fig.colorbar(im, ax=ax, label="valor estandarizado")
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def graficar_pca(X: np.ndarray, items: pd.DataFrame, centroides: np.ndarray, ruta: Path):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    proy = pca.fit_transform(X)
    proy_c = pca.transform(centroides)
    var = pca.explained_variance_ratio_

    etiqueta = next((e for e in ETIQUETAS_VALIDACION if e in items.columns and items[e].notna().any()), None)
    n_paneles = 2 if etiqueta else 1
    fig, ejes = plt.subplots(1, n_paneles, figsize=(6.5 * n_paneles, 5.5), squeeze=False)
    ax = ejes[0, 0]
    for c in sorted(items["cluster"].unique()):
        m = (items["cluster"] == c).to_numpy()
        ax.scatter(proy[m, 0], proy[m, 1], s=22, alpha=0.7, label=f"grupo {c}")
    ax.scatter(proy_c[:, 0], proy_c[:, 1], marker="X", s=180, c="black", edgecolor="white", label="centroides")
    ax.set_title("Agrupacion K-Means (proyeccion PCA)")
    ax.legend(fontsize=8)

    if etiqueta:
        ax2 = ejes[0, 1]
        for valor in sorted(items[etiqueta].dropna().unique()):
            m = (items[etiqueta] == valor).to_numpy()
            ax2.scatter(proy[m, 0], proy[m, 1], s=22, alpha=0.7, label=str(valor))
        ax2.set_title(f"Mismos items coloreados por '{etiqueta}' (no usada por K-Means)")
        ax2.legend(fontsize=8)

    for a in ejes.ravel():
        a.set_xlabel(f"componente 1 ({var[0]:.0%} de la varianza)")
        a.set_ylabel(f"componente 2 ({var[1]:.0%} de la varianza)")
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return var


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    items = construir_dataset_items(df).dropna(subset=FEATURES).reset_index(drop=True)
    print(f"Items con todas las variables: {len(items)}  |  variables: {FEATURES}")

    escalador = StandardScaler()
    X = escalador.fit_transform(items[FEATURES])

    # --- 1. Elegir k ---
    print("\n" + "=" * 78)
    print("Seleccion de k")
    metricas = evaluar_rango_k(X)
    print(metricas.round(3).to_string(index=False))
    k_auto = int(metricas.loc[metricas["silueta"].idxmax(), "k"])
    k = K_ELEGIDO or k_auto
    origen = "forzado por K_ELEGIDO" if K_ELEGIDO else "mayor silueta"
    print(f"\n-> k elegido = {k} ({origen})")
    metricas.round(4).to_csv(OUTPUT_DIR / "metricas_por_k.csv", index=False)
    graficar_seleccion_k(metricas, k, OUTPUT_DIR / "kmeans_seleccion_k.png")

    # --- 2. Modelo final ---
    modelo = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
    items["cluster"] = modelo.labels_
    items["silueta"] = silhouette_samples(X, modelo.labels_)

    print("\n" + "=" * 78)
    print(f"Evaluacion interna (k = {k})")
    print(f"  Silueta media: {silhouette_score(X, modelo.labels_):.3f}  "
          f"|  Calinski-Harabasz: {calinski_harabasz_score(X, modelo.labels_):.1f}  "
          f"|  Davies-Bouldin: {davies_bouldin_score(X, modelo.labels_):.3f}")
    print(f"  Tamano de los grupos: {dict(sorted(pd.Series(modelo.labels_).value_counts().items()))}")
    print(f"  Items con silueta negativa (posiblemente mal asignados): {(items['silueta'] < 0).sum()} de {len(items)}")
    graficar_silueta(X, modelo.labels_, k, OUTPUT_DIR / "kmeans_silueta.png")

    ari = estabilidad(X, k, modelo.labels_)
    print(f"  Estabilidad ({N_SUBMUESTRAS} submuestras de {FRAC_SUBMUESTRA:.0%}): "
          f"ARI medio = {ari.mean():.3f} ± {ari.std():.3f}  (minimo {ari.min():.3f})")

    # --- 3. Validacion externa ---
    print("\n" + "=" * 78)
    print("Validacion externa: ¿coinciden los grupos con las categorias del juego?")
    externa = validacion_externa(items)
    print(externa.round(3).to_string(index=False))
    externa.round(4).to_csv(OUTPUT_DIR / "validacion_externa.csv", index=False)
    for etiqueta in ETIQUETAS_VALIDACION:
        if etiqueta in items.columns and items[etiqueta].notna().any():
            tabla = pd.crosstab(items["cluster"], items[etiqueta])
            print(f"\n  Grupos (filas) vs {etiqueta} (columnas):")
            print(tabla.to_string())
            tabla.to_csv(OUTPUT_DIR / f"cruce_cluster_{etiqueta}.csv")

    # --- 4. Interpretacion ---
    print("\n" + "=" * 78)
    print("Perfil de cada grupo (unidades reales; 10,000 cobre = 1 oro)")
    perfiles = perfil_grupos(items)
    print(perfiles.round(3).T.to_string(header=False))
    perfiles.round(4).to_csv(OUTPUT_DIR / "perfil_grupos.csv", index=False)

    centroides_std = pd.DataFrame(modelo.cluster_centers_, columns=FEATURES)
    graficar_perfiles(centroides_std, OUTPUT_DIR / "kmeans_perfiles.png")
    var = graficar_pca(X, items, modelo.cluster_centers_, OUTPUT_DIR / "kmeans_pca.png")
    print(f"\n  PCA: los 2 primeros componentes explican {var.sum():.0%} de la varianza "
          f"(la grafica es una proyeccion; la distancia real es en 5 dimensiones)")

    columnas_salida = [c for c in ["itemID", "name", "cluster", "silueta", "rarity", "type", "level"] + FEATURES
                       if c in items.columns]
    items[columnas_salida].round(4).to_csv(OUTPUT_DIR / "items_con_cluster.csv", index=False)
    print(f"\nResultados guardados en '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
