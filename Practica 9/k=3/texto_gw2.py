#!/usr/bin/env python3
"""
texto_gw2.py

Analisis de texto del dataset de GW2: nube de palabras y analisis
exploratorio de las columnas textuales (name, description, flags).

PUNTO CLAVE: el CSV tiene una fila por item y por DIA (~250,000 filas), asi
que el nombre de cada item se repite miles de veces. Contar palabras sobre
esas filas daria frecuencias absurdas (la palabra de un item aparecería 1,826
veces). Por eso el texto se analiza a nivel de ITEM: una fila por itemID
(~140 documentos). Las frecuencias son "cuantas veces aparece en los items",
no "en cuantos dias".

Contenido (cada parte se genera con ciclos sobre listas de especificaciones):

  1. Limpieza: minusculas, quita etiquetas de formato del juego
     (<c=@flavor>...</c>, <br>) y marcadores (%str1%), separa palabras,
     elimina palabras vacias (the, of, ...). Los numeros se descartan
     ("+14 Agony Infusion" -> "agony", "infusion").
  2. Exploracion: cuantos items tienen descripcion, longitud de nombres y
     descripciones, tamano del vocabulario, diversidad lexica, palabras y
     bigramas mas frecuentes, restricciones (flags) mas comunes.
  3. Nubes de palabras: nombres, descripciones y flags; y una nube POR GRUPO
     (rareza, tipo y, si existe, grupo de K-Means), pesada con TF-IDF para
     destacar las palabras DISTINTIVAS de cada grupo y no las que son comunes
     a todos.
  4. Relacion texto <-> mercado: precio mediano de los items que contienen
     cada palabra frecuente, correlacion entre largo del nombre y precio, y
     comparacion del precio de items con y sin descripcion.

La nube usa la libreria `wordcloud` si esta instalada (pip install wordcloud);
si no, se usa una version sencilla incluida aqui (misma salida, aspecto menos
pulido).
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventanas: solo se guardan archivos (evita fallos de Tk en Windows)
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

try:
    from wordcloud import WordCloud
    HAY_WORDCLOUD = True
except ImportError:
    HAY_WORDCLOUD = False

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------

CSV_PATH = "dataset.csv"
CLUSTERS_CSV = "items_con_cluster.csv"   # salida de kmeans_gw2.py (opcional)
OUTPUT_DIR = Path("texto_resultados")

IDIOMA = "en"             # idioma del texto del juego: "en" (lang=en de la API) o "es"
MIN_LONGITUD = 3          # palabras de menos letras se descartan
TOP_N = 15                # palabras mostradas en las graficas de barras
MAX_PALABRAS_NUBE = 100
MIN_ITEMS_GRUPO = 5       # grupos con menos items no se muestran
MIN_ITEMS_PALABRA = 4     # una palabra debe aparecer en >= 4 items para compararla contra el precio
RANDOM_STATE = 42

# Palabras vacias adicionales (puedes editar). Las del ingles vienen de scikit-learn.
STOPWORDS_EXTRA = {"item", "items"}
STOPWORDS_ES = {"de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "en", "con", "por",
                "para", "del", "al", "que", "se", "su", "sus", "es", "lo", "como", "mas", "pero", "si"}
STOPWORDS = (set(ENGLISH_STOP_WORDS) if IDIOMA == "en" else set(STOPWORDS_ES)) | STOPWORDS_EXTRA

# Textos a analizar: cada uno genera su tabla de frecuencias y su nube.
CORPUS = [
    {"clave": "nombres", "col": "tok_nombre", "titulo": "Nombres de items"},
    {"clave": "descripciones", "col": "tok_desc", "titulo": "Descripciones de items"},
    {"clave": "flags", "col": "tok_flags", "titulo": "Restricciones (flags)"},
]

# Variables por las que se agrupa para las nubes distintivas (TF-IDF)
GRUPOS = ["rarity", "type", "cluster"]
TEXTO_GRUPOS = "tok_nombre"   # texto usado para caracterizar grupos: nombres (siempre disponibles)

PATRON_PALABRA = re.compile(r"[a-záéíóúñü]+(?:'[a-z]+)?")

plt.rcParams["figure.dpi"] = 120


# ---------------------------------------------------------------------------
# LIMPIEZA Y TOKENIZACION
# ---------------------------------------------------------------------------

def limpiar_texto(valor) -> str:
    if not isinstance(valor, str):
        return ""
    t = re.sub(r"<[^>]+>", " ", valor)            # etiquetas de formato: <c=@flavor>...</c>, <br>
    t = re.sub(r"%\w+%", " ", t)                  # marcadores de posicion: %str1%
    t = t.replace("\\n", " ").replace("’", "'")
    return re.sub(r"\s+", " ", t).strip().lower()


def tokens_crudos(texto: str) -> list:
    """Todas las palabras en orden (incluye palabras vacias; sirven para bigramas)."""
    return [re.sub(r"'s$", "", t) for t in PATRON_PALABRA.findall(texto)]


def filtrar(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) >= MIN_LONGITUD]


def tokens_flags(valor) -> list:
    """Las flags son nombres tipo CamelCase (AccountBound, NoSell): se conservan tal cual."""
    return re.findall(r"[A-Za-z]+", valor) if isinstance(valor, str) else []


# ---------------------------------------------------------------------------
# PREPARACION: una fila por item
# ---------------------------------------------------------------------------

def cargar_items(path: str) -> pd.DataFrame:
    columnas_texto = ["name", "description", "flags", "type", "rarity", "level"]
    permitidas = {"itemID", "sell_price_avg"} | set(columnas_texto)
    df = pd.read_csv(path, usecols=lambda c: c in permitidas)
    if "name" not in df.columns:
        raise SystemExit("El CSV no tiene la columna 'name': no hay texto que analizar.")

    base = df.groupby("itemID")[[c for c in columnas_texto if c in df.columns]].first()
    precio = (df[df["sell_price_avg"] > 0].groupby("itemID")["sell_price_avg"].median()
              .rename("precio_mediano")) if "sell_price_avg" in df.columns else None
    items = base.join(precio) if precio is not None else base
    items = items.reset_index()

    ruta = Path(CLUSTERS_CSV)
    if ruta.exists():
        c = pd.read_csv(ruta, usecols=["itemID", "cluster"])
        c["cluster"] = "grupo_" + c["cluster"].astype(int).astype(str)
        items = items.merge(c, on="itemID", how="left")
    return items


def preparar_texto(items: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    items["tok_crudo_nombre"] = items["name"].map(limpiar_texto).map(tokens_crudos)
    items["tok_nombre"] = items["tok_crudo_nombre"].map(filtrar)

    descripcion = items["description"] if "description" in items.columns else pd.Series("", index=items.index)
    items["tok_crudo_desc"] = descripcion.map(limpiar_texto).map(tokens_crudos)
    items["tok_desc"] = items["tok_crudo_desc"].map(filtrar)
    items["tiene_descripcion"] = items["tok_crudo_desc"].map(len) > 0

    flags = items["flags"] if "flags" in items.columns else pd.Series(None, index=items.index)
    items["tok_flags"] = flags.map(tokens_flags)
    items["n_palabras_nombre"] = items["tok_crudo_nombre"].map(len)
    items["n_palabras_desc"] = items["tok_crudo_desc"].map(len)
    return items


# ---------------------------------------------------------------------------
# FRECUENCIAS
# ---------------------------------------------------------------------------

def frecuencias(lista_tokens, columna: str = "palabra") -> pd.DataFrame:
    """frecuencia = apariciones totales; items = en cuantos items aparece."""
    total, docs = Counter(), Counter()
    for toks in lista_tokens:
        total.update(toks)
        docs.update(set(toks))
    if not total:
        return pd.DataFrame(columns=[columna, "frecuencia", "items"])
    tabla = pd.DataFrame({columna: list(total), "frecuencia": list(total.values())})
    tabla["items"] = tabla[columna].map(docs)
    return tabla.sort_values(["frecuencia", "items", columna], ascending=[False, False, True]).reset_index(drop=True)


def bigramas(lista_crudos) -> pd.DataFrame:
    """Pares de palabras consecutivas del texto original (sin palabras vacias en ningun extremo)."""
    pares_por_item = []
    for toks in lista_crudos:
        pares = [f"{a} {b}" for a, b in zip(toks, toks[1:])
                 if a not in STOPWORDS and b not in STOPWORDS and len(a) >= MIN_LONGITUD and len(b) >= MIN_LONGITUD]
        pares_por_item.append(pares)
    return frecuencias(pares_por_item, columna="bigrama")


# ---------------------------------------------------------------------------
# NUBES DE PALABRAS
# ---------------------------------------------------------------------------

def _nube_simple(ax, freqs: dict):
    """Nube sencilla: tamano de letra segun frecuencia, colocacion en espiral sin solapes."""
    ancho, alto = 1000, 600
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, alto)
    ax.axis("off")
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inversa = ax.transData.inverted()
    palabras = sorted(freqs.items(), key=lambda kv: -kv[1])[:MAX_PALABRAS_NUBE]
    maximo = palabras[0][1]
    mapa = plt.get_cmap("viridis")
    rng = np.random.default_rng(RANDOM_STATE)
    ocupados = []
    for palabra, f in palabras:
        texto = ax.text(ancho / 2, alto / 2, palabra, fontsize=9 + 46 * (f / maximo) ** 0.6,
                        ha="center", va="center", color=mapa(0.05 + 0.8 * rng.random()))
        for paso in range(700):
            theta = 0.35 * paso
            r = 3.2 * theta
            texto.set_position((ancho / 2 + 1.6 * r * np.cos(theta), alto / 2 + r * np.sin(theta)))
            caja = texto.get_window_extent(renderer)
            (x0, y0), (x1, y1) = inversa.transform([(caja.x0, caja.y0), (caja.x1, caja.y1)])
            if x0 < 0 or x1 > ancho or y0 < 0 or y1 > alto:
                continue
            if any(not (x1 < a0 - 4 or x0 > a1 + 4 or y1 < b0 - 4 or y0 > b1 + 4) for a0, b0, a1, b1 in ocupados):
                continue
            ocupados.append((x0, y0, x1, y1))
            break
        else:
            texto.remove()


def dibujar_nube(ax, freqs: dict, titulo: str):
    ax.set_title(titulo, fontsize=11)
    if not freqs:
        ax.text(0.5, 0.5, "sin texto suficiente", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    if HAY_WORDCLOUD:
        nube = WordCloud(width=1000, height=600, background_color="white", colormap="viridis",
                         max_words=MAX_PALABRAS_NUBE, random_state=RANDOM_STATE, prefer_horizontal=0.9
                         ).generate_from_frequencies(freqs)
        ax.imshow(nube, interpolation="bilinear")
        ax.axis("off")
    else:
        _nube_simple(ax, freqs)


def guardar_nube_simple(freqs: dict, titulo: str, ruta: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(titulo)
    fig.tight_layout()
    dibujar_nube(ax, freqs, titulo)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def pesos_tfidf_por_grupo(items: pd.DataFrame, grupo: str):
    """TF-IDF con un 'documento' por grupo: pesa mas las palabras propias de ese grupo.
    Devuelve ({grupo: {palabra: peso}}, {grupo: n_items}) o None si no hay suficientes grupos."""
    d = items.dropna(subset=[grupo])
    conteos = d[grupo].value_counts()
    d = d[d[grupo].isin(conteos[conteos >= MIN_ITEMS_GRUPO].index)]
    docs = d.groupby(grupo)[TEXTO_GRUPOS].apply(lambda s: " ".join(" ".join(t) for t in s))
    docs = docs[docs.str.strip() != ""]
    if len(docs) < 2:
        return None
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
    matriz = vec.fit_transform(docs.values)
    terminos = np.array(vec.get_feature_names_out())
    pesos = {}
    for i, g in enumerate(docs.index):
        fila = matriz[i].toarray().ravel()
        pesos[g] = {terminos[j]: float(fila[j]) for j in np.where(fila > 0)[0]}
    return pesos, conteos.loc[docs.index].to_dict()


def nubes_por_grupo(items: pd.DataFrame, grupo: str, ruta: Path):
    resultado = pesos_tfidf_por_grupo(items, grupo)
    if resultado is None:
        print(f"  '{grupo}': no hay al menos 2 grupos con >= {MIN_ITEMS_GRUPO} items y texto; se omite.")
        return []
    pesos, n_items = resultado
    grupos = sorted(pesos)
    ncols = min(3, len(grupos))
    nfilas = int(np.ceil(len(grupos) / ncols))
    fig, ejes = plt.subplots(nfilas, ncols, figsize=(6 * ncols, 3.8 * nfilas), squeeze=False)
    for ax in ejes.ravel()[len(grupos):]:
        ax.axis("off")
    for g, ax in zip(grupos, ejes.ravel()):
        ax.set_title(f"{g} (n = {n_items[g]})", fontsize=11)
    fig.suptitle(f"Palabras distintivas por '{grupo}' (peso TF-IDF)", fontsize=13)
    fig.tight_layout()
    for g, ax in zip(grupos, ejes.ravel()):
        dibujar_nube(ax, pesos[g], f"{g} (n = {n_items[g]})")
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)

    filas = []
    for g in grupos:
        for palabra, peso in sorted(pesos[g].items(), key=lambda kv: -kv[1])[:8]:
            filas.append({"variable": grupo, "grupo": g, "n_items": n_items[g], "termino": palabra, "peso_tfidf": round(peso, 4)})
    return filas


# ---------------------------------------------------------------------------
# GRAFICAS DE EXPLORACION
# ---------------------------------------------------------------------------

def graficar_top(tablas: dict, ruta: Path):
    paneles = [("nombres", "Palabras mas frecuentes en nombres"), ("bigramas", "Bigramas mas frecuentes en nombres"),
               ("descripciones", "Palabras mas frecuentes en descripciones"), ("flags", "Restricciones (flags) mas comunes")]
    fig, ejes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (clave, titulo) in zip(ejes.ravel(), paneles):
        t = tablas.get(clave)
        ax.set_title(titulo, fontsize=11)
        if t is None or t.empty:
            ax.text(0.5, 0.5, "sin datos", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        top = t.head(TOP_N).iloc[::-1]
        ax.barh(top.iloc[:, 0], top["frecuencia"], color="steelblue")
        ax.set_xlabel("apariciones en el conjunto de items")
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def graficar_distribuciones(items: pd.DataFrame, rho: float, p: float, ruta: Path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))
    ax1.hist(items["n_palabras_nombre"], bins=np.arange(0.5, items["n_palabras_nombre"].max() + 1.5, 1), color="steelblue", edgecolor="white")
    ax1.set_title("Largo de los nombres")
    ax1.set_xlabel("palabras por nombre")
    ax1.set_ylabel("items")

    con = items.loc[items["tiene_descripcion"], "n_palabras_desc"]
    if len(con):
        ax2.hist(con, bins=min(20, max(5, len(con) // 3)), color="darkorange", edgecolor="white")
    ax2.set_title(f"Largo de las descripciones\n({len(con)} de {len(items)} items tienen)")
    ax2.set_xlabel("palabras por descripcion")

    d = items.dropna(subset=["precio_mediano"]) if "precio_mediano" in items.columns else items.iloc[0:0]
    if len(d):
        jitter = np.random.default_rng(RANDOM_STATE).uniform(-0.12, 0.12, len(d))
        ax3.scatter(d["n_palabras_nombre"] + jitter, d["precio_mediano"], alpha=0.55, s=22)
        ax3.set_yscale("log")
    ax3.set_title(f"Largo del nombre vs precio\nSpearman rho = {rho:.2f} (p = {p:.3g})" if np.isfinite(rho) else "Largo del nombre vs precio")
    ax3.set_xlabel("palabras por nombre")
    ax3.set_ylabel("precio de venta mediano (cobre)")
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def palabra_vs_precio(items: pd.DataFrame, ruta: Path):
    d = items.dropna(subset=["precio_mediano"]) if "precio_mediano" in items.columns else None
    if d is None or d.empty:
        return None
    conteo = frecuencias(d["tok_nombre"])
    candidatas = conteo[conteo["items"] >= MIN_ITEMS_PALABRA].sort_values("items", ascending=False).head(20)["palabra"]
    filas = []
    for palabra in candidatas:
        precios = d.loc[d["tok_nombre"].map(lambda t: palabra in t), "precio_mediano"]
        filas.append({"palabra": palabra, "n_items": len(precios), "precio_mediano_cobre": precios.median()})
    if not filas:
        return None
    tabla = pd.DataFrame(filas).sort_values("precio_mediano_cobre")
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(tabla) + 2))
    ax.barh([f"{p}  (n={n})" for p, n in zip(tabla["palabra"], tabla["n_items"])], tabla["precio_mediano_cobre"], color="seagreen")
    ax.axvline(d["precio_mediano"].median(), color="firebrick", linestyle="--", label=f"mediana global = {d['precio_mediano'].median():,.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("precio de venta mediano de los items que la contienen (cobre, escala log)")
    ax.set_title("Precio segun las palabras del nombre")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return tabla


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    items = preparar_texto(cargar_items(CSV_PATH))
    print(f"Nube de palabras con: {'libreria wordcloud' if HAY_WORDCLOUD else 'version sencilla incluida (pip install wordcloud para la oficial)'}")

    # --- 1. Exploracion descriptiva ---
    print("\n" + "=" * 78)
    print("Exploracion del texto (una fila por item)")
    n = len(items)
    con_desc = int(items["tiene_descripcion"].sum())
    print(f"  Items unicos: {n}")
    print(f"  Nombres: {items['n_palabras_nombre'].mean():.1f} palabras en promedio "
          f"(min {items['n_palabras_nombre'].min()}, max {items['n_palabras_nombre'].max()}); "
          f"{items['name'].nunique()} nombres distintos")
    print(f"  Descripciones: {con_desc} de {n} items tienen ({con_desc / n:.0%})"
          + (f"; {items.loc[items['tiene_descripcion'], 'n_palabras_desc'].mean():.1f} palabras en promedio entre los que tienen" if con_desc else ""))

    resumen = {"items": n, "items_con_descripcion": con_desc}
    tablas = {}
    for corpus in CORPUS:
        tabla = frecuencias(items[corpus["col"]])
        tablas[corpus["clave"]] = tabla
        tabla.to_csv(OUTPUT_DIR / f"frecuencias_{corpus['clave']}.csv", index=False)
        total = int(tabla["frecuencia"].sum()) if len(tabla) else 0
        diversidad = len(tabla) / total if total else float("nan")
        print(f"\n  {corpus['titulo']}: {total} palabras, vocabulario de {len(tabla)} distintas "
              f"(diversidad lexica = {diversidad:.2f}; 1 = ninguna se repite)")
        if len(tabla):
            print("    Mas frecuentes: " + ", ".join(f"{r.palabra} ({r.frecuencia})" for r in tabla.head(10).itertuples()))
        resumen[f"vocabulario_{corpus['clave']}"] = len(tabla)

    bi = bigramas(items["tok_crudo_nombre"])
    tablas["bigramas"] = bi
    bi.to_csv(OUTPUT_DIR / "bigramas_nombres.csv", index=False)
    if len(bi):
        print("\n  Bigramas mas frecuentes en nombres: " + ", ".join(f"{r.bigrama} ({r.frecuencia})" for r in bi.head(8).itertuples()))

    if "type" in items.columns and con_desc:
        cruce = pd.crosstab(items["type"], items["tiene_descripcion"], normalize="index").round(3)
        cruce.columns = ["sin_descripcion" if not c else "con_descripcion" for c in cruce.columns]
        print("\n  Proporcion de items con descripcion por tipo:")
        print(cruce.to_string())
        cruce.to_csv(OUTPUT_DIR / "descripcion_por_tipo.csv")

    # --- 2. Nubes de palabras (ciclo sobre los corpus) ---
    print("\n" + "=" * 78)
    print("Nubes de palabras")
    for corpus in CORPUS:
        tabla = tablas[corpus["clave"]]
        freqs = dict(zip(tabla["palabra"].head(MAX_PALABRAS_NUBE), tabla["frecuencia"].head(MAX_PALABRAS_NUBE)))
        guardar_nube_simple(freqs, f"{corpus['titulo']}: palabras mas frecuentes", OUTPUT_DIR / f"nube_{corpus['clave']}.png")
        print(f"  nube_{corpus['clave']}.png" + ("" if freqs else "  (sin texto)"))

    # --- 3. Palabras distintivas por grupo (ciclo sobre GRUPOS) ---
    print("\n" + "=" * 78)
    print("Palabras distintivas por grupo (TF-IDF)")
    filas_tfidf = []
    for grupo in GRUPOS:
        if grupo not in items.columns:
            print(f"  '{grupo}': columna no disponible; se omite.")
            continue
        filas = nubes_por_grupo(items, grupo, OUTPUT_DIR / f"nubes_por_{grupo}.png")
        filas_tfidf += filas
        for g in sorted({f["grupo"] for f in filas}):
            top = [f["termino"] for f in filas if f["grupo"] == g][:5]
            n_g = next(f["n_items"] for f in filas if f["grupo"] == g)
            print(f"  {grupo} = {g} (n={n_g}): " + ", ".join(top))
    if filas_tfidf:
        pd.DataFrame(filas_tfidf).to_csv(OUTPUT_DIR / "terminos_distintivos_por_grupo.csv", index=False)

    # --- 4. Relacion texto <-> mercado ---
    print("\n" + "=" * 78)
    print("Relacion entre el texto y el mercado")
    rho = p = float("nan")
    if "precio_mediano" in items.columns:
        d = items.dropna(subset=["precio_mediano"])
        rho, p = stats.spearmanr(d["n_palabras_nombre"], d["precio_mediano"])
        print(f"  Largo del nombre vs precio: Spearman rho = {rho:.3f} (p = {p:.3g}, n = {len(d)})")
        con = d.loc[d["tiene_descripcion"], "precio_mediano"]
        sin = d.loc[~d["tiene_descripcion"], "precio_mediano"]
        if len(con) >= 3 and len(sin) >= 3:
            u = stats.mannwhitneyu(con, sin, alternative="two-sided")
            print(f"  Precio mediano con descripcion = {con.median():,.0f} cobre (n = {len(con)}) | "
                  f"sin descripcion = {sin.median():,.0f} cobre (n = {len(sin)}) | Mann-Whitney p = {u.pvalue:.3g}")
        tabla_precio = palabra_vs_precio(items, OUTPUT_DIR / "palabra_vs_precio.png")
        if tabla_precio is not None:
            tabla_precio.round(1).to_csv(OUTPUT_DIR / "palabra_vs_precio.csv", index=False)
            print("  Palabras con precio mediano mas alto: " + ", ".join(
                f"{r.palabra} ({r.precio_mediano_cobre:,.0f})" for r in tabla_precio.tail(3).iloc[::-1].itertuples()))
            print("  Palabras con precio mediano mas bajo: " + ", ".join(
                f"{r.palabra} ({r.precio_mediano_cobre:,.0f})" for r in tabla_precio.head(3).itertuples()))
    else:
        print("  El CSV no trae sell_price_avg; se omite esta parte.")

    graficar_top(tablas, OUTPUT_DIR / "top_palabras.png")
    graficar_distribuciones(items, rho, p, OUTPUT_DIR / "distribuciones_texto.png")
    pd.DataFrame([resumen]).to_csv(OUTPUT_DIR / "resumen_texto.csv", index=False)
    items.drop(columns=[c for c in items.columns if c.startswith("tok_")]).to_csv(OUTPUT_DIR / "items_texto.csv", index=False)
    print(f"\nResultados guardados en '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
