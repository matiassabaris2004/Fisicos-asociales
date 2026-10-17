"""
 Importo las librerias
"""

import os
import numpy as np
import pandas as pd
import duckdb as dd
from sentence_transformers import SentenceTransformer

#%%
"""
 Cargo todos los archivos .parquet de los datos proporcionados
"""

CSV_DIR = r"C:\Users\Diego\OneDrive\Documentos\Mati\Estudios\ranking-licitaciones\data\data\csv_exportados"

# Diccionario para almacenar los DataFrames
dfs = {}

# Recorrer todos los archivos de la carpeta
for archivo in os.listdir(CSV_DIR):
    if archivo.endswith(".csv"):
        # Nombre de la variable
        nombre_var = f"df_{os.path.splitext(archivo)[0]}"
        
        ruta_completa = os.path.join(CSV_DIR, archivo)
        
        # Cargar el CSV
        df = pd.read_csv(ruta_completa)
        
        # Inyectar en el Explorador de Variables
        dfs[nombre_var] = df
        globals()[nombre_var] = df
        
        print(f"✅ Cargado en Explorador de Variables: {nombre_var} ({df.shape[0]} filas, {df.shape[1]} columnas)")

print(f"\n¡Listo! Se cargaron {len(dfs)} DataFrames perfectamente.")

#%%
"""
Paso fechas y t a datetime para todos los grupos en un bucle
"""

grupos = ["d", "r", "j", "it", "rd"]

for g in grupos:
    print(f"\n{'=' * 10} Procesando grupo: {g.upper()} {'=' * 10}")
    
    # Obtenemos los DataFrames dinámicamente desde el entorno global
    df_licitaciones = globals()[f"df_{g}_licitaciones"]
    df_historial = globals()[f"df_{g}_historial"]
    
    # Conversión a datetime
    df_licitaciones["t"] = pd.to_datetime(
        df_licitaciones["t"],
        format="%Y-%m-%d",
        errors="coerce")

    df_historial["fecha"] = pd.to_datetime(
        df_historial["fecha"],
        format="%Y-%m-%d",
        errors="coerce")

    # Comprobar la conversión
    print(f"dtype t ({g}): {df_licitaciones['t'].dtype}")
    print(f"dtype fecha ({g}): {df_historial['fecha'].dtype}")

    # Comprobar si hubo fechas que no pudieron convertirse
    print(f"Fechas inválidas en licitaciones ({g}):", df_licitaciones["t"].isna().sum())
    print(f"Fechas inválidas en historial ({g}):", df_historial["fecha"].isna().sum())

#%%
"""
 Esta celda va a tardar un huevo porque tiene que hacer el embbeding de cada
teto de cada licitación y compararlo con el historial, a mi me tardó...
"""

SUBCARPETAS = ["d", "it", "j", "r", "rd"]
TOP_K = 10
SALIDA = "ranking_textual_global.parquet"
CACHE_LIC = "emb_licitaciones.npy"
CACHE_HIS = "emb_historial.npy"

"""
 Concateno todas las verticales ya que considero que las mismas no son
independientes, es decir, una misma empresa que esté en d por ejemplo, puede
ganar una licitación de d y de it
"""

def preparar_datos(subcarpetas):
    lics, hists = [], []
    for sub in subcarpetas:
        lic = globals().get(f"df_{sub}_licitaciones")
        his = globals().get(f"df_{sub}_historial")
        if lic is None or his is None:
            print(f"⚠️ Faltan datos de '{sub}', se omite.")
            continue

        lic = lic.copy()
        lic["vertical"] = sub
        lic["id_global"] = sub + "_" + lic["id"].astype(str)   # ids únicos entre verticales
        lics.append(lic)

        his = his.copy()
        his["vertical"] = sub
        hists.append(his)

    licitaciones = pd.concat(lics, ignore_index=True)
    historial = pd.concat(hists, ignore_index=True)

    # Fechas
    licitaciones["t"] = pd.to_datetime(licitaciones["t"])
    historial["fecha"] = pd.to_datetime(historial["fecha"])

    # Texto de cada licitación
    licitaciones["texto"] = (
        licitaciones["title"].fillna("").astype(str) + ". "
        + licitaciones["description"].fillna("").astype(str)).str.strip()

    # Limpieza del historial
    historial["parent_uei"] = historial["parent_uei"].astype("string").str.strip()
    historial["descripcion"] = historial["descripcion"].fillna("").astype(str).str.strip()
    historial = historial[
        historial["parent_uei"].notna()
        & historial["descripcion"].ne("")
        & historial["fecha"].notna()]

    # Historial ordenado por fecha (necesario para cortar con searchsorted)
    historial = historial.sort_values("fecha").reset_index(drop=True)
    licitaciones = licitaciones[licitaciones["t"].notna()].reset_index(drop=True)
    return licitaciones, historial

licitaciones, historial = preparar_datos(SUBCARPETAS)
print(f"Licitaciones: {len(licitaciones):,} | Adjudicaciones históricas: {len(historial):,}")
print(licitaciones["vertical"].value_counts().to_string())

"""
 Acá está la magia de transformer que es la que hace el embbeding, es decir, la
que calcula el vector de cada palabra en el espacio de palabras (384 dimensiones),
para luego reducirlo a 2 dimensiones y tomar la distancia angular entre palabras,
obteniendo la similarity entre las mismas.

 Este modelo 'sentence-transformers/all-MiniLM-L6-v2' es un modelo de embbeding
en inglés de ~248 tokens, una cosa así.
"""

modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embeddings(textos, ruta_cache):
    if os.path.exists(ruta_cache):
        emb = np.load(ruta_cache)
        if emb.shape[0] == len(textos):
            print(f"Embeddings cargados desde {ruta_cache}")
            return emb
    emb = modelo.encode(textos, normalize_embeddings=True,
                        batch_size=128, show_progress_bar=True)
    np.save(ruta_cache, emb)
    return emb

emb_lic = embeddings(licitaciones["texto"].tolist(), CACHE_LIC)
emb_his = embeddings(historial["descripcion"].tolist(), CACHE_HIS)


"""
 Ahora ordeno las empresas y hago un ranking global entre las seleccionadas
para cada licitación
"""

fechas_his = historial["fecha"].to_numpy().astype("datetime64[ns]")
fechas_his_i8 = fechas_his.astype("int64")
fechas_lic = licitaciones["t"].to_numpy().astype("datetime64[ns]")
NS_DIA = 86_400 * 1e9

codigos, empresas = pd.factorize(historial["parent_uei"])
n_emp = len(empresas)

verticales = sorted(set(licitaciones["vertical"]) | set(historial["vertical"]))
mapa_vert = {v: i for i, v in enumerate(verticales)}
vert_his = historial["vertical"].map(mapa_vert).to_numpy()
vert_lic = licitaciones["vertical"].map(mapa_vert).to_numpy()

ids_lic = licitaciones["id_global"].to_numpy()
nombres_vert = licitaciones["vertical"].to_numpy()

resultados = []

for i in range(len(licitaciones)):
    # Solo adjudicaciones estrictamente anteriores para no ver el futuro
    corte = np.searchsorted(fechas_his, fechas_lic[i], side="left")
    if corte == 0:
        continue

    sims = emb_his[:corte] @ emb_lic[i]
    cod = codigos[:corte]

    # Agregados por empresa
    n = np.bincount(cod, minlength=n_emp)
    suma = np.bincount(cod, weights=sims, minlength=n_emp)
    maxi = np.full(n_emp, -np.inf)
    np.maximum.at(maxi, cod, sims)

    # Promedio de los 3 contratos más parecidos de cada empresa
    orden = np.lexsort((-sims, cod))
    cod_o, sims_o = cod[orden], sims[orden]
    inicios = np.r_[0, np.flatnonzero(np.diff(cod_o)) + 1]
    tamanos = np.diff(np.r_[inicios, len(cod_o)])
    rango = np.arange(len(cod_o)) - np.repeat(inicios, tamanos)
    top = rango < 3
    top3_suma = np.bincount(cod_o[top], weights=sims_o[top], minlength=n_emp)
    top3_n = np.bincount(cod_o[top], minlength=n_emp)

    # Experiencia previa en la misma vertical que la licitación
    mismo = vert_his[:corte] == vert_lic[i]
    n_vertical = np.bincount(cod[mismo], minlength=n_emp)

    # Fecha de la última adjudicación de cada empresa
    ultima = np.full(n_emp, np.iinfo(np.int64).min)
    np.maximum.at(ultima, cod, fechas_his_i8[:corte])

    presentes = np.flatnonzero(n)
    top3_media = top3_suma[presentes] / top3_n[presentes]

    # Quedarse con las TOP_K mejores por similitud top-3
    if TOP_K and len(presentes) > TOP_K:
        sel = np.argpartition(-top3_media, TOP_K)[:TOP_K]
        presentes, top3_media = presentes[sel], top3_media[sel]

    res = pd.DataFrame({
        "id_licitacion": ids_lic[i],
        "vertical": nombres_vert[i],
        "fecha_licitacion": fechas_lic[i],
        "parent_uei": empresas[presentes],
        "similitud_top3": top3_media,
        "similitud_maxima": maxi[presentes],
        "similitud_media": suma[presentes] / n[presentes],
        "n_adj_previas": n[presentes],
        "n_adj_previas_vertical": n_vertical[presentes],
        "dias_desde_ultima": (fechas_lic[i].astype("int64") - ultima[presentes]) / NS_DIA,
    }).sort_values("similitud_top3", ascending=False)
    res["ranking"] = np.arange(1, len(res) + 1)
    resultados.append(res)

    if (i + 1) % 1000 == 0:
        print(f"  {i + 1:,}/{len(licitaciones):,} licitaciones procesadas")


"""
 Guardo los datos.
"""

if resultados:
    df_ranking = pd.concat(resultados, ignore_index=True)
    df_ranking.to_parquet(SALIDA, index=False)
    print(f"\n✅ {len(df_ranking):,} filas guardadas en {SALIDA}")
    print(df_ranking.head(20).to_string())
else:
    print("No hay licitaciones con historial previo.")

"""
 Para ver más o menos qué está dando el método de transformer, comparo con los
ganadores reales.
"""

def evaluar(df_ranking, ganadores, col_licitacion, col_empresa, ks=(1, 5, 10, 50)):
    """Hit@k: ¿qué fracción de licitaciones tiene a su ganador real entre las k primeras?
    Ajustá los nombres de columna a los de tu archivo de ganadores."""
    g = ganadores[[col_licitacion, col_empresa]].dropna().copy()
    g.columns = ["id_licitacion", "parent_uei"]
    g["parent_uei"] = g["parent_uei"].astype(str).str.strip()
    g["id_licitacion"] = g["id_licitacion"].astype(str)

    m = df_ranking.merge(g, on=["id_licitacion", "parent_uei"], how="inner")
    n_total = g["id_licitacion"].nunique()
    for k in ks:
        aciertos = m.loc[m["ranking"] <= k, "id_licitacion"].nunique()
        print(f"Hit@{k}: {aciertos / n_total:.1%}  ({aciertos:,}/{n_total:,})")

























