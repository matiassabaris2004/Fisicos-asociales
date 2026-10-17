"""
 Importo las librerias
"""

import os
import numpy as np
import pandas as pd
import duckdb as dd

#%%
"""
 Cargo todos los archivos .parquet de los datos proporcionados
"""

# Ruta donde guardaste los CSV
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
 Veo cuáles son los datos que tenemos en cada archivo
"""

#print(df_d_empresas.columns)
#print(df_d_ganadores.columns)
#print(df_d_historial.columns)
#print(df_d_licitaciones.columns)

#%%
"""
 Realizo unas primeras observaciones para los d
"""

licitaciones = df_d_licitaciones.copy()
empresas = df_d_empresas.copy()
historial = df_d_historial.copy()
ganadores = df_d_ganadores.copy()

#%%
#tablas = {
#    "Licitaciones": licitaciones,
#    "Empresas": empresas,
#    "Historial": historial,
#   "Ganadores": ganadores}

#for nombre, df in tablas.items():
#    print(f"\n{'=' * 20} {nombre} {'=' * 20}")
#    print("Dimensiones:", df.shape)
#    print("\nTipos de datos:")
#    print(df.dtypes)
#    print("\nValores faltantes:") 
#    print(df.isna().sum())

"""
En licitaciones tenemos filas faltantes en set_aside
En empresas hay filas faltantes de texto_awards
En historial todas menos fecha, parent_uei y psc
En ganadores no hay filas con nulls!
"""

#print("\nFilas duplicadas completas:", df.duplicated().sum())

#%%
# Revisar identificadores de licitaciones
#print("\nIDs de licitación duplicados:",
#    licitaciones["id"].duplicated().sum())
"""
 No hay IDs duplicados
"""

# Revisar cuántas empresas ganadoras tiene cada licitación
#ganadores_por_licitacion = (
#    ganadores.groupby("id")["parent_uei"]
#    .nunique())

#print("\nDistribución de cantidad de ganadores por licitación:")
#print(ganadores_por_licitacion.value_counts().sort_index())
"""
 Las licitaciones tienen un máximo de 3 ganadores (hay una sola con 3)
"""

# Revisar si hay ganadores que no aparecen en las licitaciones
#ids_licitaciones = set(licitaciones["id"].dropna())
#ids_ganadores = set(ganadores["id"].dropna())

#print("\nIDs de ganadores que no aparecen en licitaciones:",
#    len(ids_ganadores - ids_licitaciones))

"""
 Todos los ganadores ganan alguna licitación efectivamente.
"""

# Revisar fechas y valores de ejemplo
#print("\nPrimeros valores de t:")
#print(licitaciones["t"].head(10))

#print("\nTipo de la columna t:", licitaciones["t"].dtype)

#print("\nPrimeras fechas del historial:")
#print(historial["fecha"].head(10))

"""
 Ok, las fechas estan en el mismo formato de string...los paso en la siguiente
celda a la forma de datetime para comparar fechas y no caer en el "data leakage"
"""

#%%
"""
 Paso fechas y t a datetime
"""
df_d_licitaciones["t"] = pd.to_datetime(
    df_d_licitaciones["t"],
    format="%Y-%m-%d",
    errors="coerce")

df_d_historial["fecha"] = pd.to_datetime(
    df_d_historial["fecha"],
    format="%Y-%m-%d",
    errors="coerce")

# Comprobar la conversión
print(df_d_licitaciones["t"].dtype)
print(df_d_historial["fecha"].dtype)

# Comprobar si hubo fechas que no pudieron convertirse
print("Fechas inválidas en licitaciones:",
      df_d_licitaciones["t"].isna().sum())

print("Fechas inválidas en historial:",
      df_d_historial["fecha"].isna().sum())

#%%







#%%
"""
  Veo cuáles son los datos que tenemos en cada archivo
"""

licitaciones_it = df_it_licitaciones.copy()
empresas_it = df_it_empresas.copy()
historial_it = df_it_historial.copy()
ganadores_it = df_it_ganadores.copy()

#%%
tablas_it = {
    "Licitaciones IT": licitaciones_it,
    "Empresas IT": empresas_it,
    "Historial IT": historial_it,
    "Ganadores IT": ganadores_it}

for nombre, df in tablas_it.items():
    print(f"\n{'=' * 20} {nombre} {'=' * 20}")
    print("Dimensiones:", df.shape)
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores faltantes:") 
    print(df.isna().sum())

#%%
# Revisar identificadores de licitaciones IT
print("\nIDs de licitación duplicados (IT):",
      licitaciones_it["id"].duplicated().sum())

# Revisar cuántas empresas ganadoras tiene cada licitación en IT
ganadores_por_licitacion_it = (
    ganadores_it.groupby("id")["parent_uei"]
    .nunique())

print("\nDistribución de cantidad de ganadores por licitación (IT):")
print(ganadores_por_licitacion_it.value_counts().sort_index())

#%%
"""
Paso fechas y t a datetime para IT
"""
df_it_licitaciones["t"] = pd.to_datetime(
    df_it_licitaciones["t"],
    format="%Y-%m-%d",
    errors="coerce"
)

df_it_historial["fecha"] = pd.to_datetime(
    df_it_historial["fecha"],
    format="%Y-%m-%d",
    errors="coerce"
)

# Comprobar la conversión
print(df_it_licitaciones["t"].dtype)
print(df_it_historial["fecha"].dtype)

print("Fechas inválidas en licitaciones IT:",
      df_it_licitaciones["t"].isna().sum())
print("Fechas inválidas en historial IT:",
      df_it_historial["fecha"].isna().sum())

#%%










#%%
licitaciones_j = df_j_licitaciones.copy()
empresas_j = df_j_empresas.copy()
historial_j = df_j_historial.copy()
ganadores_j = df_j_ganadores.copy()

#%%
tablas_j = {
    "Licitaciones J": licitaciones_j,
    "Empresas J": empresas_j,
    "Historial J": historial_j,
    "Ganadores J": ganadores_j
}

for nombre, df in tablas_j.items():
    print(f"\n{'=' * 20} {nombre} {'=' * 20}")
    print("Dimensiones:", df.shape)
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores faltantes:") 
    print(df.isna().sum())

#%%
print("\nIDs de licitación duplicados (J):",
      licitaciones_j["id"].duplicated().sum())

ganadores_por_licitacion_j = (
    ganadores_j.groupby("id")["parent_uei"]
    .nunique()
)

print("\nDistribución de cantidad de ganadores por licitación (J):")
print(ganadores_por_licitacion_j.value_counts().sort_index())

#%%
"""
Paso fechas y t a datetime para J
"""
df_j_licitaciones["t"] = pd.to_datetime(
    df_j_licitaciones["t"],
    format="%Y-%m-%d",
    errors="coerce"
)

df_j_historial["fecha"] = pd.to_datetime(
    df_j_historial["fecha"],
    format="%Y-%m-%d",
    errors="coerce"
)

print(df_j_licitaciones["t"].dtype)
print(df_j_historial["fecha"].dtype)

print("Fechas inválidas en licitaciones J:",
      df_j_licitaciones["t"].isna().sum())
print("Fechas inválidas en historial J:",
      df_j_historial["fecha"].isna().sum())

#%%









#%%
licitaciones_r = df_r_licitaciones.copy()
empresas_r = df_r_empresas.copy()
historial_r = df_r_historial.copy()
ganadores_r = df_r_ganadores.copy()

#%%
tablas_r = {
    "Licitaciones R": licitaciones_r,
    "Empresas R": empresas_r,
    "Historial R": historial_r,
    "Ganadores R": ganadores_r
}

for nombre, df in tablas_r.items():
    print(f"\n{'=' * 20} {nombre} {'=' * 20}")
    print("Dimensiones:", df.shape)
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores faltantes:") 
    print(df.isna().sum())

#%%
print("\nIDs de licitación duplicados (R):",
      licitaciones_r["id"].duplicated().sum())

ganadores_por_licitacion_r = (
    ganadores_r.groupby("id")["parent_uei"]
    .nunique()
)

print("\nDistribución de cantidad de ganadores por licitación (R):")
print(ganadores_por_licitacion_r.value_counts().sort_index())

#%%
"""
Paso fechas y t a datetime para R
"""
df_r_licitaciones["t"] = pd.to_datetime(
    df_r_licitaciones["t"],
    format="%Y-%m-%d",
    errors="coerce"
)

df_r_historial["fecha"] = pd.to_datetime(
    df_r_historial["fecha"],
    format="%Y-%m-%d",
    errors="coerce"
)

print(df_r_licitaciones["t"].dtype)
print(df_r_historial["fecha"].dtype)

print("Fechas inválidas en licitaciones R:",
      df_r_licitaciones["t"].isna().sum())
print("Fechas inválidas en historial R:",
      df_r_historial["fecha"].isna().sum())

#%%









#%%
licitaciones_rd = df_rd_licitaciones.copy()
empresas_rd = df_rd_empresas.copy()
historial_rd = df_rd_historial.copy()
ganadores_rd = df_rd_ganadores.copy()

#%%
tablas_rd = {
    "Licitaciones RD": licitaciones_rd,
    "Empresas RD": empresas_rd,
    "Historial RD": historial_rd,
    "Ganadores RD": ganadores_rd
}

for nombre, df in tablas_rd.items():
    print(f"\n{'=' * 20} {nombre} {'=' * 20}")
    print("Dimensiones:", df.shape)
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores faltantes:") 
    print(df.isna().sum())

#%%
print("\nIDs de licitación duplicados (RD):",
      licitaciones_rd["id"].duplicated().sum())

ganadores_por_licitacion_rd = (
    ganadores_rd.groupby("id")["parent_uei"]
    .nunique()
)

print("\nDistribución de cantidad de ganadores por licitación (RD):")
print(ganadores_por_licitacion_rd.value_counts().sort_index())

#%%
"""
Paso fechas y t a datetime para RD
"""
df_rd_licitaciones["t"] = pd.to_datetime(
    df_rd_licitaciones["t"],
    format="%Y-%m-%d",
    errors="coerce"
)

df_rd_historial["fecha"] = pd.to_datetime(
    df_rd_historial["fecha"],
    format="%Y-%m-%d",
    errors="coerce"
)

print(df_rd_licitaciones["t"].dtype)
print(df_rd_historial["fecha"].dtype)

print("Fechas inválidas en licitaciones RD:",
      df_rd_licitaciones["t"].isna().sum())
print("Fechas inválidas en historial RD:",
      df_rd_historial["fecha"].isna().sum())


