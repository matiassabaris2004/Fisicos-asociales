#carga tablas maestras de códigos en la raíz de data/ (psc.parquet, naics.parquet, ofice.parquet),
# fusiona con las licitaciones e historial para crear textos explicativos completos!!!

import pandas as pd
from pathlib import Path

def cargar_tablas_maestras(data_dir: Path):
    """Carga las tablas de traducción de códigos PSC, NAICS y Office."""
    psc = pd.read_parquet(data_dir / "psc.parquet")
    naics = pd.read_parquet(data_dir / "naics.parquet")
    office = pd.read_parquet(data_dir / "office.parquet")
    
    # Preparar diccionarios o maopeos rápidos
    psc_map = psc.set_index("codigo")["descripcion"].to_dict()
    naics_map = naics.set_index("codigo")["descripcion"].to_dict()
    
    # En office, combinamos nombre de oficina y subagencia
    office["desc_completa"] = (
        office["oficina"].fillna("") + " " + 
        office["subagencia"].fillna("") + " " + 
        office["departamento"].fillna("")
    ).str.strip()
    office_map = office.set_index("codigo")["desc_completa"].to_dict()
    
    return psc_map, naics_map, office_map

def enriquecer_licitaciones(df_lic: pd.DataFrame, psc_map: dict, naics_map: dict, office_map: dict) -> pd.DataFrame:
    """Combina título, descripción y descripciones formales de códigos."""
    df = df_lic.copy()
    
    desc_psc = df["psc"].map(psc_map).fillna("")
    desc_naics = df["naics"].map(naics_map).fillna("")
    desc_office = df["office"].map(office_map).fillna("")
    
    df["texto_completo"] = (
        df["title"].fillna("").astype(str) + ". " +
        df["description"].fillna("").astype(str) + " " +
        "Rubro: " + desc_psc + " " +
        "Industria: " + desc_naics + " " +
        "Oficina: " + desc_office
    ).str.strip()
    
    return df

def enriquecer_historial(df_his: pd.DataFrame, psc_map: dict, office_map: dict) -> pd.DataFrame:
    """Combina la descripción histórica con las traducciones de códigos."""
    df = df_his.copy()
    
    desc_psc = df["psc"].map(psc_map).fillna("")
    desc_office = df["office"].map(office_map).fillna("")
    
    df["texto_completo"] = (
        df["descripcion"].fillna("").astype(str) + " " +
        "Rubro: " + desc_psc + " " +
        "Oficina: " + desc_office
    ).str.strip()
    
    return df
