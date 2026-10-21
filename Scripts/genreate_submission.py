#pipeline para todas las verticales, exporta rankings en CSV y ejecuta evaluación automática!!
from pathlib import Path
import pandas as pd
import b0
from src.text_enricher import cargar_tablas_maestras, enriquecer_licitaciones, enriquecer_historial
from src.feature_extractor import FeatureExtractor

DATA_DIR = Path("data")
VERTICALES = ["d", "it", "j", "r", "rd"]

def main():
    print("🚀 Cargando tablas maestras de datos...")
    psc_map, naics_map, office_map = cargar_tablas_maestras(DATA_DIR)
    
    extractor = FeatureExtractor()
    
    for vert in VERTICALES:
        vert_dir = DATA_DIR / vert
        if not vert_dir.exists():
            print(f"⚠️ Salteando vertical {vert}: no se encuentra la carpeta.")
            continue
            
        print(f"\n================ Procesando Vertical: {vert.upper()} ================")
        lic = pd.read_parquet(vert_dir / "licitaciones.parquet")
        his = pd.read_parquet(vert_dir / "historial.parquet")
        emp = pd.read_parquet(vert_dir / "empresas.parquet")
        
        # Enrich text
        lic = enriquecer_licitaciones(lic, psc_map, naics_map, office_map)
        his = enriquecer_historial(his, psc_map, office_map)
        
        # Modelo B0 oficial para el prior
        b0_model = b0.B0(his)
        
        # Generar rankings
        df_ranking = extractor.extract_features_for_vertical(lic, his, emp, b0_model)
        
        salida_path = f"ranking_{vert}.csv"
        df_ranking.to_csv(salida_path, index=False)
        print(f"✅ Guardado {salida_path} con {len(df_ranking):,} filas.")
        print(f"👉 Para evaluar corré: uv run evaluar.py {salida_path} --datos data/{vert}")

if __name__ == "__main__":
    main()
