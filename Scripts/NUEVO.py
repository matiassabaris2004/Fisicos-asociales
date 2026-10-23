# ==============================================================================
# PIPELINE INTEGRAL DE RANKING DE LICITACIONES CON LIGHTGBM Y TRANSFORMERS
# ==============================================================================

import os
import warnings
import numpy as np
import pandas as pd

# Modelos y Métricas
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

# Importar el módulo oficial de B0 proporcionado en el repositorio
import b0

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y CONSTANTES
# ------------------------------------------------------------------------------
DATA_DIR = r"data"  # Ruta a la carpeta data/ con las 5 verticales
VERTICAL_EVAL = "j"  # Vertical a procesar/evaluar (ej: 'j', 'd', 'it', 'r', 'rd')
TOP_K_CANDIDATAS = 300  # Embudo amplio para maximizar Recall@100
TOP_K_SALIDA = 100       # Exigencia final de evaluar.py
FECHA_CORTE_VAL = pd.to_datetime("2024-01-01") # Split temporal de validación

# ------------------------------------------------------------------------------
# 2. ENRIQUECIMIENTO DE TEXTO EN INGLÉS (text_enricher.py corregido)
# ------------------------------------------------------------------------------
def cargar_tablas_maestras(data_path):
    """Carga y mapea las descripciones formales de PSC, NAICS y Office."""
    psc = pd.read_parquet(os.path.join(data_path, "psc.parquet"))
    naics = pd.read_parquet(os.path.join(data_path, "naics.parquet"))
    office = pd.read_parquet(os.path.join(data_path, "office.parquet"))

    psc_map = psc.set_index("codigo")["descripcion"].to_dict()
    naics_map = naics.set_index("codigo")["descripcion"].to_dict()

    office["desc"] = (
        office["oficina"].fillna("") + " " +
        office["subagencia"].fillna("") + " " +
        office["departamento"].fillna("")
    ).str.strip()
    office_map = office.set_index("codigo")["desc"].to_dict()

    return psc_map, naics_map, office_map

def enriquecer_textos_ingles(df_lic, df_his, psc_map, naics_map, office_map):
    """Genera texto explicativo formateado íntegramente en inglés."""
    lic = df_lic.copy().reset_index(drop=True)
    his = df_his.copy().reset_index(drop=True)

    # Licitaciones
    desc_psc_lic = lic["psc"].map(psc_map).fillna("")
    desc_naics_lic = lic["naics"].map(naics_map).fillna("")
    desc_off_lic = lic["office"].map(office_map).fillna("")

    lic["texto_completo"] = (
        lic["title"].fillna("").astype(str) + ". " +
        lic["description"].fillna("").astype(str) + ". " +
        "Category: " + desc_psc_lic + ". " +
        "Industry: " + desc_naics_lic + ". " +
        "Office: " + desc_off_lic
    ).str.strip()

    # Historial
    desc_psc_his = his["psc"].map(psc_map).fillna("")
    desc_off_his = his["office"].map(office_map).fillna("")

    his["texto_completo"] = (
        his["descripcion"].fillna("").astype(str) + ". " +
        "Category: " + desc_psc_his + ". " +
        "Office: " + desc_off_his
    ).str.strip()

    return lic, his

# ------------------------------------------------------------------------------
# 3. EXTRACCIÓN AVANZADA DE FEATURES (feature_extractor.py con Afinidad & Texto)
# ------------------------------------------------------------------------------
class FeatureExtractorCompleto:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"📦 Cargando modelo de Transformer: {model_name}...")
        self.transformer = SentenceTransformer(model_name)

    def extraer_matriz_features(self, df_lic, df_his, df_emp, b0_model, df_ganadores=None):
        """
        Genera la matriz completa de features por cada par (licitacion, empresa candidata).
        Garantiza no-data leakage (fecha < t) e índices perfectamente alineados.
        """
        df_lic = df_lic.reset_index(drop=True)
        df_his = df_his.reset_index(drop=True)

        # Garantizar conversion datetime
        df_lic["t"] = pd.to_datetime(df_lic["t"])
        df_his["fecha"] = pd.to_datetime(df_his["fecha"])

        # Generar embeddings de texto
        print("⚡ Calculando embeddings densos de Transformer...")
        emb_lic = self.transformer.encode(
            df_lic["texto_completo"].tolist(),
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=True
        )
        emb_his = self.transformer.encode(
            df_his["texto_completo"].tolist(),
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=True
        )

        # Ajustar modelo de TF-IDF léxico estático
        print("🔤 Generando matriz TF-IDF (esparcida)...")
        tfidf = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
        tfidf.fit(pd.concat([df_lic["texto_completo"], df_his["texto_completo"]]))
        tfidf_lic = tfidf.transform(df_lic["texto_completo"])
        tfidf_his = tfidf.transform(df_his["texto_completo"])

        # Mapeo rápido de ganadores reales si están disponibles
        ganadores_set = set()
        if df_ganadores is not None:
            ganadores_set = set(zip(df_ganadores["id"], df_ganadores["parent_uei"]))

        filas_dataset = []
        fechas_his = df_his["fecha"].values
        NS_A_DIAS = 86400 * 1e9

        print(f"🔄 Extrayendo features para {len(df_lic):,} licitaciones...")
        for i, lic in df_lic.iterrows():
            t_q = lic["t"]
            lic_id = lic["id"]
            psc_q = lic.get("psc", "")
            naics_q = str(lic.get("naics", ""))
            office_q = lic.get("office", "")

            # 1. Candidatas oficiales autorizadas para el semestre
            semestre = f"{t_q.year}S{1 if t_q.month <= 6 else 2}"
            candidatas_sem = df_emp[df_emp["semestre"] == semestre]["parent_uei"].unique()
            if len(candidatas_sem) == 0:
                candidatas_sem = df_emp["parent_uei"].unique()

            # 2. Prior Probabilístico B0
            p_b0_series = b0_model.p_b0(lic, candidatas_sem)

            # 3. Filtrar estricto del historial anterior a t_q (No Data Leakage)
            mask_past = fechas_his < t_q.to_datetime64()
            if not mask_past.any():
                # Sin historial previo: asignación base
                for emp in candidatas_sem[:TOP_K_CANDIDATAS]:
                    filas_dataset.append({
                        "id_licitacion": lic_id, "parent_uei": emp, "t": t_q,
                        "p_b0": p_b0_series.get(emp, 0.0),
                        "similitud_top3": 0.0, "similitud_maxima": 0.0, "similitud_media": 0.0,
                        "sim_tfidf_max": 0.0, "n_adj_previas": 0, "dias_desde_ultima": 3650.0,
                        "n_adj_mismo_psc": 0, "n_adj_mismo_naics": 0, "n_adj_misma_office": 0,
                        "dias_ultima_misma_office": 3650.0, "es_ganador": 1 if (lic_id, emp) in ganadores_set else 0
                    })
                continue

            his_past = df_his.iloc[mask_past]
            emb_his_past = emb_his[mask_past]
            tfidf_his_past = tfidf_his[mask_past]

            # 4. Cálculo de Similitudes Semánticas Puras (sin acoplar recencia)
            sim_trans = emb_his_past @ emb_lic[i]  # Alineación exacta por índice i
            sim_tfidf = tfidf_his_past.dot(tfidf_lic[i]).toarray().ravel()

            # Datagrama de trabajo con el historial de esta licitación
            his_work = pd.DataFrame({
                "parent_uei": his_past["parent_uei"],
                "fecha": his_past["fecha"],
                "psc": his_past["psc"],
                "office": his_past["office"],
                "sim_trans": sim_trans,
                "sim_tfidf": sim_tfidf
            })

            # 5. Afinidad Exacta y Métrica Temporal
            his_work["mismo_psc"] = (his_work["psc"] == psc_q).astype(int)
            his_work["misma_office"] = (his_work["office"] == office_q).astype(int)

            # Agrupar estadísticas por empresa
            grp = his_work.groupby("parent_uei")
            
            # Top-3 Similitud promedio
            top3_sim = grp["sim_trans"].apply(lambda x: x.nlargest(3).mean())
            max_sim = grp["sim_trans"].max()
            mean_sim = grp["sim_trans"].mean()
            max_tfidf = grp["sim_tfidf"].max()
            
            n_adj = grp.size()
            ultima_fecha = grp["fecha"].max()
            dias_ultima = (t_q - ultima_fecha).dt.total_seconds() / 86400.0

            n_mismo_psc = grp["mismo_psc"].sum()
            n_misma_office = grp["misma_office"].sum()

            # Construir DataFrame de Candidatas
            df_cand = pd.DataFrame({"parent_uei": candidatas_sem}).set_index("parent_uei")
            df_cand["p_b0"] = p_b0_series
            df_cand["similitud_top3"] = top3_sim
            df_cand["similitud_maxima"] = max_sim
            df_cand["similitud_media"] = mean_sim
            df_cand["sim_tfidf_max"] = max_tfidf
            df_cand["n_adj_previas"] = n_adj
            df_cand["dias_desde_ultima"] = dias_ultima
            df_cand["n_adj_mismo_psc"] = n_mismo_psc
            df_cand["n_adj_misma_office"] = n_misma_office

            # Llenar valores nulos para candidatas sin historial en este rubro
            df_cand = df_cand.fillna({
                "p_b0": 0.0, "similitud_top3": 0.0, "similitud_maxima": 0.0,
                "similitud_media": 0.0, "sim_tfidf_max": 0.0, "n_adj_previas": 0,
                "dias_desde_ultima": 3650.0, "n_adj_mismo_psc": 0, "n_adj_misma_office": 0
            })

            # Pre-ordenar por heurística combinada y recortar a TOP_K_CANDIDATAS
            df_cand["score_preliminar"] = df_cand["p_b0"] + 0.1 * df_cand["similitud_top3"]
            df_cand = df_cand.sort_values("score_preliminar", ascending=False).head(TOP_K_CANDIDATAS).reset_index()

            for _, row in df_cand.iterrows():
                emp = row["parent_uei"]
                filas_dataset.append({
                    "id_licitacion": lic_id,
                    "parent_uei": emp,
                    "t": t_q,
                    "p_b0": row["p_b0"],
                    "similitud_top3": row["similitud_top3"],
                    "similitud_maxima": row["similitud_maxima"],
                    "similitud_media": row["similitud_media"],
                    "sim_tfidf_max": row["sim_tfidf_max"],
                    "n_adj_previas": row["n_adj_previas"],
                    "dias_desde_ultima": row["dias_desde_ultima"],
                    "n_adj_mismo_psc": row["n_adj_mismo_psc"],
                    "n_adj_misma_office": row["n_adj_misma_office"],
                    "es_ganador": 1 if (lic_id, emp) in ganadores_set else 0
                })

        return pd.DataFrame(filas_dataset)

# ------------------------------------------------------------------------------
# 4. EJECUCIÓN DEL PIPELINE Y VALIDACIÓN TEMPORAL (Jupyter Flow)
# ------------------------------------------------------------------------------
print(f"🚀 Iniciando Pipeline para la vertical '{VERTICAL_EVAL.upper()}'...")

# A. Cargar datos de la vertical
vert_path = os.path.join(DATA_DIR, VERTICAL_EVAL)
df_lic = pd.read_parquet(os.path.join(vert_path, "licitaciones.parquet"))
df_his = pd.read_parquet(os.path.join(vert_path, "historial.parquet"))
df_emp = pd.read_parquet(os.path.join(vert_path, "empresas.parquet"))
df_gan = pd.read_parquet(os.path.join(vert_path, "ganadores.parquet"))

# B. Cargar maestras y enriquecer textos
psc_map, naics_map, office_map = cargar_tablas_maestras(DATA_DIR)
df_lic, df_his = enriquecer_textos_ingles(df_lic, df_his, psc_map, naics_map, office_map)

# C. Instanciar B0 oficial
b0_model = b0.B0(df_his)

# D. Extraer matriz completa de features
extractor = FeatureExtractorCompleto()
dataset_completo = extractor.extraer_matriz_features(df_lic, df_his, df_emp, b0_model, df_gan)

# E. Split Temporal de Validación (Train: < 2024-01-01 | Val: >= 2024-01-01)
train_df = dataset_completo[dataset_completo["t"] < FECHA_CORTE_VAL].reset_index(drop=True)
val_df = dataset_completo[dataset_completo["t"] >= FECHA_CORTE_VAL].reset_index(drop=True)

print(f"📊 Dataset de Entrenamiento: {len(train_df):,} filas | Validación: {len(val_df):,} filas")

# F. Entrenar Reranker LightGBM
features_cols = [
    "p_b0", "similitud_top3", "similitud_maxima", "similitud_media",
    "sim_tfidf_max", "n_adj_previas", "dias_desde_ultima",
    "n_adj_mismo_psc", "n_adj_misma_office"
]

X_train, y_train = train_df[features_cols], train_df["es_ganador"]
X_val, y_val = val_df[features_cols], val_df["es_ganador"]

print("🔥 Entrenando modelo LightGBM Reranker...")
model_lgb = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_lgb.fit(X_train, y_train)

# G. Predicción y Selección de Top 100 Final
val_df["score_pred"] = model_lgb.predict_proba(X_val)[:, 1]

# Ordenar por licitación y probabilidad predicha
val_df = val_df.sort_values(["id_licitacion", "score_pred"], ascending=[True, False])
val_df["puesto"] = val_df.groupby("id_licitacion").cumcount() + 1

# Recortar estrictamente al Top 100 exigido por evaluar.py
ranking_final_val = val_df[val_df["puesto"] <= TOP_K_SALIDA][["id_licitacion", "parent_uei", "puesto"]]
ranking_final_val.columns = ["id", "parent_uei", "puesto"]

# Exportar archivo oficial de validación
archivo_salida = f"ranking_lgb_{VERTICAL_EVAL}.csv"
ranking_final_val.to_csv(archivo_salida, index=False)
print(f"✅ Ranking de validación exportado exitosamente a '{archivo_salida}'")
print("\n👉 Para evaluar métricas oficiales corré en la consola:")
print(f"uv run evaluar.py {archivo_salida} --datos data/{VERTICAL_EVAL} --b0 b0_{VERTICAL_EVAL}.csv")
