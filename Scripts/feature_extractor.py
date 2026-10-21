#calcula features clave para cada par! (licitación-empresa candidate!): 
#p_{b0}, similitud del transformer (ponderada por decaimiento temporal) y similityd de TF-IDF

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import b0

class FeatureExtractor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.transformer = SentenceTransformer(model_name)

    def extract_features_for_vertical(self, df_lic, df_his, df_emp, p_b0_model):
        """Genera rankings combinando B0, Transformer y TF-IDF."""
        # Asegurar fechas datetime
        df_lic["t"] = pd.to_datetime(df_lic["t"])
        df_his["fecha"] = pd.to_datetime(df_his["fecha"])
        
        # 1. Calculemos embeddings del Transformer
        emb_lic = self.transformer.encode(
            df_lic["texto_completo"].tolist(), 
            normalize_embeddings=True, 
            batch_size=128, 
            show_progress_bar=False
        )
        emb_his = self.transformer.encode(
            df_his["texto_completo"].tolist(), 
            normalize_embeddings=True, 
            batch_size=128, 
            show_progress_bar=False
        )
        
        # 2. Matriz TF-IDF
        tfidf = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
        all_texts = pd.concat([df_lic["texto_completo"], df_his["texto_completo"]])
        tfidf.fit(all_texts)
        
        tfidf_lic = tfidf.transform(df_lic["texto_completo"])
        tfidf_his = tfidf.transform(df_his["texto_completo"])
        
        rankings = []
        fechas_his = df_his["fecha"].values
        
        # Iterar sobre cada licitación
        for i, lic in df_lic.iterrows():
            t_q = lic["t"]
            lic_id = lic["id"]
            
            # Obtener candidatas válidas del semestre desde empresas.parquet
            # Determinar el semestre
            semestre = f"{t_q.year}S{1 if t_q.month <= 6 else 2}"
            candidatas_sem = df_emp[df_emp["semestre"] == semestre]["parent_uei"].unique()
            
            if len(candidatas_sem) == 0:
                candidatas_sem = df_emp["parent_uei"].unique()
                
            # Extraer baseline B0 para todas las candidatas del semestre
            p_b0_series = p_b0_model.p_b0(lic, candidatas_sem)
            
            # Filtrar historial strictly anterior a t_q (no data leakage)
            mask_past = fechas_his < t_q.to_datetime64()
            if not mask_past.any():
                # Si no hay historial anterior, ordenar según B0
                df_res = pd.DataFrame({
                    "id": lic_id,
                    "parent_uei": candidatas_sem,
                    "score": p_b0_series.values
                }).sort_values("score", ascending=False).head(100)
                df_res["puesto"] = np.arange(1, len(df_res) + 1)
                rankings.append(df_res[["id", "parent_uei", "puesto"]])
                continue
            
            # Subconjunto de historial válido
            his_past = df_his.iloc[mask_past]
            emb_his_past = emb_his[mask_past]
            tfidf_his_past = tfidf_his[mask_past]
            
            # Decaimiento temporal tau = 2.5 años (en días: 365.25 * 2.5 = 913.125)
            dias_antiguedad = (t_q - his_past["fecha"]).dt.total_seconds() / 86400.0
            weights = np.exp(-dias_antiguedad / 913.125).values
            
            # Similitudes
            sim_trans = (emb_his_past @ emb_lic[i]) * weights
            sim_tfidf = tfidf_his_past.dot(tfidf_lic[i]).toarray().ravel() * weights
            
            # Agrupar por empresa (parent_uei)
            his_past_df = pd.DataFrame({
                "parent_uei": his_past["parent_uei"],
                "sim_trans": sim_trans,
                "sim_tfidf": sim_tfidf
            })
            
            # Tomar el máximo/promedio top por empresa
            grp = his_past_df.groupby("parent_uei").agg(
                sim_trans_max=("sim_trans", "max"),
                sim_tfidf_max=("sim_tfidf", "max")
            )
            
            # Armar tabla de candidatos con B0
            df_cand = pd.DataFrame({"parent_uei": candidatas_sem}).set_index("parent_uei")
            df_cand["p_b0"] = p_b0_series
            df_cand = df_cand.join(grp).fillna(0.0)
            
            # Score combinado final: Ensamblado B0 + Texto
            # B0 cubre S_inc, mientras que la similitud de texto rescata el grupo S_new
            df_cand["score_final"] = (
                0.50 * df_cand["p_b0"] + 
                0.35 * df_cand["sim_trans_max"] + 
                0.15 * df_cand["sim_tfidf_max"]
            )
            
            # Ordenar y seleccionar Top 100
            df_cand = df_cand.sort_values("score_final", ascending=False).head(100).reset_index()
            df_cand["id"] = lic_id
            df_cand["puesto"] = np.arange(1, len(df_cand) + 1)
            
            rankings.append(df_cand[["id", "parent_uei", "puesto"]])
            
        return pd.concat(rankings, ignore_index=True)
