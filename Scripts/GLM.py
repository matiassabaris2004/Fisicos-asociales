#bellísimo GLM que nos va a hacer la vida más feliz!!

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def preparar_features_glm(df_ranking, df_ganadores):
    """
    Combina la tabla de pares (licitación, empresa) con los labels reales y aplica transformaciones.
    """
    df = df_ranking.copy()
    
    # 1. Cruzar con el label real (1 si es ganador, 0 si no)
    ganadores_set = set(zip(df_ganadores["id"], df_ganadores["parent_uei"]))
    df["es_ganador"] = [
        1 if (lic_id, emp) in ganadores_set else 0 
        for lic_id, emp in zip(df["id_licitacion"], df["parent_uei"])
    ]
    
    # 2. Transformaciones logarítmicas y logit
    eps = 1e-6
    df["p_b0_clipped"] = np.clip(df["p_b0"], eps, 1 - eps)
    df["logit_p_b0"] = np.log(df["p_b0_clipped"] / (1 - df["p_b0_clipped"]))
    
    df["log_n_adj"] = np.log1p(df["n_adj_previas"])
    df["log_n_adj_vert"] = np.log1p(df["n_adj_previas_vertical"])
    df["log_dias_ultima"] = np.log1p(df["dias_desde_ultima"].fillna(3650))
    
    return df

def entrenar_glm(df_train):
    """
    Ajusta un modelo GLM con enlace Logit (Regresión Logística).
    """
    formula = (
        "es_ganador ~ logit_p_b0 + similitud_top3 + similitud_maxima + "
        "similitud_media + log_n_adj + log_n_adj_vert + log_dias_ultima"
    )
    
    # Ajustar GLM con familia Binomial
    modelo_glm = smf.glm(formula=formula, data=df_train, family=sm.families.Binomial()).fit()
    print(modelo_glm.summary())
    return modelo_glm

def generar_ranking_glm(modelo_glm, df_test):
    """
    Predice la probabilidad de ganar y genera el ranking Top 100 por licitación.
    """
    df_test = df_test.copy()
    df_test["score_predicho"] = modelo_glm.predict(df_test)
    
    # Ordenar por probabilidad predicha y seleccionar top 100
    df_test = df_test.sort_values(["id_licitacion", "score_predicho"], ascending=[True, False])
    df_test["puesto"] = df_test.groupby("id_licitacion").cumcount() + 1
    
    ranking_final = df_test[df_test["puesto"] <= 100][["id_licitacion", "parent_uei", "puesto"]]
    ranking_final.columns = ["id", "parent_uei", "puesto"]
    
    return ranking_final
