###CHEQUEAR QUE NO HAYA LEAKAGE!!!!
#Auditoría con caché:
import pandas as pd
for v in ["d", "it", "j", "r", "rd"]:
    ds = pd.read_parquet(f"cache/ds_{v}_v3_150_150.parquet")
    x = ds[ds.t >= "2024-01-01"]
    w, n = x[x.es_ganador == 1], x[x.es_ganador == 0]
    print(v, "| sim_max>0.95 ganadores:", round((w.sim_max > 0.95).mean(), 3),
          "no ganadores:", round((n.sim_max > 0.95).mean(), 4),
          "| mediana sim_max ganadores:", round(w.sim_max.median(), 3))

#Auditorpia fechas!!!
import pandas as pd

def norm(s):
    return s.fillna("").astype(str).str.lower().str.replace(r"\W+", " ", regex=True).str.strip()

for v in ["d", "it", "j", "r", "rd"]:
    lic = pd.read_parquet(f"data/{v}/licitaciones.parquet")
    gan = pd.read_parquet(f"data/{v}/ganadores.parquet")
    his = pd.read_parquet(f"data/{v}/historial.parquet")
    lic["t"] = pd.to_datetime(lic["t"]); his["fecha"] = pd.to_datetime(his["fecha"])
    col = "descripcion" if "descripcion" in his else "description"
    lic["k"] = norm(lic["description"]); his["k"] = norm(his[col])
    val = lic[lic.t >= "2024-01-01"][["id", "t", "k"]].merge(gan[["id", "parent_uei"]], on="id")
    val = val[val.k.str.len() > 20]
    m = val.merge(his[["parent_uei", "fecha", "k"]], on=["parent_uei", "k"])
    m["dias"] = (m["t"] - m["fecha"]).dt.days
    previos = m[m.dias > 0].groupby("id").dias.min()      # el contrato previo igual más cercano
    mismo_dia = m[m.dias <= 0]                             # contratos iguales con fecha >= t (premio propio, esperable)
    print(f"\n{v}: ganadores val con texto exacto previo: {len(previos)}/{val.id.nunique()}"
          f" | con fecha >= t (premio propio): {mismo_dia.id.nunique()}")
    print(previos.describe().round(0).to_string())
    print("previos a <=30 días de t:", int((previos <= 30).sum()))


#LARGO!!!
# ==============================================================================
# MEJORAS SOBRE LO YA CALCULADO  (no re-extrae features, no usa embeddings ni data/)
# Entrada: cache/ds_<v>_v3_150_150.parquet  (la tabla de features que ya generó el script anterior)
#          mi_val_<v>.parquet y b0_val_<v>.csv  (los que escribió correr())
# Correr por celdas en Spyder (#%%) o entero. Tarda minutos en CPU.
#
# Qué arregla / agrega respecto de la corrida anterior:
#   1. Vuelve a poner NaN donde antes se rellenó con 0 ("sin dato" != "nada parecido").
#   2. Features relativas DENTRO de cada licitación (ranking y brecha al mejor) para casi todo.
#   3. Split con EMBARGO (fit | inner | val) y early stopping por MRR (no NDCG@10).
#   4. Busca objetivo/hiperparámetros (lambdarank, rank_xendcg, binary) SOLO en el inner.
#   5. Promedia varias semillas (bagging) + GLM + blend con peso elegido en el inner.
#   6. Ablación justa: "sin texto" sobre el pool de B0 puro (el anterior recibía ayuda textual del pool).
#   7. IC bootstrap pareado, S_inc/S_new, aporte del texto con IC, importancias por familia (SHAP).
# ==============================================================================
import pathlib
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

#%% 0. Config ------------------------------------------------------------------
# Una entrada por vertical que tengan en cache. Con >1 vertical se entrena un modelo POOLED.
DS = {v: f"cache/ds_{v}_v3_150_150.parquet" for v in ["rd", "d", "j", "r", "it"]}
VAL = {v: (f"mi_val_{v}.parquet", f"b0_val_{v}.csv") for v in ["rd", "d", "j", "r", "it"]}

VAL_INI = pd.Timestamp("2024-01-01")
EMBARGO_DIAS = 30
INNER_MESES = 6          # ventana interna (early stopping, hiperparámetros, peso del blend)
K_B0 = 150               # tamaño del pool "B0 puro" para la ablación
TOP = 100
SEEDS = [0, 1, 2, 3, 4]
MAX_RONDAS = 2000
PACIENCIA = 100
N_BOOT = 2000
OUT = pathlib.Path("outputs_mejora"); OUT.mkdir(exist_ok=True)

META = ["id_lic", "parent_uei", "t", "semestre", "es_ganador", "vertical", "qid", "rol", "pool_b0"]
TEXTO_BASE = ["sim_max", "sim_mean", "sim_top3", "sim_off_max", "sim_psc_max", "simrec_max",
              "tfidf_max", "tfidf_top3", "tfidf_mean", "sim_top3_rank", "sim_top3_gap"]
REL_TEXTO = ["sim_max", "sim_mean", "sim_off_max", "sim_psc_max", "simrec_max", "tfidf_max", "tfidf_top3"]
REL_INC_DESC = ["n_psc_off", "n_off", "n_psc", "n_prev"]      # más es mejor
REL_INC_ASC = ["dias_ult"]                                    # menos es mejor
TEXTO = set(TEXTO_BASE) | {f"rk_{c}" for c in REL_TEXTO} | {f"gp_{c}" for c in REL_TEXTO}


#%% 1. Preparación de features (reutilizable para licitaciones nuevas) -----------
def preparar(ds, v):
    ds = ds.copy()
    ds["vertical"] = v
    ds["qid"] = v + "|" + ds["id_lic"].astype(str)
    ds["t"] = pd.to_datetime(ds["t"])
    # (1) "sin dato" -> NaN (antes 0.0 / 3650)
    sin = ds["n_prev"] == 0
    for c in ["sim_max", "sim_mean", "sim_top3", "simrec_max", "tfidf_max", "tfidf_top3", "tfidf_mean", "dias_ult"]:
        if c in ds:
            ds.loc[sin, c] = np.nan
    ds.loc[ds["n_off"] == 0, "sim_off_max"] = np.nan
    ds.loc[ds["n_psc"] == 0, "sim_psc_max"] = np.nan
    ds["tiene_hist"] = (~sin).astype(np.float32)
    # (2) features relativas dentro de la licitación
    g = ds.groupby("qid", sort=False)
    for c in REL_TEXTO:
        ds[f"rk_{c}"] = g[c].rank(pct=True, ascending=False)
        ds[f"gp_{c}"] = ds[c] - g[c].transform("max")
    for c in REL_INC_DESC:
        ds[f"rk_{c}"] = g[c].rank(pct=True, ascending=False)
    for c in REL_INC_ASC:
        ds[f"rk_{c}"] = g[c].rank(pct=True, ascending=True)
    # pool "B0 puro": top-K_B0 por p_b0, desempate SIN texto (n_prev)
    o = ds.sort_values(["qid", "p_b0", "n_prev"], ascending=[True, False, False], kind="stable")
    pos = o.groupby("qid", sort=False).cumcount()
    ds["pool_b0"] = False
    ds.loc[o.index[pos.values < K_B0], "pool_b0"] = True
    return ds


def roles(t):
    emb = pd.Timedelta(days=EMBARGO_DIAS)
    train_fin = VAL_INI - emb
    inner_ini = train_fin - pd.DateOffset(months=INNER_MESES)
    fit_fin = inner_ini - emb
    return np.select([t < fit_fin, (t >= inner_ini) & (t < train_fin), t >= VAL_INI],
                     ["fit", "inner", "val"], default="gap")


#%% 2. Métricas (réplica de evaluar.py) ----------------------------------------
def top_k(d, col, k=TOP):
    d = d.sort_values(["id_lic", col], ascending=[True, False], kind="stable")
    r = d[["id_lic", "parent_uei"]].copy()
    r["puesto"] = d.groupby("id_lic", sort=False).cumcount().values + 1
    return r[r.puesto <= k].rename(columns={"id_lic": "id"})


def por_lic(rank, gan):
    pos = dict(zip(zip(rank["id"], rank["parent_uei"]), rank["puesto"]))
    filas = {}
    for lid, gs in gan.groupby("id"):
        r = sorted(pos.get((lid, u), np.inf) for u in gs["parent_uei"])
        adj = [x - k for k, x in enumerate(r)]
        filas[lid] = (np.mean([1 / a if np.isfinite(a) else 0 for a in adj]), r[0] <= 1, r[0] <= 5, r[0])
    return pd.DataFrame(filas, index=["mrr", "h1", "h5", "best"]).T.astype(float)


def zdentro(s, qid):
    g = s.groupby(qid)
    return (s - g.transform("mean")) / (g.transform("std").fillna(1) + 1e-9)


def ic(a, b, n=N_BOOT, seed=0):
    d = (a - b).values
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


#%% 3. LightGBM ranker -----------------------------------------------------------
def grp(d):
    return d.groupby("qid", sort=False).size().values


def contiguo(d):
    return int((d["qid"] != d["qid"].shift()).sum()) == d["qid"].nunique()


def con_pos(d):
    return d[d.groupby("qid")["es_ganador"].transform("max") > 0]


def mk(d, cols, ref=None):
    assert contiguo(d), "las filas de una licitación deben estar juntas"
    return lgb.Dataset(d[cols].astype(np.float32), label=d["es_ganador"].values, group=grp(d),
                       reference=ref, free_raw_data=False)


def mrr_feval(preds, ds):
    y, g = ds.get_label(), ds.get_group()
    ini, rr = 0, []
    for n in g:
        n = int(n)
        o = np.argsort(-preds[ini:ini + n], kind="stable")
        p = np.flatnonzero(y[ini:ini + n][o] > 0)
        rr.append(1.0 / (p[0] + 1) if len(p) else 0.0)
        ini += n
    return "mrr", float(np.mean(rr)), True


def params_base(seed=42):
    return dict(learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                verbose=-1, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, num_threads=-1,
                metric="None", force_row_wise=True)


GRILLA = [dict(objective=o, num_leaves=nl, min_data_in_leaf=md) |
          (dict(label_gain=[0, 1], lambdarank_truncation_level=30) if o == "lambdarank" else {})
          for o in ("lambdarank", "rank_xendcg", "binary") for nl, md in ((15, 50), (63, 100))]


def buscar(fit, inner, cols, etiqueta=""):
    """Elige objetivo/hiperparámetros por MRR en el inner (nunca mira val)."""
    fit, inner = con_pos(fit), con_pos(inner)
    dfit = mk(fit, cols)
    dinn = mk(inner, cols, ref=dfit)
    mejor = None
    for g in GRILLA:
        p = params_base() | g
        b = lgb.train(p, dfit, MAX_RONDAS, valid_sets=[dinn], feval=mrr_feval,
                      callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)])
        s = b.best_score["valid_0"]["mrr"]
        print(f"   [{etiqueta}] {g['objective']:12s} hojas={g['num_leaves']:3d}  MRR_inner={s:.4f}  iters={b.best_iteration}")
        if mejor is None or s > mejor[0]:
            mejor = (s, p, max(50, b.best_iteration))
    return mejor[1], mejor[2]


def entrenar_bag(d, cols, params, n_iter, seeds=SEEDS):
    d = con_pos(d)
    ds_ = mk(d, cols)
    return [lgb.train(params | dict(seed=s, bagging_seed=s, feature_fraction_seed=s), ds_, int(n_iter))
            for s in seeds]


def predecir_bag(bag, d, cols):
    X = d[cols].astype(np.float32)
    z = [zdentro(pd.Series(b.predict(X), index=d.index), d["qid"]) for b in bag]
    return sum(z) / len(z)


#%% 4. GLM (regresión logística estandarizada) ----------------------------------
def lr_X(d, cols):
    X = d[cols].astype(np.float64).copy()
    for c in cols:
        if c.startswith("P_") or c == "p_b0":
            X[c] = np.log(X[c] + 1e-4)
        elif c.startswith("n_") or c in ("dias_ult",):
            X[c] = np.log1p(X[c].clip(lower=0))
    return X


def entrenar_lr(d, cols, C=0.3):
    d = con_pos(d)
    pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, max_iter=1000))
    return pipe.fit(lr_X(d, cols), d["es_ganador"].values)


def score_lr(pipe, d, cols):
    return pd.Series(pipe.decision_function(lr_X(d, cols)), index=d.index)


#%% 5. Carga y preparación -------------------------------------------------------
partes = [preparar(pd.read_parquet(p), v) for v, p in DS.items()]
ds = pd.concat(partes, ignore_index=True)
if len(DS) > 1:
    ds["vertical_cod"] = ds["vertical"].map({v: i for i, v in enumerate(DS)}).astype(np.float32)
ds["rol"] = roles(ds["t"])
FEATS = [c for c in ds.columns if c not in META]
FEATS_SIN_TXT = [c for c in FEATS if c not in TEXTO]
print(f"filas {len(ds):,} | licitaciones {ds.qid.nunique():,} | features {len(FEATS)} "
      f"(texto {len(FEATS) - len(FEATS_SIN_TXT)}) | roles:\n{ds.groupby('rol').qid.nunique()}")

# Cobertura del pool en val (techo de Hit@100) con los ganadores REALES
gan_val, b0_val, sc_b0 = {}, {}, {}
for v, (gp, bp) in VAL.items():
    gan_val[v] = pd.read_parquet(gp)
    b0_val[v] = pd.read_csv(bp)
    sc_b0[v] = por_lic(b0_val[v], gan_val[v])
    en_pool = ds[(ds.vertical == v) & (ds.rol == "val") & (ds.es_ganador == 1)].id_lic.nunique()
    print(f"[{v}] cobertura del pool en val (techo Hit@100): {en_pool / gan_val[v].id.nunique():.3f}")


#%% 6. Entrenar: modelo completo y ablaciones ----------------------------------
fit, inner = ds[ds.rol == "fit"], ds[ds.rol == "inner"]
trn, val = ds[ds.rol.isin(["fit", "inner"])], ds[ds.rol == "val"].copy()


def correr_modelo(cols, filas_ok=None, etiqueta=""):
    """filas_ok: máscara booleana para restringir el pool (ablación B0 puro)."""
    f, i, t, vv = fit, inner, trn, val
    if filas_ok is not None:
        f, i, t, vv = f[f[filas_ok]], i[i[filas_ok]], t[t[filas_ok]], vv[vv[filas_ok]]
    params, n_iter = buscar(f, i, cols, etiqueta)
    bag = entrenar_bag(t, cols, params, round(n_iter * 1.1))
    return dict(params=params, n_iter=n_iter, bag=bag, val_rows=vv, score=predecir_bag(bag, vv, cols), cols=cols)


M = {}
M["lgb"] = correr_modelo(FEATS, etiqueta="completo")
M["lgb_sin_texto_poolB0"] = correr_modelo(FEATS_SIN_TXT, filas_ok="pool_b0", etiqueta="sin texto, pool B0")
M["lgb_sin_texto_poolcompleto"] = correr_modelo(FEATS_SIN_TXT, etiqueta="sin texto, pool completo")

# GLM + blend (peso elegido en el inner con modelos entrenados SOLO en fit)
lr_fit = entrenar_lr(fit, FEATS)
b_fit = entrenar_bag(fit, FEATS, M["lgb"]["params"], M["lgb"]["n_iter"], seeds=[0])
inn = inner.copy()
zl = predecir_bag(b_fit, inn, FEATS)
zg = zdentro(score_lr(lr_fit, inn, FEATS), inn["qid"])
inn_pos = con_pos(inn)


def mrr_cond(d, col):
    r = top_k(d.assign(_q=d.qid), col)
    g = d[d.es_ganador == 1][["id_lic", "parent_uei"]].rename(columns={"id_lic": "id"})
    return por_lic(r, g).mrr.mean()


mejor_w, mejor = 1.0, -1
for w in np.linspace(0, 1, 11):
    inn_pos = inn_pos.assign(s=(w * zl + (1 - w) * zg).reindex(inn_pos.index))
    s = mrr_cond(inn_pos, "s")
    if s > mejor:
        mejor, mejor_w = s, w
print(f"peso LGBM en el blend = {mejor_w:.1f} (MRR inner {mejor:.4f})")

lr_full = entrenar_lr(trn, FEATS)
M["glm"] = dict(val_rows=val, score=score_lr(lr_full, val, FEATS))
M["blend"] = dict(val_rows=val, score=mejor_w * M["lgb"]["score"] + (1 - mejor_w) * zdentro(M["glm"]["score"], val["qid"]))


#%% 7. Evaluación en val vs B0 oficial ------------------------------------------
filas, scs = [], {}
for nom, m in M.items():
    vv = m["val_rows"].assign(s=m["score"])
    for v in DS:
        d = vv[vv.vertical == v]
        r = top_k(d, "s")
        r.to_csv(OUT / f"ranking_{nom}_{v}.csv", index=False)
        s = por_lic(r, gan_val[v])
        scs[(nom, v)] = s
        b = sc_b0[v].reindex(s.index)
        inc = s.index.isin(sc_b0[v].index[sc_b0[v].best <= 10])
        dm, lo, hi = ic(s.mrr, b.mrr)
        filas.append(dict(vertical=v, modelo=nom, MRR=s.mrr.mean(), dMRR_vs_B0=dm, IC_lo=lo, IC_hi=hi,
                          Hit1=s.h1.mean(), Hit5=s.h5.mean(),
                          MRR_Sinc=s.mrr[inc].mean(), MRR_Snew=s.mrr[~inc].mean()))
for v in DS:
    s = sc_b0[v]
    inc = s.best <= 10
    filas.insert(0, dict(vertical=v, modelo="B0", MRR=s.mrr.mean(), dMRR_vs_B0=0.0, Hit1=s.h1.mean(),
                         Hit5=s.h5.mean(), MRR_Sinc=s.mrr[inc].mean(), MRR_Snew=s.mrr[~inc].mean()))
res = pd.DataFrame(filas)
res.to_csv(OUT / "resumen_validacion_mejora.csv", index=False)
print(res.round(4).to_string(index=False))

# Aporte del texto (pareado, con IC): modelo completo vs sin texto con pool B0 puro (la ablación justa)
for v in DS:
    a, b = scs[("lgb", v)].mrr, scs[("lgb_sin_texto_poolB0", v)].mrr.reindex(scs[("lgb", v)].index)
    dm, lo, hi = ic(a, b)
    print(f"[{v}] aporte del texto sobre incumbencia pura: dMRR = {dm:+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]")
for v in DS:
    print(f"uv run evaluar.py {OUT}/ranking_blend_{v}.csv --datos data/{v} --respuestas {VAL[v][0]} --b0 {VAL[v][1]}")


#%% 8. Interpretabilidad ----------------------------------------------------------
def familia(c):
    if c in TEXTO:
        return "texto"
    if c.startswith("emp_"):
        return "empresa"
    if c in ("n_cands", "vertical_cod", "tiene_hist"):
        return "contexto"
    return "incumbencia"


b = M["lgb"]["bag"][0]
Xs = val[FEATS].astype(np.float32)
Xs = Xs.sample(min(50_000, len(Xs)), random_state=0)
contrib = b.predict(Xs, pred_contrib=True)[:, :-1]
imp = pd.DataFrame({"feature": FEATS, "shap_abs_medio": np.abs(contrib).mean(0),
                    "ganancia": b.feature_importance("gain")})
imp["familia"] = imp.feature.map(familia)
imp["shap_pct"] = 100 * imp.shap_abs_medio / imp.shap_abs_medio.sum()
imp = imp.sort_values("shap_abs_medio", ascending=False).reset_index(drop=True)
imp.to_csv(OUT / "importancias.csv", index=False)
print(imp.head(20).round(4).to_string(index=False))
print(imp.groupby("familia").shap_pct.sum().round(1).sort_values(ascending=False))


#%% 9. Modelos finales con TODO (train+val+gap) para la entrega ------------------
final = entrenar_bag(ds, FEATS, M["lgb"]["params"], round(M["lgb"]["n_iter"] * 1.25))
for k, bb in enumerate(final):
    bb.save_model(str(OUT / f"lgb_final_mejorado_s{k}.txt"))
print("Guardados", len(final), "modelos finales en", OUT)
# Para licitaciones nuevas: generar el ds con V.extraer(...) del script anterior, pasarlo por
# preparar(ds_nuevo, v) y puntuar con predecir_bag(final, d, FEATS).
