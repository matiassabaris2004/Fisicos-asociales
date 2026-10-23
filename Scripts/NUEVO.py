# ==============================================================================
# Y-HAT FINAL  |  B0 por niveles + texto (MiniLM + TF-IDF) + LambdaRank + GLM + blend
# Spyder: correr por celdas (#%%) o el archivo entero, desde la raíz del repo.
# Si cambiás features, borrá la carpeta cache/ (o subí VERSION).
# ==============================================================================
#%% 0. Config
import os, sys, pathlib, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")
try:
    RAIZ = pathlib.Path(__file__).resolve().parent
except NameError:
    RAIZ = pathlib.Path.cwd()
# RAIZ = pathlib.Path(r"C:\ruta\al\repo")      # <- descomentar si hace falta
os.chdir(RAIZ); sys.path.insert(0, str(RAIZ))
import b0

VERTICALES = ["j"]            # arrancá con una; luego ["rd", "d", "j", "r", "it"]
VERSION = "v3"
CORTE_VAL = pd.Timestamp("2024-01-01")   # val = último semestre (como sugiere el README)
CORTE_ES = pd.Timestamp("2023-07-01")    # holdout interno para early stopping y peso del blend
K_B0, K_SIM = 150, 150        # pool = unión top-K por B0 y top-K por similitud
TOP_SALIDA = 100
MODELO_TXT = "sentence-transformers/all-MiniLM-L6-v2"
ABLACION = True               # entrena también sin features de texto: cuánto aporta el texto
CHEQUEAR_B0 = True
DATA = RAIZ / "data"
CACHE = RAIZ / "cache"; CACHE.mkdir(exist_ok=True)
NIV = ["P_psc_off", "P_psc", "P_naics", "P_global", "P_psc_sa"]   # orden de b0.NIVELES
PESO = np.array(b0.PESOS)[:, None]
META = ["id_lic", "parent_uei", "t", "semestre", "es_ganador"]
TEXTO = ["sim_max", "sim_mean", "sim_top3", "sim_off_max", "sim_psc_max", "simrec_max",
         "tfidf_max", "tfidf_top3", "tfidf_mean", "sim_top3_rank", "sim_top3_gap"]
AGG = ["sim_max", "sim_mean", "sim_top3", "sim_off_max", "sim_psc_max", "simrec_max",
       "tfidf_max", "tfidf_top3", "tfidf_mean", "n_prev", "dias_ult",
       "n_psc", "n_off", "n_naics", "n_psc_off"]

transformer = SentenceTransformer(MODELO_TXT)
transformer.max_seq_length = 128


#%% 1. Textos (en inglés, porque MiniLM es inglés) y datos de una vertical
def _norm(s):
    return s.astype(str).str.upper().str.strip()


def maestras():
    psc = pd.read_parquet(DATA / "psc.parquet")
    naics = pd.read_parquet(DATA / "naics.parquet")
    off = pd.read_parquet(DATA / "office.parquet")
    if "version" in naics:
        naics = naics.sort_values("version")
    naics = naics.drop_duplicates("codigo", keep="last")
    off["d"] = (off.oficina.fillna("") + " " + off.subagencia.fillna("") + " "
                + off.departamento.fillna("")).str.strip()
    return (dict(zip(_norm(psc.codigo), psc.descripcion.fillna(""))),
            dict(zip(naics.codigo.astype(str), naics.descripcion.fillna(""))),
            dict(zip(_norm(off.codigo), off.d)))


PSC_M, NAI_M, OFF_M = maestras()


def enriquecer(lic, his):
    lic, his = lic.copy(), his.copy()
    for d in (lic, his):
        d["_p"] = _norm(d.psc).map(PSC_M).fillna("")
        d["_n"] = d.naics.astype(str).map(NAI_M).fillna("")
        d["_o"] = _norm(d.office).map(OFF_M).fillna("")
    lic["texto_completo"] = (lic.title.fillna("").astype(str) + ". "
                             + lic.description.fillna("").astype(str) + ". Category: " + lic._p
                             + ". Industry: " + lic._n + ". Office: " + lic._o).str.strip()
    col = "descripcion" if "descripcion" in his else "description"
    his["texto_completo"] = (his[col].fillna("").astype(str) + ". Category: " + his._p
                             + ". Industry: " + his._n + ". Office: " + his._o).str.strip()
    return lic.drop(columns=["_p", "_n", "_o"]), his.drop(columns=["_p", "_n", "_o"])


class Vertical:
    def __init__(self, v):
        self.v = v
        d = DATA / v
        lic = pd.read_parquet(d / "licitaciones.parquet")
        his = pd.read_parquet(d / "historial.parquet")
        self.emp = pd.read_parquet(d / "empresas.parquet")
        self.gan = pd.read_parquet(d / "ganadores.parquet")
        lic, his = enriquecer(lic, his)
        lic["t"] = pd.to_datetime(lic["t"])
        self.lic = lic.sort_values(["t", "id"]).reset_index(drop=True)
        self.prior = b0.B0(his)                 # prior.h: historial ordenado por fecha
        self.H = self.prior.h                   # embeddings y tfidf se alinean a ESTE orden
        ruta = CACHE / f"emb_{v}_{VERSION}.npy"
        if ruta.exists() and len(np.load(ruta, mmap_mode="r")) == len(self.H):
            self.emb_h = np.load(ruta)
        else:
            self.emb_h = transformer.encode(self.H.texto_completo.tolist(), normalize_embeddings=True,
                                            batch_size=128, show_progress_bar=True).astype(np.float32)
            np.save(ruta, self.emb_h)
        # TF-IDF ajustado SOLO sobre historial (no mira las licitaciones)
        self.tfidf = TfidfVectorizer(max_features=30000, stop_words="english", ngram_range=(1, 2),
                                     sublinear_tf=True, min_df=2)
        self.tf_h = self.tfidf.fit_transform(self.H.texto_completo).tocsr()
        self.Hfi, self.Hts = self.H.fi.values, self.H.ts.values
        self.Hpsc, self.Hoff, self.Hnai = self.H.psc.values, self.H.office.values, self.H.naics.values

    # -- B0 por niveles (mismas cuentas que B0._p, sin ponderar)
    def niveles(self, lic):
        P = self.prior
        ts = np.datetime64(pd.Timestamp(lic.t), "D").astype(np.int64)
        ctx = {"psc": str(lic.psc).upper(), "office": str(lic.office).upper(),
               "naics": str(lic.naics), "sa": b0.set_aside(lic.set_aside)}
        out = np.zeros((len(b0.NIVELES), len(P.empresas)))
        for j, (cols, tabla) in enumerate(zip(b0.NIVELES, P.tablas)):
            e = tabla.get(tuple(ctx[c] for c in cols))
            if e is None:
                continue
            t_arr, fi, w = e
            n = np.searchsorted(t_arr, ts, side="left")          # estricto: sólo antes de t
            if n:
                v = np.bincount(fi[:n], weights=w[:n], minlength=len(P.empresas))
                out[j] = v / v.sum()
        return out

    def extraer(self, df_l, gan=None):
        P = self.prior
        emb_l = transformer.encode(df_l.texto_completo.tolist(), normalize_embeddings=True,
                                   batch_size=128, show_progress_bar=True).astype(np.float32)
        tf_l = self.tfidf.transform(df_l.texto_completo).tocsr()
        ganadores = set(zip(gan.id, gan.parent_uei)) if gan is not None else set()
        n_emp = len(P.empresas)
        piezas = []
        for i, lic in enumerate(df_l.itertuples(index=False)):
            ts_q = np.datetime64(pd.Timestamp(lic.t), "D").astype(np.int64)
            cands = b0.candidatas(self.emp, lic.t)                # sólo las del semestre
            if len(cands) == 0:
                continue
            codes = np.array([P.idx.get(u, -1) for u in cands])
            ok = codes >= 0
            niv = self.niveles(lic)
            Pm = np.zeros((len(NIV), len(cands)))
            Pm[:, ok] = niv[:, codes[ok]]
            p_b0 = (PESO * Pm).sum(0)
            if CHEQUEAR_B0 and i == 0:
                assert np.allclose(P.p_b0(lic, self.emp).values, p_b0), "p_b0 propio != b0.py"

            n = np.searchsorted(self.Hts, ts_q, side="left")      # historial < t (prefijo)
            is_c = np.zeros(n_emp, bool); is_c[codes[ok]] = True
            sel = np.flatnonzero(is_c[self.Hfi[:n]]) if n else np.array([], dtype=int)
            if len(sel):
                d = pd.DataFrame({
                    "fi": self.Hfi[sel],
                    "sim": self.emb_h[sel] @ emb_l[i],
                    "tf": (self.tf_h[sel] @ tf_l[i].T).toarray().ravel(),
                    "dt": ts_q - self.Hts[sel],
                    "mp": self.Hpsc[sel] == str(lic.psc).upper(),
                    "mo": self.Hoff[sel] == str(lic.office).upper(),
                    "mn": self.Hnai[sel] == str(lic.naics)})
                d["mpo"] = d.mp & d.mo
                d["simrec"] = d.sim * np.exp(-d.dt / (b0.TAU_ANIOS * 365.25))
                g = d.groupby("fi")
                a = pd.DataFrame({"sim_max": g.sim.max(), "sim_mean": g.sim.mean(),
                                  "tfidf_max": g.tf.max(), "tfidf_mean": g.tf.mean(),
                                  "simrec_max": g.simrec.max(), "n_prev": g.size(),
                                  "dias_ult": g.dt.min(), "n_psc": g.mp.sum(), "n_off": g.mo.sum(),
                                  "n_naics": g.mn.sum(), "n_psc_off": g.mpo.sum()})
                a["sim_top3"] = (d.sort_values(["fi", "sim"], ascending=[True, False])
                                 .groupby("fi").head(3).groupby("fi").sim.mean())
                a["tfidf_top3"] = (d.sort_values(["fi", "tf"], ascending=[True, False])
                                   .groupby("fi").head(3).groupby("fi").tf.mean())
                a["sim_off_max"] = d[d.mo].groupby("fi").sim.max()    # parecido en la MISMA oficina
                a["sim_psc_max"] = d[d.mp].groupby("fi").sim.max()    # parecido en el MISMO PSC
                a = a.reindex(codes).reset_index(drop=True)
            else:
                a = pd.DataFrame(np.nan, index=range(len(cands)), columns=AGG)
            a = a[AGG].fillna({c: 0.0 for c in AGG} | {"dias_ult": 3650.0})

            o = a.copy()
            o.insert(0, "parent_uei", cands); o.insert(0, "id_lic", lic.id)
            o["t"] = pd.Timestamp(lic.t); o["semestre"] = b0.semestre(lic.t)
            for j, nm in enumerate(NIV):
                o[nm] = Pm[j]
            o["p_b0"] = p_b0
            # features relativas DENTRO de la licitación (sobre todas las candidatas)
            o["p_b0_rank"] = o.p_b0.rank(ascending=False, method="min")
            o["p_b0_rel"] = o.p_b0 / (o.p_b0.max() + 1e-12)
            o["sim_top3_rank"] = o.sim_top3.rank(ascending=False, method="min")
            o["sim_top3_gap"] = o.sim_top3 - o.sim_top3.max()
            o["n_cands"] = len(cands)
            if len(o) > K_B0 + K_SIM:                              # pool de reranking
                x = np.lexsort((-o.sim_top3.values, -o.p_b0.values))[:K_B0]
                y = np.argsort(-o.sim_top3.values, kind="stable")[:K_SIM]
                o = o.iloc[np.unique(np.concatenate([x, y]))]
            o["es_ganador"] = [int((lic.id, u) in ganadores) for u in o.parent_uei]
            piezas.append(o)
        ds = pd.concat(piezas, ignore_index=True)
        num = list(self.emp.select_dtypes("number").columns)       # lo que se sabía al inicio del semestre
        if num:
            ef = (self.emp[["parent_uei", "semestre"] + num].drop_duplicates(["parent_uei", "semestre"])
                  .rename(columns={c: "emp_" + c for c in num}))
            ds = ds.merge(ef, on=["parent_uei", "semestre"], how="left")
        return ds


#%% 2. GLM (logit binomial) con features transformadas y estandarizadas
def glm_X(d):
    eps = 1e-6
    p = np.clip(d.p_b0.values, eps, 1 - eps)
    X = pd.DataFrame({
        "logit_p_b0": np.log(p / (1 - p)),
        "log_rank_b0": np.log1p(d.p_b0_rank),
        "p_psc_off": np.sqrt(d.P_psc_off), "p_psc": np.sqrt(d.P_psc),
        "sim_top3": d.sim_top3, "sim_top3_gap": d.sim_top3_gap,
        "sim_off_max": d.sim_off_max, "sim_psc_max": d.sim_psc_max,
        "simrec_max": d.simrec_max, "tfidf_top3": d.tfidf_top3, "tfidf_max": d.tfidf_max,
        "log_n_prev": np.log1p(d.n_prev), "log_n_psc": np.log1p(d.n_psc),
        "log_n_off": np.log1p(d.n_off), "log_n_psc_off": np.log1p(d.n_psc_off),
        "log_dias_ult": np.log1p(d.dias_ult), "tiene_hist": (d.n_prev > 0).astype(float)},
        index=d.index)
    return X


class GLMRank:
    def fit(self, d):
        X = glm_X(d)
        self.mu, self.sd = X.mean(), X.std().replace(0, 1)
        Z = sm.add_constant((X - self.mu) / self.sd, has_constant="add")
        fam = sm.families.Binomial()
        try:
            self.res = sm.GLM(d.es_ganador.values, Z, family=fam).fit()
        except Exception as e:                                   # colinealidad -> ridge
            print("GLM plano falló, uso ridge:", e)
            self.res = sm.GLM(d.es_ganador.values, Z, family=fam).fit_regularized(
                alpha=1e-3, L1_wt=0.0)
        return self

    def score(self, d):
        Z = sm.add_constant((glm_X(d) - self.mu) / self.sd, has_constant="add")
        return np.asarray(Z) @ np.asarray(self.res.params)       # predictor lineal (monótono con P)


#%% 3. Métricas (réplica de evaluar.py: ganadores múltiples sin contarse entre sí)
def top_k(d, col):
    d = d.sort_values(["id_lic", col], ascending=[True, False], kind="stable").copy()
    d["puesto"] = d.groupby("id_lic").cumcount() + 1
    return d[d.puesto <= TOP_SALIDA][["id_lic", "parent_uei", "puesto"]].rename(columns={"id_lic": "id"})


def por_lic(rank, gan):
    pos = dict(zip(zip(rank.id, rank.parent_uei), rank.puesto))
    filas = {}
    for lid, gs in gan.groupby("id"):
        r = sorted(pos.get((lid, u), np.inf) for u in gs.parent_uei)
        adj = [x - k for k, x in enumerate(r)]
        filas[lid] = (np.mean([1 / a if np.isfinite(a) else 0 for a in adj]), r[0] <= 1, r[0] <= 5, r[0])
    return pd.DataFrame(filas, index=["mrr", "h1", "h5", "best"]).T.astype(float)


def zdentro(s, ids):
    g = s.groupby(ids)
    return (s - g.transform("mean")) / (g.transform("std").fillna(1) + 1e-9)


def ic_delta(a, b, n=2000, seed=0):
    """IC95% bootstrap pareado de mean(a - b) sobre licitaciones."""
    dif = (a - b).values; rng = np.random.default_rng(seed)
    m = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(n)]
    return dif.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


#%% 4. Pipeline por vertical
def correr(v):
    V = Vertical(v)
    ruta = CACHE / f"ds_{v}_{VERSION}_{K_B0}_{K_SIM}.parquet"
    ds = pd.read_parquet(ruta) if ruta.exists() else V.extraer(V.lic, V.gan)
    if not ruta.exists():
        ds.to_parquet(ruta)
    FEATS = [c for c in ds.columns if c not in META]
    FEATS_SIN_TXT = [c for c in FEATS if c not in TEXTO]
    train, val = ds[ds.t < CORTE_VAL].reset_index(drop=True), ds[ds.t >= CORTE_VAL].reset_index(drop=True)
    fit, hold = train[train.t < CORTE_ES], train[train.t >= CORTE_ES]
    grp = lambda d: d.groupby("id_lic", sort=False).size().values
    cob = lambda d: d[d.es_ganador == 1].id_lic.nunique() / max(V.gan[V.gan.id.isin(d.id_lic.unique())].id.nunique(), 1)
    print(f"[{v}] filas train {len(train):,} val {len(val):,} | cobertura del pool (techo Hit@100): "
          f"train {cob(train):.3f} val {cob(val):.3f}")

    P = dict(objective="lambdarank", learning_rate=0.03, num_leaves=31, min_child_samples=50,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             lambdarank_truncation_level=30, random_state=42, n_jobs=-1, verbose=-1)

    def lgb_fit(d, cols, n, es=None):
        m = lgb.LGBMRanker(n_estimators=n, **P)
        kw = {}
        if es is not None:
            kw = dict(eval_set=[(es[cols], es.es_ganador)], eval_group=[grp(es)], eval_at=[10],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return m.fit(d[cols], d.es_ganador, group=grp(d), **kw)

    # (a) elegir n_estimators y peso del blend SIN tocar val
    m_a = lgb_fit(fit, FEATS, 1500, es=hold)
    best = int(m_a.best_iteration_ or 300)
    g_a = GLMRank().fit(fit)
    gh = V.gan[V.gan.id.isin(hold.id_lic.unique())]
    hold = hold.copy()
    zl, zg = zdentro(pd.Series(m_a.predict(hold[FEATS]), index=hold.index), hold.id_lic), \
        zdentro(pd.Series(g_a.score(hold), index=hold.index), hold.id_lic)
    mejor_w, mejor = 1.0, -1
    for w in np.linspace(0, 1, 11):
        hold["s"] = w * zl + (1 - w) * zg
        s = por_lic(top_k(hold, "s"), gh).mrr.mean()
        if s > mejor:
            mejor, mejor_w = s, w
    print(f"[{v}] best_iter={best} | peso LGBM en blend={mejor_w:.1f} (MRR holdout {mejor:.4f})")

    # (b) reentrenar con todo train y evaluar en val
    m_b = lgb_fit(train, FEATS, max(int(best * 1.15), 50))
    g_b = GLMRank().fit(train)
    try:
        print(g_b.res.summary().tables[1])
    except Exception:
        pass
    val["lgb"] = m_b.predict(val[FEATS]); val["glm"] = g_b.score(val)
    val["blend"] = mejor_w * zdentro(val.lgb, val.id_lic) + (1 - mejor_w) * zdentro(val.glm, val.id_lic)
    if ABLACION:
        m_nt = lgb_fit(train, FEATS_SIN_TXT, max(int(best * 1.15), 50))
        val["lgb_sin_texto"] = m_nt.predict(val[FEATS_SIN_TXT])

    vl = V.lic[V.lic.t >= CORTE_VAL]
    gv = V.gan[V.gan.id.isin(vl.id)]
    gv.to_parquet(f"mi_val_{v}.parquet")
    rb0 = pd.concat([V.prior.ranking(f, V.emp, TOP_SALIDA) for f in vl.itertuples(index=False)],
                    ignore_index=True)[["id", "parent_uei", "puesto"]]
    rb0.to_csv(f"b0_val_{v}.csv", index=False)
    s_b0 = por_lic(rb0, gv)
    inc = s_b0.index[s_b0.best <= 10]                       # S_inc: B0 ya pone un ganador en su top 10
    filas = []
    for nom in ["lgb", "glm", "blend"] + (["lgb_sin_texto"] if ABLACION else []):
        r = top_k(val, nom)
        s = por_lic(r, gv)
        dm, lo, hi = ic_delta(s.mrr, s_b0.mrr.reindex(s.index))
        filas.append(dict(vertical=v, modelo=nom, MRR=s.mrr.mean(), dMRR_vs_B0=dm, IC_lo=lo, IC_hi=hi,
                          Hit1=s.h1.mean(), Hit5=s.h5.mean(),
                          MRR_Sinc=s.mrr[s.index.isin(inc)].mean(), MRR_Snew=s.mrr[~s.index.isin(inc)].mean()))
        r.to_csv(f"ranking_{nom}_{v}.csv", index=False)
    filas.insert(0, dict(vertical=v, modelo="B0", MRR=s_b0.mrr.mean(), dMRR_vs_B0=0, Hit1=s_b0.h1.mean(),
                         Hit5=s_b0.h5.mean(), MRR_Sinc=s_b0.mrr[s_b0.index.isin(inc)].mean(),
                         MRR_Snew=s_b0.mrr[~s_b0.index.isin(inc)].mean()))
    res = pd.DataFrame(filas)
    print(res.round(4).to_string(index=False))
    imp = pd.Series(m_b.booster_.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
    print(imp.head(12).round(0))
    print(f"uv run evaluar.py ranking_blend_{v}.csv --datos data/{v} --respuestas mi_val_{v}.parquet --b0 b0_{v}.csv")

    # (c) modelos finales con TODO (train+val) para la entrega
    final_l = lgb_fit(ds.reset_index(drop=True), FEATS, max(int(best * 1.25), 50))
    final_g = GLMRank().fit(ds)
    final_l.booster_.save_model(str(CACHE / f"lgb_final_{v}.txt"))
    return dict(V=V, ds=ds, res=res, lgb=final_l, glm=final_g, w=mejor_w, feats=FEATS)


def rankear_nuevas(R, df_lic_nueva, salida):
    """Para licitaciones nuevas: usa los modelos finales. `df_lic_nueva` crudo (con columnas de
    licitaciones.parquet); el historial del B0 debe incluir todo lo anterior a t."""
    V = R["V"]
    l, _ = enriquecer(df_lic_nueva, V.H.drop(columns=["texto_completo"]).assign(texto_completo=V.H.texto_completo))
    l["t"] = pd.to_datetime(l["t"])
    d = V.extraer(l.sort_values(["t", "id"]).reset_index(drop=True))
    d["s"] = R["w"] * zdentro(pd.Series(R["lgb"].predict(d[R["feats"]]), index=d.index), d.id_lic) \
        + (1 - R["w"]) * zdentro(pd.Series(R["glm"].score(d), index=d.index), d.id_lic)
    top_k(d, "s").to_csv(salida, index=False)


#%% 5. Ejecutar
RESULTADOS = {}
for v in VERTICALES:
    print(f"\n================ VERTICAL {v.upper()} ================")
    RESULTADOS[v] = correr(v)
resumen = pd.concat([r["res"] for r in RESULTADOS.values()], ignore_index=True)
resumen.to_csv("resumen_validacion.csv", index=False)
print(resumen.round(4).to_string(index=False))
