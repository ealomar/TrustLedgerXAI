"""
TrustLedger-XAI - Stage 1 (ranking-oriented detector), pre-declared protocol.

PRE-DECLARED BEFORE ANY RUN (do not edit after results exist):
  Primary metric    mean per-year NDCG@1% over test years 2003-2008 (Bao et al.'s
                    primary period and preferred metric), under the corrected
                    serial-fraud recoding.
  Secondary         mean per-year AUC 2003-2008; pooled AUC and within-year counts
                    2003-2014, firm-clustered bootstrap.
  Selection         each configuration in GRID is run on development years 1998-2002
                    only; the one with the highest mean per-year NDCG@1% of the fused
                    model is selected; configurations within 0.005 of the best are
                    tied, and the tie goes to the higher development mean AUC.
  Test              the selected configuration is run ONCE on 2003-2014 and compared
                    with the v10 model and the RUSBoost and logit reproductions
                    (scores reused from checkpoints_v3). All development results
                    are reported.
  Inference         firm-clustered bootstrap (1,000 replicates) of the difference
                    in mean per-year NDCG@1% (2003-2008) between the selected model
                    and each comparator; two-sided; Holm across comparators.

GRID (12 configurations) = features x learner x weighting
  features   raw+scaled | raw+scaled+within-year percentiles
  learner    hgb-reg (regularised gradient boosting, classification)
             lgb-lambdarank | lgb-xendcg (LightGBM ranking, one query per fiscal year)
  weighting  none | recency+censor (half-life 5 years; negatives in the last two
             training years weighted 0.5, since their misstatements may not yet be
             enforced)

Resumable: every (configuration, year) result is checkpointed.
    python stage1.py dev
    python stage1.py test
"""

import json
import os
import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from aaer_pipeline import RAW28, TEMPORAL, holm, load, ndcg_at_k, recode_serial_fraud, temporal_matrix, topk_stats
from branches import _hgb
from pipeline_v3 import REG, SCALED, add_scaled

warnings.filterwarnings("ignore")
RNG = 20260913
OUT, CK = "results_s1", "checkpoints_s1"
os.makedirs(OUT, exist_ok=True); os.makedirs(CK, exist_ok=True)
DEV, TEST = list(range(1998, 2003)), list(range(2003, 2015))
PRIMARY = list(range(2003, 2009))
PCT = [f"p_{c}" for c in RAW28 + SCALED]
GRID = {f"{f}|{l}|{w}": dict(features=f, learner=l, weighting=w)
        for f in ["raw+scaled", "raw+scaled+pct"]
        for l in ["hgb-reg", "lgb-lambdarank", "lgb-xendcg"]
        for w in ["none", "recency+censor"]}
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
           random_state=RNG, verbose=-1, deterministic=True, force_row_wise=True, n_jobs=1)


def add_pct(d):
    for c in RAW28 + SCALED:
        d[f"p_{c}"] = d.groupby("fyear")[c].rank(pct=True).fillna(0.5)
    return d


def cs_cols(f):
    return RAW28 + SCALED + (PCT if f.endswith("pct") else [])


def mat(frame, cols):
    return np.nan_to_num(frame[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)


def weights(train, ty, mode):
    if mode == "none":
        return np.ones(len(train))
    age = (ty - 2) - train.fyear.to_numpy()
    w = 0.5 ** (age / 5.0)
    w[(age <= 1) & (train.misstate.to_numpy() == 0)] *= 0.5
    return w


class Branch:
    def __init__(self, learner):
        self.learner, self.m = learner, None

    def fit(self, X, y, w, year):
        if self.learner == "hgb-reg":
            self.m = _hgb(REG, RNG).fit(X, y, sample_weight=w)
            return self
        order = np.argsort(year, kind="stable")
        _, counts = np.unique(year[order], return_counts=True)
        obj = "lambdarank" if self.learner == "lgb-lambdarank" else "rank_xendcg"
        extra = {"lambdarank_truncation_level": 100} if obj == "lambdarank" else {}
        self.m = lgb.LGBMRanker(objective=obj, **LGB, **extra).fit(
            X[order], y[order], sample_weight=w[order], group=counts)
        return self

    def decision_function(self, X):
        return self.m.decision_function(X) if self.learner == "hgb-reg" else self.m.predict(X, raw_score=True)


class Fused:
    def __init__(self, cs, tp, iso, scaler, lr, cfg):
        self.m_cs, self.m_tp, self.iso, self.scaler, self.lr, self.cfg = cs, tp, iso, scaler, lr, cfg

    def logit(self, Xc, Xt):
        M = np.column_stack([self.m_cs.decision_function(Xc), self.m_tp.decision_function(Xt),
                             -self.iso.score_samples(Xc[:, :len(RAW28)])])
        z = (M - self.scaler.mean_) / self.scaler.scale_
        w = self.lr.coef_[0]
        return ((self.lr.intercept_[0] + z[:, 0] * w[0]) + z[:, 1] * w[1]) + z[:, 2] * w[2]


def split(d, ty):
    train = d[(d.fyear >= 1991) & (d.fyear <= ty - 2)]
    train, _ = recode_serial_fraud(train, ty, d)
    return train, d[d.fyear == ty]


def fit(train, ty, cfg):
    Xc, Xt = mat(train, cs_cols(cfg["features"])), temporal_matrix(train)
    y, g, yr = train.misstate.to_numpy(int), train.gvkey.to_numpy(), train.fyear.to_numpy()
    w = weights(train, ty, cfg["weighting"])
    cs = Branch(cfg["learner"]).fit(Xc, y, w, yr)
    tp = Branch(cfg["learner"]).fit(Xt, y, w, yr)
    iso = IsolationForest(n_estimators=300, random_state=RNG).fit(Xc[:, :len(RAW28)])
    oof = np.zeros((len(y), 2))
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=RNG).split(Xc, y, g):
        oof[va, 0] = Branch(cfg["learner"]).fit(Xc[tr], y[tr], w[tr], yr[tr]).decision_function(Xc[va])
        oof[va, 1] = Branch(cfg["learner"]).fit(Xt[tr], y[tr], w[tr], yr[tr]).decision_function(Xt[va])
    M = np.column_stack([oof, -iso.score_samples(Xc[:, :len(RAW28)])])
    sc = StandardScaler().fit(M)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG).fit(
        sc.transform(M), y, sample_weight=w)
    return Fused(cs, tp, iso, sc, lr, cfg)


def score(f, frame):
    return f.logit(mat(frame, cs_cols(f.cfg["features"])), temporal_matrix(frame))


def metrics(stage, name, ty, y, s):
    return {"stage": stage, "config": name, "test_year": ty, "auc": roc_auc_score(y, s),
            "pr_auc": average_precision_score(y, s), **topk_stats(y, s)}


def csv_done(p):
    return set() if not os.path.exists(p) else {(r.config, r.test_year) for r in pd.read_csv(p).itertuples()}


def append(p, rows):
    pd.DataFrame(rows).to_csv(p, mode="a", header=not os.path.exists(p), index=False)


def dev(d):
    p = f"{OUT}/dev_per_year.csv"; have = csv_done(p)
    for ty in DEV:
        train, te = split(d, ty); y = te.misstate.to_numpy(int)
        for name, cfg in GRID.items():
            if (name, ty) in have:
                continue
            r = metrics("dev", name, ty, y, score(fit(train, ty, cfg), te))
            append(p, [r]); print(f"dev {ty} {name}: ndcg {r['ndcg_at_k']:.3f} auc {r['auc']:.3f}", flush=True)
    D = pd.read_csv(p)
    S = D.groupby("config").agg(dev_ndcg=("ndcg_at_k", "mean"), dev_auc=("auc", "mean"),
                                dev_pr_auc=("pr_auc", "mean"), dev_found=("frauds_in_topk", "sum"),
                                years=("test_year", "nunique")).reset_index()
    S.to_csv(f"{OUT}/dev_summary.csv", index=False)
    print(S.sort_values("dev_ndcg", ascending=False).round(4).to_string(index=False))


def select():
    S = pd.read_csv(f"{OUT}/dev_summary.csv")
    assert (S.years == len(DEV)).all() and len(S) == len(GRID), "development grid incomplete"
    tied = S[S.dev_ndcg >= S.dev_ndcg.max() - 0.005]
    return tied.sort_values("dev_auc", ascending=False).config.iloc[0]


def boot_ndcg(ys, ss, gs, ref, n=1000):
    """Firm-clustered bootstrap of mean per-year NDCG@1% over the primary years."""
    rng = np.random.default_rng(RNG)
    firms = np.unique(np.concatenate(gs))
    rows = {f: [] for f in firms}
    for yi, g in enumerate(gs):
        for j, f in enumerate(g):
            rows[f].append((yi, j))
    reps = {m: [] for m in ss}
    for _ in range(n):
        pick = rng.choice(firms, len(firms), replace=True)
        idx = [[] for _ in ys]
        for f in pick:
            for yi, j in rows[f]:
                idx[yi].append(j)
        for m in ss:
            vals = []
            for yi in range(len(ys)):
                ii = np.array(idx[yi]); y = ys[yi][ii]
                if y.sum() == 0:
                    continue
                s = ss[m][yi][ii]; k = max(1, int(round(0.01 * len(ii))))
                vals.append(ndcg_at_k(y, s, k))
            reps[m].append(np.mean(vals))
    out = {}
    for m in ss:
        diff = np.array(reps[ref]) - np.array(reps[m])
        out[m] = (np.percentile(reps[m], 2.5), np.percentile(reps[m], 97.5),
                  np.percentile(diff, 2.5), np.percentile(diff, 97.5),
                  1.0 if m == ref else min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
    return out


def test(d):
    best = select()
    json.dump({"selected": best, "rule": "max dev mean NDCG@1%; ties within 0.005 -> higher dev AUC"},
              open(f"{OUT}/selection.json", "w"), indent=1)
    print("selected:", best, flush=True)
    cfg = GRID[best]; label = "Stage 1 (selected)"
    old = pd.read_csv("results/aaer_per_year.csv")
    for ty in TEST:
        ck = f"{CK}/scores_{ty}.json"
        if os.path.exists(ck):
            continue
        train, te = split(d, ty)
        f = fit(train, ty, cfg)
        sc = {label: score(f, te).tolist()}
        v3 = json.load(open(f"checkpoints_v3/scores_{ty}.json"))
        sc["RUSBoost (reproduction)"] = v3["RUSBoost (tuned)"]
        sc["Logit on ratios"] = v3["Logit on ratios"]
        sc["TrustLedger-XAI v10"] = v3["TrustLedger-XAI (v10, untuned)"]
        json.dump(sc, open(ck, "w"))
        if ty == TEST[-1]:
            joblib.dump({"fused": f, "cfg": cfg, "year": ty}, f"{CK}/final_year_s1.joblib")
        print(f"test {ty} done", flush=True)
    rows, ys, ss, gs = [], [], {}, []
    for ty in TEST:
        te = d[d.fyear == ty]; y = te.misstate.to_numpy(int)
        sc = {k: np.array(v) for k, v in json.load(open(f"{CK}/scores_{ty}.json")).items()}
        rows += [metrics("test", m, ty, y, s) for m, s in sc.items()]
        if ty in PRIMARY:
            ys.append(y); gs.append(te.gvkey.to_numpy())
            for m, s in sc.items():
                ss.setdefault(m, []).append(s)
    P = pd.DataFrame(rows); P.to_csv(f"{OUT}/test_per_year.csv", index=False)
    B = boot_ndcg(ys, ss, gs, label)
    adj = holm({m: B[m][4] for m in B if m != label})
    out = []
    for m in ss:
        a, b = P[(P.config == m) & P.test_year.isin(PRIMARY)], P[P.config == m]
        out.append({"model": m, "ndcg_2003_2008": round(a.ndcg_at_k.mean(), 4),
                    "ndcg_ci95": f"[{B[m][0]:.4f}, {B[m][1]:.4f}]",
                    "diff_vs_stage1_ci95": None if m == label else f"[{B[m][2]:.4f}, {B[m][3]:.4f}]",
                    "p_holm": None if m == label else round(adj[m], 4),
                    "found_2003_2008": int(a.frauds_in_topk.sum()),
                    "auc_2003_2008": round(a.auc.mean(), 4), "pr_auc_2003_2008": round(a.pr_auc.mean(), 4),
                    "ndcg_2003_2014": round(b.ndcg_at_k.mean(), 4), "auc_2003_2014": round(b.auc.mean(), 4),
                    "found_2003_2014": int(b.frauds_in_topk.sum())})
    R = pd.DataFrame(out).sort_values("ndcg_2003_2008", ascending=False)
    R.to_csv(f"{OUT}/test_summary.csv", index=False)
    print(R.to_string(index=False))


if __name__ == "__main__":
    d = add_pct(add_scaled(load()))
    {"dev": dev, "test": test}[sys.argv[1]](d)
