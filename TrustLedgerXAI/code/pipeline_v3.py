"""
TrustLedger-XAI - revision 3: tuning on development years, then one frozen test run.

PROTOCOL (fixed before any 2003-2014 result was computed with these configurations)
  * Development years: 1998-2002, same protocol as the test years (expanding window
    from 1991, two-year gap, serial-fraud recoding).
  * Every configuration in CONFIGS and RUS_CONFIGS is evaluated on the development
    years only. The selection criterion is the mean per-year AUC of the fused model
    (for RUSBoost: of RUSBoost itself) over 1998-2002; ties within 0.002 are broken
    by the simpler configuration (lower position in the list).
  * The selected fused configuration and the selected RUSBoost configuration are
    then run ONCE on 2003-2014, together with the untuned v10 configuration for
    reference. Every development result is reported, not only the winners.

Both stages checkpoint after every (configuration, year), so an interrupted run
resumes where it stopped:

    python pipeline_v3.py dev        # development grid  -> results_v3/dev_*.csv
    python pipeline_v3.py test       # frozen test run   -> results_v3/test_*.csv
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from aaer_pipeline import (RATIOS14, RAW28, TEMPORAL, clustered_boot, holm, load,
                           logit_ratios, recode_serial_fraud, temporal_matrix, topk_stats)
from branches import GBDTBranch

warnings.filterwarnings("ignore")
RNG = 20260913
OUT = "results_v3"; CK = "checkpoints_v3"
os.makedirs(OUT, exist_ok=True); os.makedirs(CK, exist_ok=True)
DEV_YEARS = list(range(1998, 2003))
TEST_YEARS = list(range(2003, 2015))
GAP, TRAIN_START = 2, 1991
SCALED = [f"s_{c}" for c in RAW28 if c != "at"]

V10 = {"iters": 300, "lr": 0.05, "leaves": 31, "min_leaf": 40, "l2": 1.0}
REG = {"iters": 500, "lr": 0.03, "leaves": 8, "min_leaf": 100, "l2": 1.0, "max_features": 0.5}
SMALL = {"iters": 150, "lr": 0.05, "leaves": 8, "min_leaf": 20, "l2": 1.0}
LEARNERS = {
    "single-v10": {"kind": "single", "params": V10},
    "single-regularised": {"kind": "single", "params": REG},
    "bagged-r1": {"kind": "bagged", "bags": 25, "ratio": 1, "params": SMALL},
    "bagged-r5": {"kind": "bagged", "bags": 25, "ratio": 5, "params": SMALL},
}
CONFIGS = {f"{l}|{f}": {"learner": l, "features": f}
           for f in ["raw", "raw+scaled"] for l in LEARNERS}
RUS_CONFIGS = {
    "rus-1000-leaf5-lr0.1": dict(n=1000, leaf=5, lr=0.1),
    "rus-1000-leaf20-lr0.1": dict(n=1000, leaf=20, lr=0.1),
    "rus-500-leaf5-lr0.05": dict(n=500, leaf=5, lr=0.05),
}


def add_scaled(d):
    at = d["at"].replace(0, np.nan)
    for c in RAW28:
        if c != "at":
            d[f"s_{c}"] = d[c] / at
    return d


def cs_cols(features):
    return RAW28 + (SCALED if features == "raw+scaled" else [])


def cs_matrix(frame, features):
    return np.nan_to_num(frame[cs_cols(features)].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)


class FusedV3:
    def __init__(self, b_cs, b_tp, iso, scaler, lr, features):
        self.m_cs, self.m_tp, self.iso, self.scaler, self.lr = b_cs, b_tp, iso, scaler, lr
        self.features = features

    def margins(self, Xc, Xt):
        return np.column_stack([self.m_cs.decision_function(Xc), self.m_tp.decision_function(Xt),
                                -self.iso.score_samples(Xc[:, :len(RAW28)])])

    def logit(self, Xc, Xt):
        z = (self.margins(Xc, Xt) - self.scaler.mean_) / self.scaler.scale_
        w = self.lr.coef_[0]
        return ((self.lr.intercept_[0] + z[:, 0] * w[0]) + z[:, 1] * w[1]) + z[:, 2] * w[2]

    def proba(self, Xc, Xt):
        return 1 / (1 + np.exp(-self.logit(Xc, Xt)))


def split(d, ty):
    train = d[(d.fyear >= TRAIN_START) & (d.fyear <= ty - GAP)]
    test = d[d.fyear == ty]
    train, _ = recode_serial_fraud(train, ty, d)
    return train, test


def fit_fused(train, cfg):
    learner = LEARNERS[cfg["learner"]]
    Xc, Xt = cs_matrix(train, cfg["features"]), temporal_matrix(train)
    y, g = train.misstate.to_numpy(int), train.gvkey.to_numpy()
    b_cs = GBDTBranch(learner).fit(Xc, y)
    b_tp = GBDTBranch(learner).fit(Xt, y)
    iso = IsolationForest(n_estimators=300, random_state=RNG).fit(Xc[:, :len(RAW28)])
    oof = np.zeros((len(y), 2))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=RNG)
    for tr, va in cv.split(Xc, y, g):
        oof[va, 0] = GBDTBranch(learner).fit(Xc[tr], y[tr]).decision_function(Xc[va])
        oof[va, 1] = GBDTBranch(learner).fit(Xt[tr], y[tr]).decision_function(Xt[va])
    M = np.column_stack([oof, -iso.score_samples(Xc[:, :len(RAW28)])])
    scaler = StandardScaler().fit(M)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=RNG).fit(scaler.transform(M), y)
    return FusedV3(b_cs, b_tp, iso, scaler, lr, cfg["features"])


def rus(c):
    return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=c["leaf"]),
                              n_estimators=c["n"], learning_rate=c["lr"], random_state=RNG)


def row(stage, name, ty, y, s):
    return {"stage": stage, "config": name, "test_year": ty, "n_test": len(y), "n_fraud": int(y.sum()),
            "auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), **topk_stats(y, s)}


def done(path):
    return set() if not os.path.exists(path) else {
        (r.config, r.test_year) for r in pd.read_csv(path).itertuples()}


def append(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ------------------------------------------------------------------ development
def dev(d):
    path = f"{OUT}/dev_per_year.csv"
    have = done(path)
    for ty in DEV_YEARS:
        train, test = split(d, ty)
        y = test.misstate.to_numpy(int)
        for name, cfg in CONFIGS.items():
            if (name, ty) in have:
                continue
            f = fit_fused(train, cfg)
            Xc, Xt = cs_matrix(test, cfg["features"]), temporal_matrix(test)
            rows = [row("dev", name, ty, y, f.proba(Xc, Xt)),
                    row("dev", name + " [cs branch]", ty, y, f.m_cs.decision_function(Xc))]
            append(path, rows)
            print(f"dev {ty} {name}: fused AUC {rows[0]['auc']:.3f}", flush=True)
        for name, c in RUS_CONFIGS.items():
            if (name, ty) in have:
                continue
            m = rus(c).fit(train[RAW28].to_numpy(float), train.misstate.to_numpy(int))
            append(path, [row("dev", name, ty, y, m.predict_proba(test[RAW28].to_numpy(float))[:, 1])])
            print(f"dev {ty} {name}: done", flush=True)
    D = pd.read_csv(path)
    S = D.groupby("config").agg(mean_auc=("auc", "mean"), mean_pr_auc=("pr_auc", "mean"),
                                mean_ndcg=("ndcg_at_k", "mean"), found=("frauds_in_topk", "sum"),
                                years=("test_year", "nunique")).reset_index()
    S.to_csv(f"{OUT}/dev_summary.csv", index=False)
    print(S.sort_values("mean_auc", ascending=False).to_string(index=False))
    return S


def select(S, names):
    s = S[S.config.isin(names)].copy()
    s["order"] = s.config.map({n: i for i, n in enumerate(names)})
    best = s.mean_auc.max()
    return s[s.mean_auc >= best - 0.002].sort_values("order").config.iloc[0]


# ------------------------------------------------------------------ frozen test
def test(d):
    S = pd.read_csv(f"{OUT}/dev_summary.csv")
    assert (S.years == len(DEV_YEARS)).all(), "development grid incomplete"
    best = select(S, list(CONFIGS)); best_rus = select(S, list(RUS_CONFIGS))
    json.dump({"fused": best, "rusboost": best_rus, "criterion": "mean per-year AUC 1998-2002"},
              open(f"{OUT}/selection.json", "w"), indent=1)
    print("selected:", best, "|", best_rus, flush=True)
    runs = {"TrustLedger-XAI (tuned)": ("fused", best),
            "TrustLedger-XAI (v10, untuned)": ("fused", "single-v10|raw"),
            "RUSBoost (tuned)": ("rus", best_rus),
            "RUSBoost (v10)": ("rus", "rus-1000-leaf5-lr0.1"),
            "Logit on ratios": ("logit", None)}
    if best_rus == "rus-1000-leaf5-lr0.1":          # tuning retained the v10 configuration
        del runs["RUSBoost (v10)"]
    path = f"{OUT}/test_per_year.csv"
    have = done(path)
    for ty in TEST_YEARS:
        train, te = split(d, ty)
        y = te.misstate.to_numpy(int)
        ck = f"{CK}/scores_{ty}.json"
        scores = json.load(open(ck)) if os.path.exists(ck) else {}
        for label, (kind, name) in runs.items():
            if label in scores:
                continue
            if kind == "fused":
                cfg = CONFIGS[name]
                f = fit_fused(train, cfg)
                Xc, Xt = cs_matrix(te, cfg["features"]), temporal_matrix(te)
                scores[label] = f.proba(Xc, Xt).tolist()
                if label.endswith("(tuned)"):
                    scores["GBDT cross-sectional (tuned)"] = f.m_cs.decision_function(Xc).tolist()
                    scores["GBDT temporal (tuned)"] = f.m_tp.decision_function(Xt).tolist()
                    wts = dict(zip(["w_cs", "w_tp", "w_an"], f.lr.coef_[0].tolist()))
                    append(f"{OUT}/test_fusion_weights.csv", [{"test_year": ty, **wts}])
                    if ty == TEST_YEARS[-1]:
                        joblib.dump({"fused": f, "test": te, "year": ty, "config": cfg}, f"{CK}/final_year_v3.joblib")
            elif kind == "rus":
                m = rus(RUS_CONFIGS[name]).fit(train[RAW28].to_numpy(float), train.misstate.to_numpy(int))
                scores[label] = m.predict_proba(te[RAW28].to_numpy(float))[:, 1].tolist()
            else:
                m = logit_ratios().fit(np.nan_to_num(train[RATIOS14].to_numpy(float)), train.misstate.to_numpy(int))
                scores[label] = m.predict_proba(np.nan_to_num(te[RATIOS14].to_numpy(float)))[:, 1].tolist()
            json.dump(scores, open(ck, "w"))          # checkpoint after every model
        rows = [row("test", k, ty, y, np.array(v)) for k, v in scores.items() if (k, ty) not in have]
        if rows:
            append(path, rows)
        f = next(r for r in rows if r["config"] == "TrustLedger-XAI (tuned)") if rows else None
        print(f"test {ty} done" + (f": tuned fused AUC {f['auc']:.3f}" if f else ""), flush=True)

    # pooled, firm-clustered
    ys, gs, S2 = [], [], {}
    for ty in TEST_YEARS:
        te = d[d.fyear == ty]
        ys.append(te.misstate.to_numpy(int)); gs.append(te.gvkey.to_numpy())
        for k, v in json.load(open(f"{CK}/scores_{ty}.json")).items():
            S2.setdefault(k, []).append(np.array(v))
    y, g = np.concatenate(ys), np.concatenate(gs)
    S2 = {k: np.concatenate(v) for k, v in S2.items()}
    ref = "TrustLedger-XAI (tuned)"
    boot = clustered_boot(y, S2, g, ref)
    adj = holm({m: boot[m][2] for m in S2 if m != ref})
    P = pd.read_csv(path)
    out = []
    for m, s in S2.items():
        py = P[P.config == m]
        out.append({"model": m, "pooled_auc": round(roc_auc_score(y, s), 4),
                    "ci95": f"[{boot[m][0]:.4f}, {boot[m][1]:.4f}]",
                    "p_vs_tuned_fused_holm": None if m == ref else round(adj[m], 4),
                    "mean_year_auc": round(py.auc.mean(), 4),
                    "mean_year_auc_2003_2008": round(py[py.test_year <= 2008].auc.mean(), 4),
                    "mean_year_pr_auc": round(py.pr_auc.mean(), 4),
                    "mean_year_ndcg": round(py.ndcg_at_k.mean(), 4),
                    "found_within_year_top1pct": int(py.frauds_in_topk.sum())})
    R = pd.DataFrame(out).sort_values("pooled_auc", ascending=False)
    R.to_csv(f"{OUT}/test_summary.csv", index=False)
    print(R.to_string(index=False))


if __name__ == "__main__":
    d = add_scaled(load())
    {"dev": dev, "test": test}[sys.argv[1]](d)
