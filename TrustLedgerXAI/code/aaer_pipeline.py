"""
TrustLedger-XAI - detection pipeline on the Bao et al. (2020) AAER panel (revision 2).

Changes from the previous version, each made in response to review:
  * fusion weights are fitted on out-of-fold scores from StratifiedGroupKFold
    grouped by firm (gvkey), so no firm contributes to both the fitting and
    the scoring of an out-of-fold prediction;
  * the fusion operates on standardised branch margins (GBDT log-odds and the
    isolation-forest score), so fitted weights are comparable in magnitude;
  * because the fused log-odds is linear in the standardised margins, and each
    GBDT margin is exactly the sum of its TreeSHAP values plus a base value,
    feature attributions of the FUSED decision are exact:
        phi_fused(j) = (w_k / sigma_k) * phi_k(j)
    The anchored artifact therefore explains the score that triggered the flag;
  * RUSBoost on the 28 raw items (the model class of Bao et al.) is added as a
    baseline;
  * PR-AUC, within-year top-1% counts and per-year mean AUC are reported, and
    uncertainty uses a firm-clustered bootstrap with Holm adjustment;
  * artifacts are produced for the within-year top 1% of every test year
    (682 firm-years), each carrying a record digest, a model commitment and an
    input commitment.

    pip install pandas numpy scikit-learn shap rfc8785 imbalanced-learn joblib
    python aaer_pipeline.py
"""

import hashlib
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import rfc8785
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from commitments import input_commitment, model_commitment, record_digest
from fusion import Fused

warnings.filterwarnings("ignore")
RNG = 20260913
OUT = "results"
os.makedirs(OUT, exist_ok=True)

CSV = "uscecchini28.csv"
URL = ("https://raw.githubusercontent.com/JarFraud/FraudDetection/master/"
       "data_FraudDetection_JAR2020.csv")

RAW28 = ["act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
         "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
         "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f"]
RATIOS14 = ["dch_wc", "ch_rsst", "dch_rec", "dch_inv", "soft_assets", "ch_cs",
            "ch_cm", "ch_roa", "issue", "bm", "dpi", "reoa", "EBIT", "ch_fcf"]
TEMPORAL = ([f"d_{c}" for c in RAW28] + [f"g_{c}" for c in RAW28]
            + RATIOS14 + ["yrs_observed"])

TEST_YEARS = list(range(2003, 2015))
GAP, TRAIN_START, TOPK_FRAC = 2, 1991, 0.01
N_BOOT = 1000
MODEL_VERSION = "trustledger-fused-2.0.0"


# --------------------------------------------------------------------- data
def load():
    if not os.path.exists(CSV):
        sys.exit(f"missing {CSV}\n  curl -L -o {CSV} {URL}")
    d = pd.read_csv(CSV).sort_values(["gvkey", "fyear"]).reset_index(drop=True)
    return add_temporal(d)


def add_temporal(d):
    """Within-firm changes: d_ = change scaled by lagged total assets,
    g_ = relative change. Lags only look backward."""
    g = d.groupby("gvkey")
    lag_at = g["at"].shift(1)
    d["lag_at_"] = lag_at
    for c in RAW28:
        lag = g[c].shift(1)
        d[f"lag_{c}"] = lag
        d[f"d_{c}"] = (d[c] - lag) / lag_at.replace(0, np.nan)
        d[f"g_{c}"] = (d[c] / lag.replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
    d["yrs_observed"] = g.cumcount()
    return d


def temporal_matrix(frame):
    return np.nan_to_num(frame[TEMPORAL].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)


def recode_serial_fraud(train, test_year, d):
    test_cases = set(d.loc[(d.fyear == test_year) & (d.misstate == 1), "p_aaer"].dropna())
    train = train.copy()
    spans = train.p_aaer.isin(test_cases)
    train.loc[spans, "misstate"] = 0
    return train, int(spans.sum())


# ------------------------------------------------------------------ metrics
def ndcg_at_k(y, score, k):
    order = np.argsort(-score, kind="stable")[:k]
    dcg = np.sum(y[order] / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(y)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else np.nan


def topk_stats(y, score, frac=TOPK_FRAC):
    k = max(1, int(round(frac * len(y))))
    order = np.argsort(-score, kind="stable")[:k]
    caught = int(y[order].sum())
    return {"k": k, "frauds_in_topk": caught, "precision_at_k": caught / k,
            "recall_at_k": caught / max(1, int(y.sum())), "ndcg_at_k": ndcg_at_k(y, score, k)}


# ------------------------------------------------------------------- models
def gbdt():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, class_weight="balanced", random_state=RNG)


def logit_ratios():
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RNG))


def rusboost():
    return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                              n_estimators=1000, learning_rate=0.1, random_state=RNG)


def oof_margin(X, y, groups, folds=5):
    """Out-of-fold GBDT log-odds, folds grouped by firm."""
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(folds, shuffle=True, random_state=RNG)
    for tr, va in cv.split(X, y, groups):
        m = gbdt().fit(X[tr], y[tr])
        oof[va] = m.decision_function(X[va])
    return oof


def run_year(d, ty):
    train = d[(d.fyear >= TRAIN_START) & (d.fyear <= ty - GAP)]
    test = d[d.fyear == ty]
    train, recoded = recode_serial_fraud(train, ty, d)
    Xc_tr, Xc_te = train[RAW28].to_numpy(float), test[RAW28].to_numpy(float)
    Xt_tr, Xt_te = temporal_matrix(train), temporal_matrix(test)
    y_tr, y_te = train.misstate.to_numpy(int), test.misstate.to_numpy(int)
    g_tr = train.gvkey.to_numpy()

    m_cs = gbdt().fit(Xc_tr, y_tr)
    m_tp = gbdt().fit(Xt_tr, y_tr)
    iso = IsolationForest(n_estimators=300, random_state=RNG).fit(Xc_tr)

    oof = np.column_stack([oof_margin(Xc_tr, y_tr, g_tr), oof_margin(Xt_tr, y_tr, g_tr),
                           -iso.score_samples(Xc_tr)])
    scaler = StandardScaler().fit(oof)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=RNG).fit(scaler.transform(oof), y_tr)
    fused = Fused(m_cs, m_tp, iso, scaler, lr)

    m_lr = logit_ratios().fit(np.nan_to_num(train[RATIOS14].to_numpy(float)), y_tr)
    m_rb = rusboost().fit(Xc_tr, y_tr)

    scores = {
        "Logit on ratios": m_lr.predict_proba(np.nan_to_num(test[RATIOS14].to_numpy(float)))[:, 1],
        "RUSBoost (raw items)": m_rb.predict_proba(Xc_te)[:, 1],
        "GBDT cross-sectional": m_cs.decision_function(Xc_te),
        "GBDT temporal": m_tp.decision_function(Xt_te),
        "Isolation forest": -iso.score_samples(Xc_te),
        "TrustLedger-XAI (fused)": fused.proba(Xc_te, Xt_te),
    }
    rows = []
    for name, s in scores.items():
        rows.append({"test_year": ty, "model": name, "n_test": len(y_te),
                     "n_fraud": int(y_te.sum()), "recoded_train_obs": recoded,
                     "auc": roc_auc_score(y_te, s), "pr_auc": average_precision_score(y_te, s),
                     **topk_stats(y_te, s)})
    wrow = {"test_year": ty, "w_cross_sectional": lr.coef_[0][0], "w_temporal": lr.coef_[0][1],
            "w_anomaly": lr.coef_[0][2], "intercept": lr.intercept_[0]}
    return rows, wrow, y_te, scores, test, fused


def build_artifacts(ty, test, fused, flagged):
    """Explanation artifacts of the fused decision for the flagged firm-years."""
    import shap
    Xc, Xt = test[RAW28].to_numpy(float), temporal_matrix(test)
    w, b = fused.lr.coef_[0], fused.lr.intercept_[0]
    mu, sd = fused.scaler.mean_, fused.scaler.scale_
    e_cs, e_tp = shap.TreeExplainer(fused.m_cs), shap.TreeExplainer(fused.m_tp)
    sv_cs = e_cs.shap_values(Xc[flagged]); sv_tp = e_tp.shap_values(Xt[flagged])
    base_cs = float(np.ravel(e_cs.expected_value)[0]); base_tp = float(np.ravel(e_tp.expected_value)[0])
    marg = fused.margins(Xc[flagged], Xt[flagged])
    z = (marg - mu) / sd
    flogit = fused.logit(Xc[flagged], Xt[flagged])
    mc = model_commitment(fused, MODEL_VERSION)
    arts, check = [], []
    for r, i in enumerate(flagged):
        row = test.iloc[i]
        attr = {f"cs:{c}": float(w[0] / sd[0] * v) for c, v in zip(RAW28, sv_cs[r])}
        attr.update({f"tp:{c}": float(w[1] / sd[1] * v) for c, v in zip(TEMPORAL, sv_tp[r])})
        branch = {"cross_sectional": float(w[0] * z[r, 0]), "temporal": float(w[1] * z[r, 1]),
                  "anomaly": float(w[2] * z[r, 2])}
        # base value of the fused log-odds before feature attribution:
        base = float(b + w[0] * (base_cs - mu[0]) / sd[0] + w[1] * (base_tp - mu[1]) / sd[1]
                     + branch["anomaly"])
        check.append(abs(base + sum(attr.values()) - flogit[r]))
        arts.append({
            "assessment_id": f"AAER-{int(row.gvkey)}-{int(row.fyear)}",
            "record_digest": record_digest(row, RAW28),
            "model_version": MODEL_VERSION,
            "model_commitment": mc,
            "input_commitment": input_commitment(RAW28 + TEMPORAL,
                                                 list(Xc[i]) + list(Xt[i])),
            "generated_at": f"{ty + 1}-06-30T00:00:00Z",
            "attribution_space": "fused_log_odds",
            "composite_score": float(1 / (1 + np.exp(-flogit[r]))),
            "composite_log_odds": float(flogit[r]),
            "branch_contributions": branch,
            "fusion": {"intercept": float(b), "weights": [float(v) for v in w]},
            "base_value": base,
            "attributions": attr,
        })
    return arts, max(check)


# ------------------------------------------------------------ clustered bootstrap
def clustered_boot(y, scores, groups, ref, n=N_BOOT, seed=RNG):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    names = list(scores)
    aucs = {m: [] for m in names}
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[p] for p in pick])
        if y[idx].sum() == 0:
            continue
        for m in names:
            aucs[m].append(roc_auc_score(y[idx], scores[m][idx]))
    out = {}
    for m in names:
        a = np.array(aucs[m]); r = np.array(aucs[ref])
        diff = r - a
        p = 1.0 if m == ref else min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
        out[m] = (np.percentile(a, 2.5), np.percentile(a, 97.5), p, len(a))
    return out


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


# ---------------------------------------------------------------------- main
def main():
    d = load()
    print(f"panel: {len(d):,} firm-years, {int(d.misstate.sum())} misstated, "
          f"{d.p_aaer.nunique()} cases, {d.fyear.min()}-{d.fyear.max()}")
    all_rows, wrows, ys, groups, pooled, artifacts, add_err = [], [], [], [], {}, [], []
    for ty in TEST_YEARS:
        rows, wrow, y_te, scores, test, fused = run_year(d, ty)
        all_rows += rows; wrows.append(wrow); ys.append(y_te); groups.append(test.gvkey.to_numpy())
        for k, v in scores.items():
            pooled.setdefault(k, []).append(v)
        k = max(1, int(round(TOPK_FRAC * len(y_te))))
        flagged = np.argsort(-scores["TrustLedger-XAI (fused)"], kind="stable")[:k]
        arts, err = build_artifacts(ty, test, fused, flagged)
        artifacts += arts; add_err.append(err)
        if ty == TEST_YEARS[-1]:
            joblib.dump({"fused": fused, "test": test, "flagged": flagged, "year": ty,
                         "train_range": (TRAIN_START, ty - GAP)}, "final_year.joblib")
        f = rows[-1]
        print(f"  {ty}: n={len(y_te):,} fraud={int(y_te.sum())} fused AUC={f['auc']:.3f} "
              f"top1%={f['frauds_in_topk']}/{f['k']} w={np.round(list(wrow.values())[1:4], 2)}",
              flush=True)

    per_year = pd.DataFrame(all_rows); per_year.to_csv(f"{OUT}/aaer_per_year.csv", index=False)
    pd.DataFrame(wrows).to_csv(f"{OUT}/fusion_weights_by_year.csv", index=False)

    y = np.concatenate(ys); grp = np.concatenate(groups)
    S = {k: np.concatenate(v) for k, v in pooled.items()}
    ref = "TrustLedger-XAI (fused)"
    boot = clustered_boot(y, S, grp, ref)
    adj = holm({m: boot[m][2] for m in S if m != ref})
    summary = []
    for m, s in S.items():
        py = per_year[per_year.model == m]
        summary.append({
            "model": m,
            "pooled_auc": round(roc_auc_score(y, s), 4),
            "auc_ci95_firm_clustered": f"[{boot[m][0]:.4f}, {boot[m][1]:.4f}]",
            "p_vs_fused": round(boot[m][2], 4) if m != ref else None,
            "p_vs_fused_holm": round(adj[m], 4) if m != ref else None,
            "pooled_pr_auc": round(average_precision_score(y, s), 4),
            "mean_year_auc": round(py.auc.mean(), 4),
            "mean_year_auc_2003_2008": round(py[py.test_year <= 2008].auc.mean(), 4),
            "mean_year_pr_auc": round(py.pr_auc.mean(), 4),
            "within_year_top1pct_found": int(py.frauds_in_topk.sum()),
            "within_year_top1pct_examined": int(py.k.sum()),
            "mean_year_ndcg_at_1pct": round(py.ndcg_at_k.mean(), 4),
            "boot_reps": boot[m][3],
        })
    summ = pd.DataFrame(summary); summ.to_csv(f"{OUT}/aaer_pooled.csv", index=False)
    print(f"\n=== test years {TEST_YEARS[0]}-{TEST_YEARS[-1]} (n={len(y):,}, "
          f"misstated={int(y.sum())}, firms={len(np.unique(grp)):,}) ===")
    print(summ.to_string(index=False))

    with open("artifacts.ndjson", "w") as f:
        for a in artifacts:
            f.write(json.dumps(a) + "\n")
    att = pd.DataFrame([a["attributions"] for a in artifacts]).abs().mean()
    top = att.sort_values(ascending=False).head(10).rename("mean_abs_attribution")
    top.to_csv(f"{OUT}/aaer_attribution_top10.csv", index_label="feature")
    bc = pd.DataFrame([a["branch_contributions"] for a in artifacts])
    bc.abs().mean().rename("mean_abs_contribution").to_csv(
        f"{OUT}/aaer_branch_contributions.csv", index_label="branch")
    print(f"\nartifacts: {len(artifacts)}; max additivity error {max(add_err):.2e}")
    print(top.to_string())
    print(bc.abs().mean().to_string())
    dig = hashlib.sha256(rfc8785.dumps(artifacts[0])).hexdigest()
    print(f"first artifact digest: {dig}")


if __name__ == "__main__":
    main()
