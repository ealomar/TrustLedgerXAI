"""Appendix A table: all five models on common test scores (descriptive intervals)."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from aaer_pipeline import load, clustered_boot, topk_stats
from stage1 import boot_ndcg
d = load()
M = {"TrustLedger-XAI v10 (manuscript)": ("s1", "TrustLedger-XAI v10"),
     "Attempt 1: tuned configuration": ("v3", "TrustLedger-XAI (tuned)"),
     "Attempt 2: Stage 1 selected": ("s1", "Stage 1 (selected)"),
     "RUSBoost (reproduction)": ("s1", "RUSBoost (reproduction)"),
     "Logit on ratios": ("s1", "Logit on ratios")}
ys, gs, S = [], [], {m: [] for m in M}
for ty in range(2003, 2015):
    te = d[d.fyear == ty]; ys.append(te.misstate.to_numpy(int)); gs.append(te.gvkey.to_numpy())
    v3 = json.load(open(f"checkpoints_v3/scores_{ty}.json")); s1 = json.load(open(f"checkpoints_s1/scores_{ty}.json"))
    for m, (src, k) in M.items():
        S[m].append(np.array((v3 if src == "v3" else s1)[k]))
P = slice(0, 6)
B = boot_ndcg(ys[P], {m: v[P] for m, v in S.items()}, gs[P], "TrustLedger-XAI v10 (manuscript)")
y, g = np.concatenate(ys), np.concatenate(gs)
pooled = {m: np.concatenate(v) for m, v in S.items()}
A = clustered_boot(y, pooled, g, "TrustLedger-XAI v10 (manuscript)")
rows = []
for m in M:
    per = [topk_stats(ys[i], S[m][i]) for i in range(12)]
    rows.append({"model": m,
                 "ndcg_0308": np.mean([p["ndcg_at_k"] for p in per[:6]]), "ndcg_ci": (B[m][0], B[m][1]),
                 "found_0308": sum(p["frauds_in_topk"] for p in per[:6]),
                 "auc_0308": np.mean([roc_auc_score(ys[i], S[m][i]) for i in range(6)]),
                 "pooled_auc": roc_auc_score(y, pooled[m]), "auc_ci": (A[m][0], A[m][1]),
                 "ndcg_0314": np.mean([p["ndcg_at_k"] for p in per]),
                 "found_0314": sum(p["frauds_in_topk"] for p in per)})
R = pd.DataFrame(rows); R.to_csv("results_s1/appendix_summary.csv", index=False); print(R.round(4).to_string())
