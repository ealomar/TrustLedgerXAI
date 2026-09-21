"""Figures 4-8 from results/ (run after aaer_pipeline.py and counterfactual_bench.py)."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.grid": True,
                     "grid.alpha": .3, "axes.axisbelow": True})
P = pd.read_csv("results/aaer_pooled.csv"); Y = pd.read_csv("results/aaer_per_year.csv")
T = pd.read_csv("results/aaer_attribution_top10.csv")
ORDER = ["Logit on ratios", "RUSBoost (raw items)", "GBDT cross-sectional", "GBDT temporal",
         "Isolation forest", "TrustLedger-XAI (fused)"]
SHORT = {"Logit on ratios": "Logit\n(ratios)", "RUSBoost (raw items)": "RUSBoost\n(raw items)",
         "GBDT cross-sectional": "GBDT\ncross-sec.", "GBDT temporal": "GBDT\ntemporal",
         "Isolation forest": "Isolation\nforest", "TrustLedger-XAI (fused)": "Fused\n(proposed)"}
P = P.set_index("model").loc[ORDER].reset_index()
col = ["#9aa5b1"] * 5 + ["#2f4858"]; x = range(6)

fig, ax = plt.subplots(1, 2, figsize=(11, 5.5), dpi=600)
lo = [float(s.strip("[]").split(",")[0]) for s in P.auc_ci95_firm_clustered]
hi = [float(s.strip("[]").split(",")[1]) for s in P.auc_ci95_firm_clustered]
ax[0].bar(x, P.pooled_auc, yerr=[P.pooled_auc - lo, np.array(hi) - P.pooled_auc], capsize=4,
          color=col, edgecolor="black", lw=.6)
ax[0].set_ylim(.55, .8); ax[0].set_ylabel("Pooled AUC (firm-clustered 95% CI)")
ax[0].set_title("(a) Discrimination")
ax[1].bar(x, P.mean_year_pr_auc, color=col, edgecolor="black", lw=.6)
ax[1].set_ylabel("Mean per-year PR-AUC"); ax[1].set_title("(b) Precision-recall, within year")
for a in ax:
    a.set_xticks(list(x)); a.set_xticklabels([SHORT[m] for m in P.model], fontsize=8)
plt.tight_layout(); plt.savefig("results/f4.png"); plt.close()

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=600)
for i, m in enumerate(ORDER):
    d = Y[Y.model == m].sort_values("test_year")
    ax.plot(d.test_year, d.auc, marker="osv^DP"[i], ms=4, lw=2.2 if "fused" in m else 1.1,
            color="#2f4858" if "fused" in m else None, alpha=1 if "fused" in m else .8,
            label=SHORT[m].replace("\n", " "))
ax.axhline(.5, ls=":", c="grey", lw=1); ax.set_xlabel("Test fiscal year"); ax.set_ylabel("AUC")
ax.set_title("Out-of-sample AUC by test year"); ax.legend(fontsize=8, loc="lower left", ncol=3)
plt.tight_layout(); plt.savefig("results/f5.png"); plt.close()

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=600)
T = T.iloc[::-1]
c = ["#2f4858" if f.startswith("cs:") else "#c0552e" for f in T.feature]
ax.barh([f.split(":")[1] for f in T.feature], T.mean_abs_attribution, color=c, edgecolor="black", lw=.6)
ax.set_xlabel("Mean |attribution| to the fused log-odds (682 flagged firm-years)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2f4858", label="cross-sectional item"),
                   Patch(color="#c0552e", label="temporal feature")], fontsize=9, loc="lower right")
ax.set_title("Decision-level attribution: ten largest features")
plt.tight_layout(); plt.savefig("results/f6.png"); plt.close()

A = [json.loads(l) for l in open("artifacts.ndjson")]
B = pd.DataFrame([a["branch_contributions"] for a in A])
fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
ax.boxplot([B.cross_sectional, B.temporal, B.anomaly], showfliers=False)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Cross-sectional", "Temporal", "Anomaly"])
ax.axhline(0, c="grey", lw=.8); ax.set_ylabel("Contribution to fused log-odds")
ax.set_title("Branch contributions for the 682 flagged firm-years")
plt.tight_layout(); plt.savefig("results/f7.png"); plt.close()

fig, ax = plt.subplots(figsize=(9, 5), dpi=600)
n = P.within_year_top1pct_found
b = ax.bar(x, n, color=col, edgecolor="black", lw=.6)
ax.set_xticks(list(x)); ax.set_xticklabels([SHORT[m] for m in P.model], fontsize=8)
ax.set_ylabel("Misstatements identified"); ax.set_ylim(0, max(n) + 3)
ax.set_title("Misstatements in the top 1% of each test year (682 firm-years examined)")
for r, v in zip(b, n): ax.text(r.get_x() + r.get_width() / 2, v + .3, str(v), ha="center")
plt.tight_layout(); plt.savefig("results/f8.png"); plt.close()
print("figures written")
