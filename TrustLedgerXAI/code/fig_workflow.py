import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams.update({"font.family": "serif"})
stages = [("Stage 1   Data Preparation", ["SEC AAER\nfirm-year panel", "Expanding window,\ntwo-year gap", "Serial-fraud\nrecoding"]),
          ("Stage 2   Branch Scoring", ["Cross-sectional\nGBDT", "Temporal\nGBDT", "Isolation\nforest"]),
          ("Stage 3   Fusion", ["Firm-grouped\nout-of-fold margins", "Logistic fusion on\nstandardised margins", "Composite score;\ntop 1% flagged"]),
          ("Stage 4   Explanation", ["Decision-level\nattribution (Eq. 4)", "Sparse counterfactual\n(Eq. 5)", "Artifact assembly\n(Eq. 6)"]),
          ("Stage 5   Anchoring", ["Model commitment\nregistered first", "Canonical bytes\nand SHA-256", "Write-once key,\nendorsed and validated"])]
fig, ax = plt.subplots(figsize=(10.5, 11), dpi=600); ax.set_xlim(0, 100); ax.set_ylim(0, 110); ax.axis("off")
for s, (title, boxes) in enumerate(stages):
    y = 110 - (s + 1) * 21.5
    ax.add_patch(Rectangle((2, y), 96, 18, fill=False, ls=(0, (4, 3)), lw=1.2))
    ax.text(3.5, y + 16, title, fontsize=13, fontweight="bold", va="center")
    for b, t in enumerate(boxes):
        x = 5 + b * 31.5
        ax.add_patch(FancyBboxPatch((x, y + 2.5), 27, 9.5, boxstyle="round,pad=0.2,rounding_size=1",
                                    fc="white", ec="black", lw=1.3))
        ax.text(x + 13.5, y + 7.25, t, ha="center", va="center", fontsize=11)
        if b < 2:
            ax.annotate("", xy=(x + 31.5, y + 7.25), xytext=(x + 27, y + 7.25),
                        arrowprops=dict(arrowstyle="-|>", lw=1.3))
    if s < 4:
        ax.annotate("", xy=(50, y - 3.5), xytext=(50, y), arrowprops=dict(arrowstyle="-|>", lw=1.3))
plt.savefig("results/f3.png", bbox_inches="tight"); print("ok")
