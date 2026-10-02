import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
fig, ax = plt.subplots(figsize=(12, 11), dpi=600)
ax.set_xlim(0, 120); ax.set_ylim(0, 110); ax.axis("off")

def group(x, y, w, h, title, right=False):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ls=(0, (4, 3)), lw=1.3))
    tx = x + w - 1.5 if right else x + 1.5
    ax.text(tx, y + h - 2.6, title, fontsize=13, fontweight="bold", va="center",
            ha="right" if right else "left")

def box(x, y, w, h, text, dashed=False, shade=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc="#ebebeb" if shade else "white", ec="black", lw=1.4,
                                ls=(0, (4, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

def path(pts, dashed=False):
    ls = (0, (4, 2)) if dashed else "-"
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.4, ls=ls, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", lw=1.4, ls=ls, color="black", shrinkA=0, shrinkB=0))

def label(x, y, s, ha="center"):
    ax.text(x, y, s, fontsize=10, style="italic", ha=ha, va="center")

C1, C2, C3, W = 4, 42, 80, 30          # column x-positions and box width
# ---------------- source records
group(2, 83, 110, 25, "Source records (SEC AAER firm-year panel)")
box(C1, 86, W, 10, "Current-period record\n(28 raw accounting items)")
box(C2, 86, W, 10, "Record digest $h_b$\n(RFC 8785, SHA-256)")
box(C3, 86, W, 10, "Firm history\n(lagged items, ratios)")
path([(C1 + W, 91), (C2, 91)])
# ---------------- engine
group(2, 32, 110, 47, "TrustLedger-XAI engine (audited organisation)", right=True)
box(C1, 59, W, 12, "Three branches:\nGBDT cross-sectional,\nGBDT temporal, isolation forest")
box(C2, 59, W, 12, "Logistic fusion on\nstandardised branch margins")
box(C3, 59, W, 12, "Decision-level attribution\n(exact, fused log-odds)\n+ sparse counterfactual")
box(C1, 38, W, 12, "Model commitment $C_M$\n(registered before\ndeployment)", dashed=True)
box(C2, 38, W, 12, "Artifact $E_i$ → canonical bytes\n→ digest $H_i$")
box(C3, 38, W, 12, "Off-chain artifact store\n(access-controlled)", shade=True)
path([(13, 86), (13, 71.3)])                                   # current record -> branches
path([(95, 86), (95, 81), (27, 81), (27, 71.3)])               # history -> branches
path([(C1 + W, 65), (C2, 65)])                                  # branches -> fusion
path([(C2 + W, 65), (C3, 65)])                                  # fusion -> attribution
path([(95, 59), (95, 54.5), (61, 54.5), (61, 50.3)])            # attribution -> artifact
path([(C2 + W, 44), (C3, 44)]); label(76, 46.3, "artifact")    # artifact -> store
# ---------------- ledger
group(2, 2, 110, 27, "Ledger anchoring layer (jointly operated)", right=True)
box(C1, 6, W, 13, "Endorsement by an\norganisation not controlled\nby the generator")
box(C2, 6, W, 13, "Write-once chaincode\n+ MVCC read-set validation")
box(C3, 6, W, 13, "World state\nexpl~ID → ($H_i$, $h_b$)\nmodel~ver → $C_M$")
path([(13, 38), (13, 19.3)], dashed=True); label(14.5, 23.6, "registered\nfirst", ha="left")
path([(57, 38), (57, 30.5), (27, 30.5), (27, 19.3)]); label(58.5, 35, "digest $H_i$ only", ha="left")
path([(C1 + W, 12.5), (C2, 12.5)])
path([(C2 + W, 12.5), (C3, 12.5)])
path([(57, 96), (57, 100.5), (116, 100.5), (116, 12.5), (C3 + W + 0.3, 12.5)], dashed=True)
label(116.8, 60, "$h_b$", ha="left")
plt.savefig("results/f1.png", bbox_inches="tight", pad_inches=0.1)
print("ok")
