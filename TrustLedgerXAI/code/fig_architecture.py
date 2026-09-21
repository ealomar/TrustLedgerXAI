import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(12, 10.5), dpi=600)
ax.set_xlim(0, 120); ax.set_ylim(0, 105); ax.axis("off")

def group(x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ls=(0, (4, 3)), lw=1.3))
    ax.text(x + 1.5, y + h - 2.6, title, fontsize=13, fontweight="bold", va="center")

def box(x, y, w, h, text, dashed=False, shade=False, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc="#ebebeb" if shade else "white", ec="black", lw=1.4,
                                ls=(0, (4, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return (x, y, w, h)

def arrow(p, q, style="-|>", ls="-", label=None, lx=0, ly=0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, lw=1.4, ls=ls, color="black"))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=10.5, style="italic")

group(2, 83, 116, 21, "Source records (SEC AAER firm-year panel)")
c = box(6, 86, 32, 10, "Record digest $h_b$\n(RFC 8785, SHA-256)", fs=10)
a = box(46, 86, 30, 10, "Current-period record\n(28 raw accounting items)", fs=10)
b = box(84, 86, 31, 10, "Firm history\n(lagged items, ratios)", fs=10)
arrow((46, 91), (38, 91))

group(2, 36, 116, 42, "TrustLedger-XAI engine")
d = box(6, 57, 34, 12, "Branches: GBDT cross-sectional,\nGBDT temporal, isolation forest")
e = box(46, 57, 30, 12, "Logistic fusion on\nstandardised branch margins")
f = box(82, 57, 33, 12, "Decision-level attribution\n(exact, fused log-odds)\n+ sparse counterfactual")
g = box(6, 38.5, 34, 12, "Model commitment $C_M$\n(registered before deployment)", dashed=True)
h = box(46, 38.5, 30, 12, "Artifact $E_i$ \u2192 canonical bytes\n\u2192 digest $H_i$")
i = box(82, 38.5, 33, 12, "Off-chain artifact store\n(access-controlled)", shade=True)
arrow((61, 86), (30, 69.5)); arrow((99, 86), (40, 67))
arrow((40, 63), (46, 63)); arrow((76, 63), (82, 63))
arrow((98, 57), (68, 50.5)); arrow((76, 44.5), (82, 44.5), label="artifact", lx=-3.5, ly=-3.2)

group(2, 2, 116, 29, "Ledger anchoring layer")
j = box(6, 7, 33, 13, "Endorsement by an organisation\nnot controlled by the generator")
k = box(45, 7, 32, 13, "Write-once chaincode\n+ MVCC read-set validation")
l = box(83, 7, 32, 13, "World state\nexpl~ID \u2192 ($H_i$, $h_b$)\nmodel~ver \u2192 $C_M$")
arrow((39, 13.5), (45, 13.5)); arrow((77, 13.5), (83, 13.5))
arrow((61, 38.5), (22.5, 20), label="digests only", lx=-2, ly=2)
arrow((23, 38.5), (18, 20), ls="--")
plt.savefig("results/f1.png", bbox_inches="tight"); print("ok")
