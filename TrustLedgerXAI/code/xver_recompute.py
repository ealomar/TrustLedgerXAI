"""
TrustLedger-XAI - E6b: honest recomputation across library versions.

Bound verification assumes that an honest verifier recomputing an attribution
obtains bit-identical values. This script recomputes the 56 final-year
decision-level artifacts under whatever shap / numpy build is installed and
appends the outcome to results/E6_crossversion.csv. Run it in two or more
environments that share the pinned scikit-learn version (the model object is
loaded with joblib, which requires the same scikit-learn).
"""
import csv, json, os, platform
import joblib, numpy as np, shap, sklearn, rfc8785
from aaer_pipeline import build_artifacts
from binding_bench import decision_digest

S = joblib.load("final_year.joblib")
arts = [json.loads(l) for l in open("artifacts.ndjson")][-len(S["flagged"]):]
import sys
mode = sys.argv[1] if len(sys.argv) > 1 else "batch"
if mode == "batch":
    re_a, _ = build_artifacts(S["year"], S["test"], S["fused"], S["flagged"])
else:   # one record at a time, as a verifier checking a single assessment would
    re_a = [build_artifacts(S["year"], S["test"], S["fused"], np.array([i]))[0][0] for i in S["flagged"]]
exact = sum(decision_digest(a) == decision_digest(b) for a, b in zip(arts, re_a))
maxdiff = max(abs(a["attributions"][k] - b["attributions"][k]) for a, b in zip(arts, re_a)
              for k in a["attributions"])
ulps = max(abs(int(np.float64(a["attributions"][k]).view(np.int64)) - int(np.float64(b["attributions"][k]).view(np.int64)))
           for a, b in zip(arts, re_a) for k in a["attributions"])
os.makedirs("results", exist_ok=True)
path = "results/E6_crossversion.csv"; new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["mode", "python", "shap", "numpy", "sklearn", "artifacts", "bit_exact", "max_abs_diff", "max_ulp_diff"])
    w.writerow([mode, platform.python_version(), shap.__version__, np.__version__, sklearn.__version__,
                len(arts), exact, f"{maxdiff:.3e}", ulps])
print(f"{mode} shap {shap.__version__}: {exact}/{len(arts)} bit-exact, max |diff| {maxdiff:.3e}, max ULP {ulps}")
