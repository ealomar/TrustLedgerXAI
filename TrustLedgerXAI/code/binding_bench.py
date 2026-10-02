"""
TrustLedger-XAI - E6: bound verification (revision 2).

Uses the fused model and the 56 real decision-level artifacts of the final test
year produced by aaer_pipeline.py. Changes from the previous version:

  * the model commitment is REGISTERED on the ledger (key model~<version>)
    before any assessment is anchored; a verifier accepts an artifact only if
    its model commitment equals the registered one and the registration block
    precedes the artifact's anchoring block;
  * the commitment covers missing-value routing, the baseline prediction, the
    isolation forest, the scaler and the fusion coefficients (commitments.py);
  * the verifier recomputes the complete decision - composite score, branch
    contributions and every attribution - and requires bit-exact agreement;
  * tests include subtle fabrications (two attributions swapped; one attribution
    changed by one unit in the last place), not only a sign-flipped rationale;
  * a commitment-completeness check shows that the previous commitment format
    (node structure without routing or baseline) cannot distinguish two
    functions that make different decisions.

    python binding_bench.py
"""

import copy
import hashlib
import json
import os
import struct

import joblib
import numpy as np
import pandas as pd
import rfc8785

from aaer_pipeline import MODEL_VERSION, RAW28, TEMPORAL, build_artifacts, temporal_matrix
from commitments import model_commitment, sha
from anchoring_ledger import MVCCLedger

OUT = "results"
os.makedirs(OUT, exist_ok=True)


def ulp(v):
    return struct.unpack("<d", struct.pack("<q", struct.unpack("<q", struct.pack("<d", v))[0] + 1))[0]


def decision_digest(a):
    keys = ["composite_score", "composite_log_odds", "branch_contributions", "base_value",
            "attributions", "fusion"]
    return sha(rfc8785.dumps({k: a[k] for k in keys}))


class Verifier:
    def __init__(self, ledger, model, test):
        self.L, self.model, self.test = ledger, model, test
        self.index = {f"AAER-{int(r.gvkey)}-{int(r.fyear)}": i for i, r in
                      enumerate(test.itertuples())}

    def unbound(self, a):
        e = self.L.query(f"expl~{a['assessment_id']}")
        return e is not None and e["H"] == sha(rfc8785.dumps(a))

    def bound(self, a, record_pos=None):
        if not self.unbound(a):
            return False, "digest mismatch"
        reg = self.L.query(f"model~{a['model_version']}")
        if reg is None or reg["H"] != a["model_commitment"]:
            return False, "model not registered"
        anchored = self.L.query(f"expl~{a['assessment_id']}")
        if reg["height"] >= anchored["height"]:
            return False, "model registered after assessment"
        if model_commitment(self.model, a["model_version"]) != reg["H"]:
            return False, "verifier's model does not match registration"
        pos = self.index[a["assessment_id"]] if record_pos is None else record_pos
        re_a, _ = build_artifacts(int(self.test.iloc[pos].fyear) , self.test, self.model,
                                  np.array([pos]))
        re_a = re_a[0]
        if re_a["input_commitment"] != a["input_commitment"]:
            return False, "input commitment mismatch"
        if decision_digest(re_a) != decision_digest(a):
            return False, "recomputed decision differs"
        return True, "ok"


def anchor_all(L, arts, model_first=True, commitment=None):
    if model_first:
        L.submit(f"model~{MODEL_VERSION}", commitment, "", "AuditedCo", ["AuditFirm"])
        L.cut_block()                      # registration commits before any assessment
    for a in arts:
        L.submit(f"expl~{a['assessment_id']}", sha(rfc8785.dumps(a)), a["record_digest"],
                 "AuditedCo", ["AuditFirm"])
    if not model_first:
        L.submit(f"model~{MODEL_VERSION}", commitment, "", "AuditedCo", ["AuditFirm"])
    L.cut_block()


def legacy_commitment(m):
    """The previous (v1) commitment: node arrays without routing or baseline."""
    trees = []
    for stage in m._predictors:
        for p in stage:
            n = p.nodes
            trees.append([[int(n["feature_idx"][i]), float(n["num_threshold"][i]),
                           int(n["left"][i]), int(n["right"][i]), float(n["value"][i]),
                           bool(n["is_leaf"][i])] for i in range(len(n))])
    return sha(rfc8785.dumps({"n_trees": len(trees), "trees": trees}))


def main():
    S = joblib.load("final_year.joblib")
    fused, test, flagged = S["fused"], S["test"], S["flagged"]
    arts = [json.loads(l) for l in open("artifacts.ndjson")][-len(flagged):]
    assert arts[0]["assessment_id"].endswith(str(S["year"]))
    mc = model_commitment(fused, MODEL_VERSION)
    assert all(a["model_commitment"] == mc for a in arts)

    L = MVCCLedger(["AuditedCo", "AuditFirm", "Regulator"], "AuditedCo")
    anchor_all(L, arts, True, mc)
    V = Verifier(L, fused, test)
    n = len(arts); rows = []

    def run(name, expected, fn):
        res = [fn(k, a) for k, a in enumerate(arts)]
        ok = sum(r[0] for r in res)
        reasons = sorted({r[1] for r in res})
        rows.append({"scenario": name, "expected": expected, "trials": n,
                     "accepted": ok, "rejected": n - ok, "reasons": "; ".join(reasons)})
        print(rows[-1], flush=True)

    run("Honest artifact", "accept", lambda k, a: V.bound(a))

    def fabricated(k, a, how):
        f = copy.deepcopy(a)
        at = f["attributions"]
        if how == "negate":
            f["attributions"] = {q: -v for q, v in at.items()}
        elif how == "swap":
            top = sorted(at, key=lambda q: -abs(at[q]))[:2]
            at[top[0]], at[top[1]] = at[top[1]], at[top[0]]
        elif how == "ulp":
            q = max(at, key=lambda q: abs(at[q])); at[q] = ulp(at[q])
        # the fabricated rationale is anchored at decision time on a fresh ledger
        L2 = MVCCLedger(["AuditedCo", "AuditFirm", "Regulator"], "AuditedCo")
        anchor_all(L2, [f], True, mc)
        V2 = Verifier(L2, fused, test)
        return V2.unbound(f), V2.bound(f)

    for how, label in [("negate", "Fabricated rationale (all attributions negated)"),
                       ("swap", "Fabricated rationale (two largest attributions swapped)"),
                       ("ulp", "Fabricated rationale (one attribution +1 ULP)")]:
        unb = []
        def fn(k, a, how=how):
            u, b = fabricated(k, a, how); unb.append(u); return b
        run(label + " - bound", "reject", fn)
        rows.append({"scenario": label + " - unbound", "expected": "accept (non-guarantee)",
                     "trials": n, "accepted": sum(unb), "rejected": n - sum(unb),
                     "reasons": "digest matches anchored value"})
        print(rows[-1], flush=True)

    run("Genuine rationale presented against a different record", "reject",
        lambda k, a: V.bound(a, record_pos=int(flagged[(k + 1) % n])))

    # substituted model: operator decides with an unregistered model
    alt = copy.deepcopy(fused)
    from sklearn.ensemble import HistGradientBoostingClassifier
    alt.lr = copy.deepcopy(fused.lr); alt.lr.coef_ = fused.lr.coef_ * np.array([[1.0, 0.5, 1.0]])
    alt_arts, _ = build_artifacts(S["year"], test, alt, flagged)
    for a in alt_arts:
        a["model_commitment"] = model_commitment(alt, MODEL_VERSION)
    L3 = MVCCLedger(["AuditedCo", "AuditFirm", "Regulator"], "AuditedCo")
    anchor_all(L3, alt_arts, True, mc)             # the benign model is what was registered
    V3 = Verifier(L3, fused, test)
    res = [V3.bound(a) for a in alt_arts]
    rows.append({"scenario": "Decision made with an unregistered model", "expected": "reject",
                 "trials": n, "accepted": sum(r[0] for r in res),
                 "rejected": n - sum(r[0] for r in res),
                 "reasons": "; ".join(sorted({r[1] for r in res}))})
    print(rows[-1], flush=True)
    for a in alt_arts:
        a["model_commitment"] = mc                  # ...and claims the registered one
    L3b = MVCCLedger(["AuditedCo", "AuditFirm", "Regulator"], "AuditedCo")
    anchor_all(L3b, alt_arts, True, mc)
    V3b = Verifier(L3b, fused, test)
    res = [V3b.bound(a) for a in alt_arts]
    rows.append({"scenario": "Decision made with an unregistered model, registered commitment claimed",
                 "expected": "reject", "trials": n, "accepted": sum(r[0] for r in res),
                 "rejected": n - sum(r[0] for r in res),
                 "reasons": "; ".join(sorted({r[1] for r in res}))})
    print(rows[-1], flush=True)

    # late registration: model commitment registered after the assessments
    L4 = MVCCLedger(["AuditedCo", "AuditFirm", "Regulator"], "AuditedCo")
    for a in arts:
        L4.submit(f"expl~{a['assessment_id']}", sha(rfc8785.dumps(a)), a["record_digest"],
                  "AuditedCo", ["AuditFirm"])
    L4.cut_block()
    L4.submit(f"model~{MODEL_VERSION}", mc, "", "AuditedCo", ["AuditFirm"]); L4.cut_block()
    V4 = Verifier(L4, fused, test)
    res = [V4.bound(a) for a in arts]
    rows.append({"scenario": "Model registered after the assessments", "expected": "reject",
                 "trials": n, "accepted": sum(r[0] for r in res),
                 "rejected": n - sum(r[0] for r in res),
                 "reasons": "; ".join(sorted({r[1] for r in res}))})
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/E6_binding.csv", index=False)

    # ---------------------------------------------------- commitment completeness
    comp = []
    Xc, Xt = test[RAW28].to_numpy(float), temporal_matrix(test)
    base_lg = fused.logit(Xc, Xt)

    def variant(label, mutate):
        v = copy.deepcopy(fused); mutate(v)
        lg = v.logit(Xc, Xt)
        comp.append({"variant": label,
                     "legacy_commitment_changes": legacy_commitment(v.m_cs) != legacy_commitment(fused.m_cs)
                     or legacy_commitment(v.m_tp) != legacy_commitment(fused.m_tp),
                     "revised_commitment_changes": model_commitment(v, MODEL_VERSION) != mc,
                     "firm_years_scored_differently": int((np.abs(lg - base_lg) > 0).sum()),
                     "flag_set_changes": int(len(set(np.argsort(-lg, kind='stable')[:len(flagged)])
                                                 ^ set(flagged)) // 2)})
        print(comp[-1], flush=True)

    def shift_baseline(v):
        v.m_tp._baseline_prediction = v.m_tp._baseline_prediction + 0.5

    def flip_routing(v):
        for stage in v.m_tp._predictors:
            for p in stage:
                split = ~p.nodes["is_leaf"].astype(bool)
                p.nodes["missing_go_to_left"][split] = 1 - p.nodes["missing_go_to_left"][split]

    def change_fusion(v):
        v.lr.coef_ = v.lr.coef_ * np.array([[1.0, 1.0, 0.0]])

    variant("Temporal-branch baseline prediction shifted by 0.5", shift_baseline)
    variant("Missing-value routing flipped in every temporal split", flip_routing)
    variant("Anomaly-branch fusion weight set to zero", change_fusion)
    pd.DataFrame(comp).to_csv(f"{OUT}/E6_commitment_completeness.csv", index=False)


if __name__ == "__main__":
    main()
