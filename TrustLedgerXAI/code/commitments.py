"""
Commitments used by TrustLedger-XAI artifacts.

model_commitment  SHA-256 over an RFC 8785 serialisation of every parameter
                  that determines the fused decision: for each gradient-boosted
                  branch, its baseline (initial) raw prediction and, for every
                  node, feature index, threshold, missing-value routing,
                  children, leaf value and leaf flag; for the isolation forest,
                  every tree's split arrays, feature subsets and the score
                  offset; and the scaler and fusion coefficients. Omitting
                  missing-value routing or the baseline prediction would let
                  two different functions share one commitment.
input_commitment  SHA-256 over the canonical feature vector under a committed
                  feature ordering.
record_digest     SHA-256 over the canonical source record (keys + raw items).
"""

import hashlib
import math

import rfc8785


def _f(v):
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def sha(b):
    return hashlib.sha256(b).hexdigest()


def hgb_structure(model):
    trees = []
    for stage in model._predictors:          # private attribute; pinned sklearn version
        for pred in stage:
            n = pred.nodes
            trees.append([[int(n["feature_idx"][i]), _f(n["num_threshold"][i]),
                           bool(n["missing_go_to_left"][i]), int(n["left"][i]),
                           int(n["right"][i]), _f(n["value"][i]), bool(n["is_leaf"][i])]
                          for i in range(len(n))])
    return {"baseline_prediction": [_f(v) for v in model._baseline_prediction.ravel()],
            "trees": trees}


def iforest_structure(iso):
    out = []
    for est, feats in zip(iso.estimators_, iso.estimators_features_):
        t = est.tree_
        out.append({"features": [int(f) for f in feats],
                    "feature": t.feature.tolist(), "threshold": [_f(v) for v in t.threshold],
                    "left": t.children_left.tolist(), "right": t.children_right.tolist(),
                    "n_node_samples": t.n_node_samples.tolist()})
    return {"trees": out, "offset": _f(iso.offset_), "max_samples": int(iso.max_samples_)}


def model_commitment(fused, version):
    doc = {
        "version": version,
        "cross_sectional": hgb_structure(fused.m_cs),
        "temporal": hgb_structure(fused.m_tp),
        "anomaly": iforest_structure(fused.iso),
        "scaler": {"mean": [_f(v) for v in fused.scaler.mean_],
                   "scale": [_f(v) for v in fused.scaler.scale_]},
        "fusion": {"coef": [_f(v) for v in fused.lr.coef_[0]],
                   "intercept": _f(fused.lr.intercept_[0])},
    }
    return sha(rfc8785.dumps(doc))


def input_commitment(feature_names, values):
    return sha(rfc8785.dumps({"feature_order": list(feature_names),
                              "values": [_f(v) for v in values]}))


def record_digest(row, raw_items):
    return sha(rfc8785.dumps({"gvkey": int(row.gvkey), "fyear": int(row.fyear),
                              "items": {c: _f(row[c]) for c in raw_items}}))
