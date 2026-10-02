"""
Branch learners for TrustLedger-XAI (revision 3).

GBDTBranch wraps either one HistGradientBoostingClassifier ("single") or an
undersampled bag of them ("bagged"). A bagged branch trains K models, each on
all positives plus a fresh random sample of r x P negatives, and its margin is
the MEAN of the K log-odds margins. Because the mean of additive decompositions
is additive, TreeSHAP stays exact:

    m(x) = (1/K) sum_k m_k(x) = (1/K) sum_k [phi_k0 + sum_j phi_kj(x)]
         = phi_0 + sum_j phi_j(x),  phi_j = (1/K) sum_k phi_kj

so Equation (4) of the manuscript is unchanged.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

RNG = 20260913


def _hgb(p, seed):
    return HistGradientBoostingClassifier(
        max_iter=p["iters"], learning_rate=p["lr"], max_leaf_nodes=p["leaves"],
        min_samples_leaf=p["min_leaf"], l2_regularization=p.get("l2", 1.0),
        max_features=p.get("max_features", 1.0), class_weight="balanced",
        random_state=seed)


class GBDTBranch:
    def __init__(self, cfg, seed=RNG):
        self.cfg, self.seed, self.models = cfg, seed, []

    def fit(self, X, y):
        p = self.cfg["params"]
        if self.cfg["kind"] == "single":
            self.models = [_hgb(p, self.seed).fit(X, y)]
            return self
        rng = np.random.default_rng(self.seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_neg = min(len(neg), int(self.cfg["ratio"] * len(pos)))
        self.models = []
        for k in range(self.cfg["bags"]):
            idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
            self.models.append(_hgb(p, self.seed + k).fit(X[idx], y[idx]))
        return self

    def decision_function(self, X):
        out = np.zeros(len(X))
        for m in self.models:               # fixed-order accumulation, batch-independent
            out = out + m.decision_function(X)
        return out / len(self.models)

    def explain(self, X):
        import shap
        base, phi = 0.0, np.zeros(X.shape, float)
        for m in self.models:
            e = shap.TreeExplainer(m)
            base = base + float(np.ravel(e.expected_value)[0])
            phi = phi + e.shap_values(X)
        return base / len(self.models), phi / len(self.models)
