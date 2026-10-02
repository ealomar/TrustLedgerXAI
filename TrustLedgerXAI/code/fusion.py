"""Logistic fusion of the three branch margins (kept in its own module so saved models load from any script)."""
import numpy as np


class Fused:
    """Logistic fusion on standardised branch margins."""

    def __init__(self, m_cs, m_tp, iso, scaler, lr):
        self.m_cs, self.m_tp, self.iso, self.scaler, self.lr = m_cs, m_tp, iso, scaler, lr

    def margins(self, Xc, Xt):
        return np.column_stack([self.m_cs.decision_function(Xc),
                                self.m_tp.decision_function(Xt),
                                -self.iso.score_samples(Xc)])

    def logit(self, Xc, Xt):
        """Fused log-odds with a fixed, batch-independent summation order.

        A BLAS matrix-vector product (z @ w) may reduce in an order that depends
        on the batch shape, so the same record can receive a log-odds that
        differs by one unit in the last place when scored alone rather than in
        a batch. Bound verification recomputes records individually and
        requires bit-exact agreement, so the sum is written out explicitly.
        """
        z = (self.margins(Xc, Xt) - self.scaler.mean_) / self.scaler.scale_
        w = self.lr.coef_[0]
        return ((self.lr.intercept_[0] + z[:, 0] * w[0]) + z[:, 1] * w[1]) + z[:, 2] * w[2]

    def proba(self, Xc, Xt):
        return 1 / (1 + np.exp(-self.logit(Xc, Xt)))
