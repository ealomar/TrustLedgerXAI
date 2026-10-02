"""
TrustLedger-XAI - E7: decision-level sparse counterfactuals (revision 2).

Counterfactuals are now computed against the FUSED decision that flagged the
firm-year, using the exact models saved by aaer_pipeline.py for the final test
year. The search changes current-period raw accounting items only; the firm's
reporting history (lagged items) is held fixed, so the temporal features and
the isolation-forest score are recomputed from the changed items. The 14
precomputed accounting ratios and the years-observed counter are held fixed.

Objective (Section 4.2):   min  ||x' - x||_{1,MAD}   s.t.  f(x') < tau,  ||x' - x||_0 <= C
Search: greedy sparse selection. At each step every unchanged item is tried at
each of Q empirical training quantiles (so candidate values are always values
observed in the training window, including their sign), and the move with the
lowest hinge + lambda * distance is taken. Once below tau, each changed item
is pulled back toward its original value by bisection while the decision stays
below tau, and items whose full reversion keeps it below tau are dropped.
If no counterfactual is found within C changes, the artifact records
attainable = false.

    python aaer_pipeline.py   # writes final_year.joblib
    python counterfactual_bench.py
"""

import os

import joblib
import numpy as np
import pandas as pd

from aaer_pipeline import (CSV, GAP, RAW28, TEMPORAL, TRAIN_START, add_temporal,
                           temporal_matrix)

OUT = "results"
os.makedirs(OUT, exist_ok=True)
C_MAX, Q, LAM = 6, 41, 0.01


def mad_scale(X):
    med = np.median(X, 0)
    mad = np.median(np.abs(X - med), 0)
    fallback = np.mean(np.abs(X - med), 0)
    s = np.where(mad > 0, mad, fallback)
    return np.where(s > 0, s, 1.0)


def temporal_batch(row, cands):
    """Temporal feature matrix for modified current-period records (vectorised).
    Column order matches TEMPORAL: d_ (28), g_ (28), ratios (14), yrs_observed."""
    C = np.asarray(cands, float)
    lag = np.array([row[f"lag_{c}"] for c in RAW28], float)
    lag_at = float(row["lag_at_"]) if pd.notna(row["lag_at_"]) else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (C - lag) / lag_at if (np.isfinite(lag_at) and lag_at != 0) else np.full_like(C, np.nan)
        gg = np.where((lag != 0) & np.isfinite(lag), C / lag - 1, np.nan)
    fixed = np.array([row[c] for c in TEMPORAL[56:]], float)
    M = np.hstack([dd, gg, np.tile(fixed, (len(C), 1))])
    return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)


def fused_logit_batch(fused, row, cands):
    return fused.logit(np.asarray(cands, float), temporal_batch(row, cands))


def search(fused, row, x, tau_logit, grid, scale, mutable):
    cur = x.copy(); changed = []
    lg = fused_logit_batch(fused, row, [cur])[0]
    while lg >= tau_logit and len(changed) < C_MAX:
        best = None
        for j in mutable:
            if j in changed:
                continue
            cands = []
            for v in grid[j]:
                c = cur.copy(); c[j] = v; cands.append(c)
            lgs = fused_logit_batch(fused, row, cands)
            dist = np.array([np.abs((c - x) / scale).sum() for c in cands])
            obj = np.maximum(0, lgs - tau_logit + 1e-9) + LAM * dist
            k = int(np.argmin(obj))
            if best is None or obj[k] < best[0]:
                best = (obj[k], j, grid[j][k], lgs[k])
        if best is None:
            break
        _, j, v, lg = best
        cur[j] = v; changed.append(j)
    if lg >= tau_logit:
        return cur, lg, changed, False
    # pull each changed item back toward the original while staying below tau
    for j in list(changed):
        test = cur.copy(); test[j] = x[j]
        if fused_logit_batch(fused, row, [test])[0] < tau_logit:
            cur = test; changed.remove(j); continue
        lo, hi = 0.0, 1.0          # fraction of the move retained
        for _ in range(20):
            mid = (lo + hi) / 2
            t = cur.copy(); t[j] = x[j] + mid * (cur[j] - x[j])
            if fused_logit_batch(fused, row, [t])[0] < tau_logit:
                hi = mid
            else:
                lo = mid
        cur[j] = x[j] + hi * (cur[j] - x[j])
    lg = fused_logit_batch(fused, row, [cur])[0]
    return cur, lg, changed, bool(lg < tau_logit)


def main():
    S = joblib.load("final_year.joblib")
    fused, test, flagged, ty = S["fused"], S["test"], S["flagged"], S["year"]
    d = add_temporal(pd.read_csv(CSV).sort_values(["gvkey", "fyear"]).reset_index(drop=True))
    train = d[(d.fyear >= TRAIN_START) & (d.fyear <= ty - GAP)]
    Xtr = train[RAW28].to_numpy(float)
    scale = mad_scale(Xtr)
    grid = [np.unique(np.quantile(Xtr[:, j], np.linspace(0, 1, Q))) for j in range(len(RAW28))]

    Xc = test[RAW28].to_numpy(float)
    Xt = temporal_matrix(test)
    lg_all = fused.logit(Xc, Xt)
    # the vectorised rebuild must reproduce the pipeline's temporal features exactly
    for i in flagged:
        assert np.allclose(temporal_batch(test.iloc[i], [Xc[i]])[0], Xt[i], rtol=1e-12, atol=1e-12)
    tau_logit = float(np.sort(lg_all)[-len(flagged)])   # k-th highest: the flagging boundary
    tau = 1 / (1 + np.exp(-tau_logit))
    print(f"test year {ty}: tau = {tau:.4f} (log-odds {tau_logit:.4f}); {len(flagged)} flagged")

    variants = {"all 28 items": list(range(len(RAW28))),
                "financial-statement items only (excl. prcc_f, csho)":
                    [j for j, c in enumerate(RAW28) if c not in ("prcc_f", "csho")]}
    details, summaries, freqs = [], [], []
    for vname, mutable in variants.items():
        rows, freq = [], {}
        for n, i in enumerate(flagged):
            row = test.iloc[i]
            x = Xc[i].copy()
            xp, lg, changed, ok = search(fused, row, x, tau_logit, grid, scale, mutable)
            for j in changed:
                if ok:
                    freq[RAW28[j]] = freq.get(RAW28[j], 0) + 1
            rows.append({"variant": vname, "assessment": n, "assessment_id": f"AAER-{int(row.gvkey)}-{ty}",
                         "orig_score": round(float(1 / (1 + np.exp(-lg_all[i]))), 4),
                         "cf_score": round(float(1 / (1 + np.exp(-lg))), 4), "attainable": ok,
                         "items_changed": len(changed) if ok else None,
                         "l1_mad": round(float(np.abs((xp - x) / scale).sum()), 3) if ok else None,
                         "changed": ";".join(RAW28[j] for j in changed) if ok else ""})
            print(f"  [{vname[:9]}] {n:2d} ok={ok} {rows[-1]['changed']}", flush=True)
        df = pd.DataFrame(rows)
        att = df[df.attainable]
        summaries.append({
            "variant": vname, "test_year": ty, "flagged": len(df), "tau": round(tau, 4),
            "attainable": int(df.attainable.sum()),
            "attainable_pct": round(100 * df.attainable.mean(), 1),
            "median_items_changed": float(att.items_changed.median()),
            "max_items_changed": int(att.items_changed.max()),
            "median_l1_mad": round(float(att.l1_mad.median()), 3),
            "median_margin_below_tau": round(float((tau - att.cf_score).median()), 4),
            "c_max": C_MAX, "quantiles": Q})
        details.append(df)
        freqs += [{"variant": vname, "item": k, "times_changed": v}
                  for k, v in sorted(freq.items(), key=lambda kv: -kv[1])]
    pd.concat(details).to_csv(f"{OUT}/E7_counterfactual_detail.csv", index=False)
    pd.DataFrame(summaries).to_csv(f"{OUT}/E7_counterfactual.csv", index=False)
    pd.DataFrame(freqs).to_csv(f"{OUT}/E7_items_changed.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(freqs).to_string(index=False))

if __name__ == "__main__":
    main()
