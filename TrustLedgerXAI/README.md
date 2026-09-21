# TrustLedger-XAI — code and reproducibility package

Accompanies the manuscript submitted to *Systems* (MDPI). Section, table, figure and appendix numbers below refer to that manuscript; they are unchanged from manuscript v11.

Every number, table and figure in the manuscript is produced by the scripts
below; the tables in the manuscript were generated directly from `results/`.

## Data

The SEC AAER panel of Bao, Ke, Li, Yu and Zhang (2020), JAR 58(1):

    curl -L -o uscecchini28.csv https://raw.githubusercontent.com/JarFraud/FraudDetection/master/data_FraudDetection_JAR2020.csv

146,045 firm-years, 1990–2014, 964 misstated firm-years, 412 cases.

## Environment used for the reported results

Linux x86-64, one core of an Intel Xeon at 2.10 GHz; Python 3.12.3,
scikit-learn 1.8.0, shap 0.52.0 (0.48.0 for the cross-version test),
imbalanced-learn 0.14.2, NumPy 2.4.4, pandas 3.0.2, rfc8785 0.1.4, joblib 1.5.3;
Node.js 22.22.2 with canonicalize 5.0.0. Seed 20260913 throughout.

    pip install pandas numpy scikit-learn shap rfc8785 imbalanced-learn joblib matplotlib
    npm install canonicalize@5.0.0

The model object saved in `final_year.joblib` must be loaded with the same
scikit-learn version, and `commitments.py` reads scikit-learn internal
attributes (`_predictors`, `_baseline_prediction`); pin scikit-learn 1.8.0.

## Order of execution (run from the directory containing the CSV)

| Step | Command | Produces | Manuscript | Runtime (1 core) |
|---|---|---|---|---|
| 1 | `python aaer_pipeline.py` | per-year and pooled metrics, fusion weights, 682 artifacts, `final_year.joblib` | Tables 5, 7; Figs 4–8 | ~40 min |
| 2 | `python counterfactual_bench.py` | E7 counterfactuals (two variants) | Table 8 | ~8 min |
| 3 | `python binding_bench.py` | E6 bound verification, commitment completeness | Tables 14, 15 | ~2 min |
| 4 | `python xver_recompute.py batch` and `python xver_recompute.py single` (repeat in a second shap environment) | E6 cross-version | Table 16 | <1 min |
| 5 | `python anchoring_bench.py` | E1 determinism, E2 conformance, E3 cost, E4 Merkle | Tables 9, 11, 12, 13 | ~2 min |
| 6 | `python crossimpl_export.py real && node crossimpl_verify.mjs real` (and `stress`) | E5 cross-implementation | Table 10 | <1 min |
| 7 | `python figs.py`, `python fig_architecture.py`, `python fig_workflow.py` | Figures 1, 3–8 at 600 dpi | Figures | ~1 min |

### Appendix A (pre-declared attempts to improve detection)

| Step | Command | Produces | Manuscript | Runtime (1 core) |
|---|---|---|---|---|
| A1 | `python pipeline_v3.py dev` then `python pipeline_v3.py test` | `results/appendix_attempt1/` | Table A1; Table A3 rows | ~40 + 15 min |
| A2 | `python stage1.py dev` then `python stage1.py test` (needs step A1's checkpoints) | `results/appendix_attempt2/` | Table A2; Table A3 rows | ~70 + 10 min |
| A3 | `python robustness_summary.py` | `appendix_summary.csv` | Table A3 | ~3 min |

Both attempts write their protocol, selection rule and candidate set in the script header; the selection is written to `selection.json` before the test run. Both checkpoint after every configuration and year and resume if interrupted. Attempt 2 also needs `pip install lightgbm` (4.7.0 used).

Run the cost timings (step 5) on an otherwise idle machine; they are the only
non-deterministic results.

## Files

- `fusion.py` — logistic fusion on standardised branch margins, with a
  fixed-order (batch-independent) sum.
- `commitments.py` — model, input and record commitments.
- `anchoring_ledger.py` — software model of Fabric endorsement, write-once
  chaincode and MVCC validation. Not a Fabric deployment: no consensus, network,
  signatures or timing.
- `results/E6_binding_before_fixed_order_sum.csv` — bound verification with the
  earlier matrix–vector fusion sum (10 of 56 honest artifacts rejected), kept as
  the evidence for Section 6.5.5.
