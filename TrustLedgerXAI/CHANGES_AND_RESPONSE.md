# Revision notes: v9 → v10

Mapping from each review comment to the change made.

**Major 1 — residual IEEE-CIS/ERP text.** Removed from 3.1, 7.7, Abstract, Introduction and Figure 1. XGBoost reference removed; the model is scikit-learn HistGradientBoosting throughout. The 434-feature and 118,108-artifact values are replaced by the real artifact size (99 attributions) and the 68,230 firm-years of the test window. The trust-transfer premise is stated as an assumption, and record digests are committed instead of assuming on-chain source records.

**Major 2 — explanation did not explain the decision.** The fusion now runs on standardised branch margins, so the fused log-odds decomposes exactly into TreeSHAP attributions of both supervised branches (Eq. 4; max additivity error 4.6e-14). All 682 flagged firm-years have decision-level artifacts. Counterfactual and binding experiments use the same saved final-year fused model.

**Major 3 — manuscript/package mismatch.** All tables are generated from `results/`. Counts, n = 68,230, 682 artifacts, and cross-implementation results are preserved in the package.

**Major 4 — detection framing and statistics.** Adds a firm-clustered bootstrap, Holm adjustment, RUSBoost, PR-AUC, within-year budget counts, 2003–2008 means and a label-censoring discussion. The detector is now described as competitive, not superior.

**Major 5 — undisclosed temporal features and fusion leakage.** The 71 temporal features are disclosed. Folds are grouped by firm, and weights are reported for every year (anomaly 0.16–0.26). The earlier −0.05 anomaly weight does not recur under grouped folds and standardised margins, so the "two-branch" limitation is withdrawn.

**Major 6 — tautological adversarial tests.** Recast as conformance tests (Table 11). Adds an MVCC race, a controlled-endorser case and subtle fabrications (swap, 1 ULP). Cross-version and single-record honest recomputation are tested (Table 16).

**Major 7 — Fabric claims.** "Freshness" is replaced by "ordering", with the client-set timestamp caveat (CVE-2024-45244). The block reference is removed from the stored value. World-state bytes are measured, and envelope overhead is declared unmeasured. Merkle batching now uses an RFC 6962-style tree, and the two weaknesses of the old construction are demonstrated. The Abstract, Tables 4/6 and the Conclusion no longer claim a Fabric deployment.

**Major 8 — model commitment.** The commitment now includes missing-value routing, baseline prediction, the isolation forest, the scaler and fusion coefficients. It is registered on the ledger before the assessments. Table 15 shows the old commitment missed a change that altered 31 of 56 flags.

**Major 9 — counterfactuals.** The L1 misstatement is removed. The search is now greedy and sparse, with MAD scaling, empirical-quantile candidates and history held fixed. It is reported with and without market variables: 55/56 vs 40/56 attainable, median one item changed.

**Major 10 — novelty.** Adds tamper-evident logging (Schneier & Kelsey; Crosby & Wallach), RFC 6962, Nassar et al. (2020) and Wang (2026). Contributions are repositioned accordingly.

**Major 11 — overgeneralisation.** The "one in twenty" claim is replaced by measured real-artifact results (0/682 agreement for sorted-key JSON, with the mechanism identified). The stress set is labelled as a stress test.

**Minor.** Section cross-references are corrected, and references renumbered by first appearance. The metrics section now matches the reported metrics, with NDCG defined. The SHAP estimator is described correctly (path-dependent TreeSHAP), and the broken sentence in 4.3 is fixed. Table 7 is rebuilt, with the size-proxy caveat. The ethics statement is simplified.

**Open items for the authors.**
1. Insert the repository URL in Declarations.
2. Reference [39] of v9 (Al-Hchaimi et al., 2026) could not be verified and was removed. Reinstate it if you hold its DOI.
3. If you rerun on Colab, update Table 3 and the timings in Tables 12–13.

# v10 → v11

- **Appendix A added.** It reports two pre-declared attempts to improve detection, each selected on 1998–2002 and run once on 2003–2014:
  - tuning of the branch learners and features, with equal tuning for RUSBoost;
  - a ranking-oriented redesign whose primary metric was NDCG@1% over 2003–2008.
- **Result.** Neither attempt differed from RUSBoost or the ratio model beyond chance, and the manuscript configuration is retained. All candidate configurations are reported (Tables A1–A3).
- **Section 6.1 now notes the 2022 Bao et al. erratum.** Published NDCG figures from before the correction are not comparable with the corrected protocol used here. New references: Bao et al. (2022) erratum; Walker (2021).
- **Abstract, Section 7.7 and Conclusion** each have one added sentence reporting the attempts.
