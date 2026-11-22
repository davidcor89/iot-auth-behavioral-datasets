# R4.1 repeated device-disjoint evaluation

## A. Execution-integrity report

The campaign is complete and independently valid: **180/180 runs**, **84 predictions per run**, **15,120 predictions**, and **60 paired C0→C1 comparisons**. All 20 frozen partitions remained device-disjoint and retained exact test/cohort memberships. Historical Phase 4 verification remained `sha256:cc100d1e234d8c0605639def25351d4c756e99a9b7b99acf3639c03065a97c11`.

## B. Complete descriptive results

Values are median [Q1–Q3; min–max] across the 20 predefined grouped partitions.

| Model | Condition | Timing-HN FPR | Timing-anomaly recall |
|---|---|---:|---:|
| LR | C0 | 1.000 [1.000–1.000; 0.833–1.000] | 1.000 [1.000–1.000; 1.000–1.000] |
| LR | C1 | 0.750 [0.667–0.833; 0.583–0.917] | 0.917 [0.833–1.000; 0.750–1.000] |
| LR | C2 | 0.500 [0.417–0.583; 0.167–0.667] | 0.833 [0.833–0.917; 0.583–1.000] |
| DT | C0 | 1.000 [1.000–1.000; 1.000–1.000] | 1.000 [1.000–1.000; 1.000–1.000] |
| DT | C1 | 0.000 [0.000–0.000; 0.000–0.250] | 1.000 [1.000–1.000; 0.917–1.000] |
| DT | C2 | 0.000 [0.000–0.000; 0.000–0.167] | 1.000 [1.000–1.000; 0.833–1.000] |
| RF | C0 | 0.708 [0.583–0.833; 0.333–1.000] | 1.000 [1.000–1.000; 0.917–1.000] |
| RF | C1 | 0.000 [0.000–0.000; 0.000–0.250] | 1.000 [1.000–1.000; 0.917–1.000] |
| RF | C2 | 0.000 [0.000–0.104; 0.000–0.333] | 0.917 [0.833–0.938; 0.750–1.000] |

Secondary values use the same median [Q1–Q3; min–max] format:

| Model | Condition | Ordinary FPR | Same-attempt Nonce-HN FPR | Fresh cross-attempt Nonce-HN FPR | Nonce-anomaly recall |
|---|---|---:|---:|---:|---:|
| LR | C0 | 0.000 [0.000–0.000; 0.000–0.042] | 0.000 [0.000–0.000; 0.000–0.083] | 0.000 [0.000–0.000; 0.000–0.000] | 1.000 [1.000–1.000; 1.000–1.000] |
| LR | C1 | 0.000 [0.000–0.000; 0.000–0.000] | 0.000 [0.000–0.083; 0.000–0.083] | 0.000 [0.000–0.000; 0.000–0.000] | 1.000 [1.000–1.000; 1.000–1.000] |
| LR | C2 | 0.042 [0.000–0.083; 0.000–0.125] | 0.083 [0.083–0.167; 0.000–0.333] | 0.167 [0.083–0.250; 0.000–0.333] | 1.000 [1.000–1.000; 1.000–1.000] |
| DT | C0 | 0.000 [0.000–0.000; 0.000–0.083] | 0.000 [0.000–0.083; 0.000–0.167] | 0.000 [0.000–0.000; 0.000–0.000] | 1.000 [1.000–1.000; 1.000–1.000] |
| DT | C1 | 0.000 [0.000–0.000; 0.000–0.125] | 0.000 [0.000–0.000; 0.000–0.167] | 0.000 [0.000–0.000; 0.000–0.167] | 1.000 [1.000–1.000; 1.000–1.000] |
| DT | C2 | 0.000 [0.000–0.000; 0.000–0.125] | 0.000 [0.000–0.000; 0.000–0.167] | 0.000 [0.000–0.000; 0.000–0.333] | 1.000 [1.000–1.000; 0.917–1.000] |
| RF | C0 | 0.000 [0.000–0.000; 0.000–0.000] | 0.000 [0.000–0.000; 0.000–0.083] | 0.000 [0.000–0.000; 0.000–0.083] | 1.000 [1.000–1.000; 1.000–1.000] |
| RF | C1 | 0.000 [0.000–0.000; 0.000–0.167] | 0.000 [0.000–0.000; 0.000–0.083] | 0.000 [0.000–0.000; 0.000–0.167] | 1.000 [1.000–1.000; 1.000–1.000] |
| RF | C2 | 0.042 [0.000–0.052; 0.000–0.292] | 0.042 [0.000–0.083; 0.000–0.250] | 0.083 [0.000–0.104; 0.000–0.333] | 1.000 [1.000–1.000; 1.000–1.000] |

The near-zero C0/C1 Nonce-HN FPRs and consistently high Nonce-anomaly recall show that the main paired change is concentrated in the Timing boundary rather than representing a uniform shift across every semantic role. C2 remains descriptive and is not interpreted as improved observability.

## C. Paired C0→C1 analysis

| Model | Δ Timing-HN FPR | Δ Timing recall | ΔFPR −/0/+ | clean/trade-off/no improvement/deterioration |
|---|---:|---:|---:|---:|
| LR | -0.167 [-0.333–-0.146; -0.417–-0.083] | -0.083 [-0.167–0.000; -0.250–0.000] | 20/0/0 | 6/14/0/0 |
| DT | -1.000 [-1.000–-1.000; -1.000–-0.750] | 0.000 [0.000–0.000; -0.083–0.000] | 20/0/0 | 18/2/0/0 |
| RF | -0.667 [-0.833–-0.562; -1.000–-0.333] | 0.000 [0.000–0.000; -0.083–0.000] | 20/0/0 | 19/1/0/0 |

These are paired differences within the same frozen partition, not differences between independently calculated medians. No split-level hypothesis test was performed.

## D. Model-specific consistency assessment

- **LR: consistent boundary correction with recurring recall trade-off.** Historical grouped sensitivity had ΔFPR=-0.312, ΔRecall=-0.417; the repeated-partition sign counts are 20/0/0 (lower/equal/higher Timing-HN FPR under C1), with 14 recall-cost partition(s).
- **DT: consistent, predominantly clean boundary correction.** Historical grouped sensitivity had ΔFPR=-1.000, ΔRecall=0.000; the repeated-partition sign counts are 20/0/0 (lower/equal/higher Timing-HN FPR under C1), with 2 recall-cost partition(s).
- **RF: consistent, predominantly clean boundary correction.** Historical grouped sensitivity had ΔFPR=-0.875, ΔRecall=0.000; the repeated-partition sign counts are 20/0/0 (lower/equal/higher Timing-HN FPR under C1), with 1 recall-cost partition(s).

## E. Unexpected implementation or data issues

- The initially named execution repository was corrected before this campaign; no model was run in `PFE_simulator`.
- The same-attempt Nonce HN retains its frozen `semantic_family=direct_timing` provenance but is reported by authoritative scientific role, not misclassified as Timing HN.
- The compatible external virtual environment was used only as an interpreter/dependency environment; all code and inputs came from this authorized repository.
- scikit-learn 1.9 emitted a nonfatal deprecation warning for the frozen explicit `penalty="l2"` Logistic Regression parameter. The parameter was retained unchanged; all LR runs completed.
- The focused ML regression suite passed 24/24. `unittest discover` passed 468 tests with one skip. Raw `pytest -q` additionally exposes a pre-existing collection defect in `tests/test_dp2_profile_provenance.py`, where a helper named `test_population` is mistaken for a fixture-based test; no unrelated test was modified.
- No fallback split, cohort resampling, tuning, calibration, or imputation occurred.

## F. Scientific interpretation

The results address only the frozen 336-trace synthetic Timing+Nonce candidate. The Timing-boundary effect persists under all 20 predefined device-disjoint partitions for LR, DT, and RF: every C0→C1 paired Timing-HN FPR difference is negative. For LR, the correction is accompanied by a recurring Timing-anomaly recall cost in 14/20 partitions. For DT and RF, recall is preserved in 18/20 and 19/20 partitions respectively, so their correction is predominantly clean with limited partition-specific recall loss. Clean corrections and trade-offs are reported separately rather than averaged away.

This agrees directionally with the historical single device-disjoint sensitivity result for all three classifiers. The historical LR result also showed a recall trade-off; repeated partitions show that this cost recurs, although its magnitude varies. Historical DT and RF corrections preserved recall, consistent with the predominantly clean repeated-partition outcomes.

Device-disjointness prevents persistent-device overlap between train and test, but it does not establish independence from shared generator logic, scenario templates, population strata, or campaign seeds. These results therefore do not establish operational IoT performance, independent-device population generalization, generator-independent transfer, statistical significance, a new IDS contribution, or improved observability from C2.

## G. Reproducibility

Execution command:

```bash
PYTHONPATH="$PWD/IoT_auth_simulator" ../PFE_simulator/.venv/bin/python -m ml.evaluation.r4_1_execution \
  --partition-manifest /tmp/r4.1-exact-partition-audit/partition-manifest.json \
  --accepted-partitions /tmp/r4.1-exact-partition-audit/accepted-partitions.csv \
  --output artifacts/r4.1-device-disjoint-evaluation
```

- Repository checkpoint: `97003b1d9448caa105b6e206ceb5deb18c02a207`
- Partition manifest SHA-256: `80c8621860237999b95c2d4e7a0f12a7b56f42ef9c3df3720af5a393bc0131fe`
- Accepted partitions SHA-256: `4b7fd17277a6765bba0e509ee48a39f3c7f59492a4e69d3d238ba2cb91671435`
- Historical Phase 4 fingerprint: `cc100d1e234d8c0605639def25351d4c756e99a9b7b99acf3639c03065a97c11`
- Independent validation: rerun the same command with `--validate-only`.
