# R4.2 representation-sensitivity campaign

The frozen campaign completed and passed independent validation: **120 new C/D runs**, **10,080 new predictions**, and **120 immutable A/B anchors referenced without retraining**. Results are descriptive across 20 predefined device-disjoint partitions; no split-level hypothesis tests were performed.

## Timing-boundary results

Values are median [Q1–Q3; min–max] across the 20 partitions.

| Model | Representation | Timing-HN FPR | Timing-anomaly recall |
|---|---|---|---|
| LR | A | 0.750 [0.667–0.833; 0.583–0.917] | 0.917 [0.833–1.000; 0.750–1.000] |
| LR | B | 0.500 [0.417–0.583; 0.167–0.667] | 0.833 [0.833–0.917; 0.583–1.000] |
| LR | C | 0.500 [0.417–0.583; 0.167–0.667] | 0.917 [0.833–0.917; 0.583–1.000] |
| LR | D | 0.000 [0.000–0.083; 0.000–0.167] | 0.042 [0.000–0.167; 0.000–0.417] |
| DT | A | 0.000 [0.000–0.000; 0.000–0.250] | 1.000 [1.000–1.000; 0.917–1.000] |
| DT | B | 0.000 [0.000–0.000; 0.000–0.167] | 1.000 [1.000–1.000; 0.833–1.000] |
| DT | C | 0.000 [0.000–0.000; 0.000–0.167] | 1.000 [1.000–1.000; 0.833–1.000] |
| DT | D | 0.167 [0.083–0.271; 0.000–0.500] | 0.167 [0.083–0.271; 0.000–0.583] |
| RF | A | 0.000 [0.000–0.000; 0.000–0.250] | 1.000 [1.000–1.000; 0.917–1.000] |
| RF | B | 0.000 [0.000–0.104; 0.000–0.333] | 0.917 [0.833–0.938; 0.750–1.000] |
| RF | C | 0.000 [0.000–0.083; 0.000–0.250] | 0.875 [0.812–0.917; 0.667–1.000] |
| RF | D | 0.167 [0.000–0.167; 0.000–0.417] | 0.083 [0.083–0.250; 0.000–0.500] |

## Paired representation changes

Deltas are target minus source within the same frozen partition and model.

| Model | Contrast | ΔFPR | ΔRecall | FPR directions (−/0/+) | Recall directions (−/0/+) | Joint outcomes |
|---|---|---|---|---|---|---|
| LR | A→C | -0.250 [-0.354–-0.167; -0.667–-0.083] | 0.000 [-0.083–0.000; -0.250–0.000] | 20/0/0 | 8/12/0 | {"fpr_improved_recall_degraded": 8, "fpr_improved_recall_preserved_or_improved": 12} |
| LR | C→D | -0.417 [-0.521–-0.333; -0.667–-0.083] | -0.792 [-0.917–-0.667; -1.000–-0.500] | 20/0/0 | 20/0/0 | {"fpr_improved_recall_degraded": 20} |
| LR | A→B | -0.250 [-0.354–-0.167; -0.667–-0.083] | 0.000 [-0.083–0.000; -0.250–0.000] | 20/0/0 | 9/11/0 | {"fpr_improved_recall_degraded": 9, "fpr_improved_recall_preserved_or_improved": 11} |
| LR | C→B | 0.000 [0.000–0.000; 0.000–0.083] | 0.000 [0.000–0.000; -0.083–0.083] | 0/19/1 | 2/17/1 | {"fpr_unchanged_recall_degraded": 2, "fpr_unchanged_recall_improved": 1, "fpr_worsened_recall_preserved_or_improved": 1, "no_change": 16} |
| DT | A→C | 0.000 [0.000–0.000; -0.083–0.000] | 0.000 [0.000–0.000; -0.083–0.000] | 1/19/0 | 1/19/0 | {"fpr_improved_recall_degraded": 1, "no_change": 19} |
| DT | C→D | 0.167 [0.083–0.250; -0.167–0.500] | -0.792 [-0.917–-0.729; -1.000–-0.417] | 1/2/17 | 20/0/0 | {"both_fpr_and_recall_worsened": 17, "fpr_improved_recall_degraded": 1, "fpr_unchanged_recall_degraded": 2} |
| DT | A→B | 0.000 [0.000–0.000; -0.083–0.000] | 0.000 [0.000–0.000; -0.083–0.000] | 1/19/0 | 1/19/0 | {"fpr_improved_recall_degraded": 1, "no_change": 19} |
| DT | C→B | 0.000 [0.000–0.000; 0.000–0.000] | 0.000 [0.000–0.000; 0.000–0.000] | 0/20/0 | 0/20/0 | {"no_change": 20} |
| RF | A→C | 0.000 [0.000–0.083; 0.000–0.167] | -0.083 [-0.187–-0.062; -0.333–0.000] | 0/14/6 | 15/5/0 | {"both_fpr_and_recall_worsened": 5, "fpr_unchanged_recall_degraded": 10, "fpr_worsened_recall_preserved_or_improved": 1, "no_change": 4} |
| RF | C→D | 0.083 [0.000–0.104; 0.000–0.333] | -0.750 [-0.771–-0.646; -1.000–-0.417] | 0/8/12 | 20/0/0 | {"both_fpr_and_recall_worsened": 12, "fpr_unchanged_recall_degraded": 8} |
| RF | A→B | 0.000 [0.000–0.083; 0.000–0.250] | -0.083 [-0.167–0.000; -0.250–0.000] | 0/13/7 | 14/6/0 | {"both_fpr_and_recall_worsened": 6, "fpr_unchanged_recall_degraded": 8, "fpr_worsened_recall_preserved_or_improved": 1, "no_change": 5} |
| RF | C→B | 0.000 [0.000–0.000; -0.083–0.250] | 0.000 [0.000–0.083; -0.083–0.167] | 1/16/3 | 2/12/6 | {"both_fpr_and_recall_worsened": 1, "fpr_improved_recall_preserved_or_improved": 1, "fpr_unchanged_recall_degraded": 1, "fpr_unchanged_recall_improved": 4, "fpr_worsened_recall_preserved_or_improved": 2, "no_change": 11} |

## Interpretation

A→C isolates the empirical sensitivity to removing the explicit signed Timing-margin summary while retaining generic temporal evidence and `semantic_timing_relation_count`. C→D measures the additional empirical change after removing the seven frozen generic-temporal summaries. A→B is historical continuity rather than a Timing-only ablation because B also removes two direct Nonce summaries. C→B is the additional joint removal of those two Nonce summaries.

### Primary A→C result

- **LR was sensitive in Timing-HN acceptance:** FPR decreased in all 20 partitions (median ΔFPR −0.250). Timing-anomaly recall had median change zero, but decreased in 8/20 partitions. Thus the explicit margin was associated with a consistent LR boundary-FPR change and a partition-dependent recall trade-off.
- **DT was almost invariant:** 19/20 partitions were unchanged on both primary metrics; the remaining partition improved FPR while losing one Timing-anomaly true positive. The median changes were zero.
- **RF changed adversely overall:** FPR was unchanged in 14/20 partitions and increased in 6/20, while Timing-anomaly recall decreased in 15/20 (median ΔRecall −0.083). Removing F1 did not reproduce the LR acceptance improvement for RF.

### Secondary C→D result

Removing the seven generic-temporal summaries after F1 removal produced a large Timing-anomaly recall loss for every model in every partition: median ΔRecall was −0.792 for LR, −0.792 for DT, and −0.750 for RF. The Timing-HN FPR response was model-specific. LR FPR decreased in all 20 partitions (median −0.417), but this accompanied the universal recall collapse. DT FPR increased in 17/20 partitions (median +0.167), and RF FPR increased in 12/20 and was unchanged in 8/20 (median +0.083). Within this candidate, the generic-temporal evidence therefore materially affected classification behavior after F1 removal, but not in a uniformly beneficial direction.

### Supporting contrasts

A→B broadly mirrored A→C for LR and DT, while RF showed additional variation. C→B was exactly unchanged for DT across all 20 partitions and mostly unchanged for LR; RF showed mixed small changes. Because B jointly removes the two direct Nonce summaries as well as F1, these contrasts provide historical continuity and decomposition context rather than a Timing-only causal estimate.

The model-specific direction and magnitude are reported above rather than collapsed into a single claim. Any stability means only that the frozen classifier/partition combination retained similar behavior under the specified representation; any change means sensitivity to that controlled representation change. Neither outcome establishes unique feature-family necessity, sufficiency, formal observability, operational detectability, or causal attribution.

These conclusions are restricted to the frozen 336-trace synthetic Timing+Nonce candidate. Device-disjoint evaluation prevents persistent-device overlap within each split but does not establish independence from shared generator logic, scenario templates, population construction, or repeated test-trace reuse across partitions.

## Reproducibility

- Repository checkpoint: `bd00f098ec86902e5c01e5da1cf1eb4c6e92d77b`.
- Pre-execution fingerprint: `sha256:29b2792d1555c51f1d0d5a9aac5ae549ac23c1d6e817023b8acc78e31aab5da1`.
- Frozen R4.1 artifact tree: `sha256:ed387b03bcacf12369dbaeb0016ba027db301ffdb7f01ecba7609507906ec284`.
- Executed implementation: `execution-implementation/r4_2_execution.executed.py` (hash recorded in run provenance).
- Command: `PYTHONPATH=$PWD/IoT_auth_simulator:$PWD ../PFE_simulator/.venv/bin/python artifacts/r4.2-representation-sensitivity/execution-implementation/r4_2_execution.executed.py --root . --output artifacts/r4.2-representation-sensitivity`.
- The artifact fingerprint manifest records every output byte except itself.
