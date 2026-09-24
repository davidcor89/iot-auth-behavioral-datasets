# ICDS Phase 4 frozen scientific results

Status: completed frozen experiment. Raw persisted predictions are authoritative; all metrics in this report were reconstructed from those records. No configuration, feature, cohort, threshold, seed, or metric was adapted after execution began.

## Execution

- Primary: 45/45 runs and 3,780/3,780 predictions.
- Device-grouped sensitivity: 9/9 runs and 819/819 predictions.
- Models: LR, DT, and RF used as measurement instruments, not ranked competitors.
- Primary repetitions: five persisted Phase 2 splits.
- Grouped sensitivity: one approved 48/16-device split with zero device overlap.

## Primary experimental outcomes

Values are medians across five repetitions; brackets show Q1–Q3. Full individual results, ranges, confusion matrices, and subgroup counts remain in the machine-readable artifacts.

| Model | Condition | Anomaly F1 | Balanced accuracy | Timing anomaly recall | Nonce anomaly recall |
|---|---|---:|---:|---:|---:|
| LR | C0 | 0.787 [0.787–0.800] | 0.892 [0.892–0.900] | 1.000 | 1.000 |
| LR | C1 | 0.814 [0.800–0.821] | 0.904 [0.883–0.908] | 0.917 | 1.000 |
| LR | C2 | 0.750 [0.690–0.830] | 0.846 [0.800–0.900] | 0.750 | 1.000 |
| DT | C0 | 0.800 [0.787–0.800] | 0.900 [0.892–0.900] | 1.000 | 1.000 |
| DT | C1 | 1.000 [0.941–1.000] | 1.000 [0.975–1.000] | 1.000 | 1.000 |
| DT | C2 | 1.000 [0.889–1.000] | 1.000 [0.950–1.000] | 1.000 | 1.000 |
| RF | C0 | 0.842 [0.842–0.857] | 0.925 [0.925–0.933] | 1.000 | 1.000 |
| RF | C1 | 0.980 [0.980–1.000] | 0.992 [0.992–1.000] | 1.000 | 1.000 |
| RF | C2 | 0.880 [0.875–0.889] | 0.913 [0.908–0.925] | 0.750 | 1.000 |

These high values are controlled synthetic-candidate outcomes, not evidence of operational IDS effectiveness.

## C0 to C1: hard-negative effect

Median normal-role false-positive rates and anomaly-family recall:

| Model | Condition | Ordinary FPR | Timing-HN FPR | Same-attempt nonce-HN FPR | Fresh nonce-HN FPR | Timing recall | Nonce recall |
|---|---|---:|---:|---:|---:|---:|---:|
| LR | C0 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| LR | C1 | 0.000 | 0.750 | 0.000 | 0.000 | 0.917 | 1.000 |
| DT | C0 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| DT | C1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| RF | C0 | 0.000 | 0.750 | 0.000 | 0.000 | 1.000 | 1.000 |
| RF | C1 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

The central effect is boundary-specific. C0 nearly always classified small-positive Timing HNs as anomalous while handling the two nonce-HN roles and ordinary normals well. C1 changed this behavior for all three baselines:

- LR reduced median Timing-HN FPR by 0.25, but the role remained difficult (median 0.75), and Timing-anomaly recall declined by a median 0.083. Aggregate paired changes were small and varied across repetitions: median F1 +0.021 and balanced accuracy +0.004.
- DT reduced median Timing-HN FPR from 1.0 to 0.0. Median F1 increased by 0.200 and balanced accuracy by 0.100; median anomaly-family recall remained 1.0. One repetition had slightly reduced nonce-anomaly recall.
- RF reduced median Timing-HN FPR from 0.75 to 0.0. Median F1 increased by 0.127 and balanced accuracy by 0.058; both anomaly-family recall medians remained 1.0.

Thus semantic HNs changed the measured error distribution, but the size and associated anomaly trade-off depended on model family. The evidence does not support reducing the result to one aggregate score.

## C1 to C2: representation sensitivity

C2 retains exactly C1's samples and removes only three reconstructed predicate-equivalent relation features.

- LR: median F1 fell from 0.814 to 0.750 and balanced accuracy from 0.904 to 0.846. Timing-anomaly recall fell from 0.917 to 0.750. Timing-HN FPR also fell, but false positives increased for ordinary, same-attempt-nonce, and fresh-nonce normal roles. The change is a redistribution of errors, not a uniform degradation.
- DT: median F1, balanced accuracy, and both anomaly-family recall values remained 1.0. Some repetitions acquired ordinary/fresh-nonce false positives or a nonce miss, but the median response was robust.
- RF: median F1 fell from 0.980 to 0.880 and balanced accuracy from 0.992 to 0.913. Timing-anomaly recall fell from 1.0 to 0.75, while nonce-anomaly recall remained 1.0.

For LR and RF, C1 separability depends materially on the explicit semantic-relation representation, especially for Timing evidence. DT retained similar median outcomes, demonstrating that the dependence is not universal: broader O03 structure was sufficient for that baseline on most repetitions.

No post-hoc feature search was performed.

## Cross-model consistency

All three baselines agree that the ordinary-only C0 composition exposes a major Timing-HN false-positive problem and that C1 changes it. They disagree on the extent:

- DT and RF nearly eliminate the Timing-HN error and improve aggregate outcomes without median Timing-recall loss.
- LR shows only partial Timing-HN improvement and a Timing-anomaly recall trade-off.

For C1 to C2, LR and RF agree on lower aggregate outcomes and Timing recall; DT is substantially less sensitive in its median results. This is a model-family interaction, not grounds for declaring a best model.

## Device-grouped sensitivity

The grouped test has 25 ordinary normals, 16 Timing HNs, 11 same-attempt nonce HNs, 16 fresh nonce HNs, 12 Timing anomalies, and 11 Nonce anomalies. Raw counts accompany rates because role sizes are small.

| Model | Condition | F1 | Balanced acc. | Ordinary FP | Timing-HN FP | Same-nonce FP | Fresh-nonce FP | Timing TP | Nonce TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LR | C0 | 0.742 | 0.882 | 0/25 | 16/16 | 0/11 | 0/16 | 12/12 | 11/11 |
| LR | C1 | 0.679 | 0.803 | 0/25 | 11/16 | 1/11 | 0/16 | 7/12 | 11/11 |
| LR | C2 | 0.679 | 0.803 | 1/25 | 7/16 | 1/11 | 3/16 | 7/12 | 11/11 |
| DT | C0 | 0.742 | 0.882 | 0/25 | 16/16 | 0/11 | 0/16 | 12/12 | 11/11 |
| DT | C1 | 1.000 | 1.000 | 0/25 | 0/16 | 0/11 | 0/16 | 12/12 | 11/11 |
| DT | C2 | 1.000 | 1.000 | 0/25 | 0/16 | 0/11 | 0/16 | 12/12 | 11/11 |
| RF | C0 | 0.767 | 0.897 | 0/25 | 14/16 | 0/11 | 0/16 | 12/12 | 11/11 |
| RF | C1 | 1.000 | 1.000 | 0/25 | 0/16 | 0/11 | 0/16 | 12/12 | 11/11 |
| RF | C2 | 0.818 | 0.869 | 0/25 | 1/16 | 0/11 | 2/16 | 7/12 | 11/11 |

The DT/RF primary pattern persists descriptively under device disjointness. LR again shows only partial Timing-HN improvement and a stronger Timing-recall cost; its grouped aggregate outcomes decline from C0 to C1. C2 leaves grouped DT unchanged, redistributes LR false positives without changing its aggregate result, and reduces grouped RF Timing recall.

This is one controlled synthetic device partition with pragmatic support floors. It does not establish statistically powered unseen-device or real-population generalization.

## Claim audit

### Directly supported

- Normal training composition changes boundary false-positive behavior under the frozen protocol.
- The strongest C0 weakness is the small-positive Timing-HN boundary, not a general inability to classify all normal traces.
- DT/RF C1 training strongly reduces that boundary error while preserving median anomaly recall; LR exhibits a partial reduction and Timing-recall trade-off.
- C1-to-C2 sensitivity is material for LR/RF Timing behavior and smaller in DT median results.
- The grouped split preserves the DT/RF qualitative C0-to-C1 result but exposes model-specific LR limitations.

### Supported with qualification

Any claim about semantic hard-negative utility or semantic-relation representation must be restricted to:

- this frozen synthetic Timing/Nonce candidate;
- the O03 observation contract;
- complete-trace offline classification;
- the fixed LR/DT/RF baselines;
- the frozen primary splits or single grouped sensitivity partition.

### Not supported

The experiment does not support claims of a new/superior detector, a best model, general behavioral or unseen-anomaly detection, universal IoT authentication detection, operational IDS effectiveness, real-world representativeness, synthetic-to-operational transfer, universal observability, distributed-monitor sufficiency, sequence-model necessity, generalization outside Timing/Nonce, or universal device-independent generalization.

## Artifact authority

Raw prediction JSONL files are authoritative for metrics. Per-run metrics are reconstructable from them. The aggregate summary preserves all five primary values and reports medians, linear Q1/Q3/IQR, and min/max without inferential confidence intervals.
