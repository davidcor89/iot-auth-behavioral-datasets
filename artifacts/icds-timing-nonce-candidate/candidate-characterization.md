# Controlled Timing+Nonce ICDS scientific candidate

This is a controlled scientific candidate pending researcher inspection, not an ML dataset.

## Build result

- Samples: 336 (normal 240; anomaly 96)
- Qualified HNs: 144
- Base opportunities attempted/admitted/rejected: 512/510/2
- Candidate fingerprint: `sha256:be34fcbce532d4791cf3f27f056146215fdd255d1e28415e00e3e77f248d3f5c`
- Build fingerprint: `sha256:bfcfd48e9e6b213325e12c3dd3cf25efad0677753eb9623e406859625b13e3ef`

## Scientific roles

| Role | Count | Devices | Maximum/device | O03 equal/different from base |
|---|---:|---:|---:|---:|
| `ordinary_normal` | 96 | 64 | 2 | 96/0 |
| `hard_negative:small_positive_causal_margin` | 48 | 43 | 2 | 0/48 |
| `anomaly:timestamp_inconsistency` | 48 | 44 | 2 | 0/48 |
| `hard_negative:legitimate_same_attempt_nonce_recurrence` | 48 | 46 | 2 | 48/0 |
| `hard_negative:fresh_nonce_cross_attempt` | 48 | 43 | 2 | 48/0 |
| `anomaly:nonce_reuse` | 48 | 43 | 2 | 0/48 |

## Integrity

- Unique O03 evidence: 336/336
- Same-truth duplicate groups: 0
- Conflicting-truth duplicate groups: 0
- Unexpected multi-invariant samples: 0
- All canonical reload and aggregate validation gates passed.

## C0/C1 projections

- C0: 192 samples
- C1: 336 samples
- Caveat: C0 to C1 changes both HN presence and sample count/class prevalence; 04 owns weighting, matching, evaluation sets, and prevalence control.

## Limitations

- not empirically calibrated or operationally representative
- not comprehensive IoT or behavioral anomaly coverage
- no unseen-anomaly or deployment-performance claim
- no universal-observability or optimal-monitoring claim
- token replay and access/identity/binding roles are excluded
- no final ML features, horizon, split regime, or detector claim
