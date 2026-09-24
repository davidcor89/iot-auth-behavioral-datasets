# Dataset, field roles, and authoritative manifests

## Information layers

The artifact keeps the following concepts distinct:

1. **Simulator/source state** — internal device, profile, planning, and security
   objects used to generate a trace.
2. **Generated trace data** — canonical event evidence after six-decimal
   serialization and admission validation.
3. **Semantic ground truth** — normal/anomaly truth and violated invariants.
4. **Hard-negative qualification** — independent confirmation that a generated
   hard negative remains semantically normal and satisfies its variant rule.
5. **Debug/provenance** — transformation inputs, source references, rejection
   evidence, seeds, versions, and reconstruction metadata.
6. **Device Profile information** — privileged latent T/A/R/N/L values and
   device/profile association.
7. **Applicability/control membership** — privileged scheduler information used
   to construct controlled scientific cohorts.
8. **Detector observations** — only the frozen O03 fields available to the
   feature extractor.
9. **Derived ML features** — deterministic complete-trace summaries declared in
   the feature specification.
10. **Experiment targets/predictions** — privileged targets and persisted model
    outputs used for evaluation.

Co-location in a JSON package does not make privileged fields detector-visible.

## O03 detector observations

The synchronized candidate exposes these eleven fields to the detector-facing
projection:

| Field | Role |
|---|---|
| `event_type` | observed implementation event identifier |
| `observed_timestamp` | canonical serialized observation time |
| `observed_delay_since_previous_event` | observed inter-event delay |
| `result` | observed event result |
| `failure_reason` | observed failure evidence where present |
| `retry_count` | observed retry counter |
| `token_id` | observed token reference where present |
| `nonce` | observed nonce material where present |
| `topic` | observed MQTT topic where present |
| `resource_id` | observed protected resource reference |
| `requested_action` | observed requested operation |

The authoritative implementation is the candidate projection plus
`feature_spec_icds_v1.json`; this guide does not redefine it.

## Manifest index

| Question | Authoritative repository source |
|---|---|
| Candidate rows and detector/GT/provenance separation | `artifacts/icds-timing-nonce-candidate/scientific-candidate.json` |
| Candidate seeds, composition, counts, validation, fingerprints | `artifacts/icds-timing-nonce-candidate/build-manifest.json` |
| O03 derived feature definitions and C1/C2 eligibility | `IoT_auth_simulator/ml/icds/feature_spec_icds_v1.json` |
| Phase 1 structural extraction evidence | `artifacts/icds-ml-phase1/structural-extraction-audit.json` |
| Scientific roles, targets, five splits, C0/C1 membership | `artifacts/icds-ml-phase2/icds_experiment_manifest_v1.json` |
| Preprocessing, models, thresholds, metrics, feature profiles | `IoT_auth_simulator/ml/icds/icds_pipeline_spec_v1.json` |
| Device-grouped sensitivity membership | `artifacts/icds-ml-phase3/icds_grouped_cohorts_v1.json` |
| Phase 4 execution plan and counts | `artifacts/icds-ml-phase4/execution-manifest.json` |
| Per-run predictions and metrics | `artifacts/icds-ml-phase4/primary/` and `artifacts/icds-ml-phase4/grouped/` |
| Frozen Phase 4 file/tree hashes | `artifacts/icds-ml-phase4/fingerprints.json` |
| Permitted empirical claims | `artifacts/icds-ml-phase4/claim-audit.json` |
| Paper-export derivation and hashes | `artifacts/icds-paper-figures/paper-artifact-manifest.json` |

## Candidate validation evidence

The build manifest records:

- 336/336 planned samples admitted;
- 96/96 ordinary normals valid;
- 144/144 hard negatives normal and qualified;
- 48/48 Timing anomalies with `INV-TIME-OBSERVED-CAUSAL`;
- 48/48 Nonce anomalies with `INV-REPLAY-NONCE-CROSS-ATTEMPT`;
- zero unexpected multi-invariant samples;
- 336/336 unique canonical O03 evidence records;
- no same-truth or conflicting-truth duplicate groups;
- one scientific role per base and no rejected base exported.

## Scope warning

The feature representation is complete-trace and future-dependent relative to
online event time. It supports this frozen offline task only. Neither Device
Profile parameters nor privileged semantic evidence may be treated as detector
observations.

