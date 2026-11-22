# Paper artifact map

The reusable venue-neutral entry point is
`publication-artifacts/eval-v1.0/manifest.json`. Original submission filenames
below remain unchanged because they are frozen historical evidence.

Final manuscript numbering has not been supplied. This map therefore uses
descriptive names rather than inventing paper figure or table numbers.

| Paper element | Repository source |
|---|---|
| Candidate composition and role counts | `artifacts/icds-timing-nonce-candidate/build-manifest.json` |
| Canonical candidate traces | `artifacts/icds-timing-nonce-candidate/scientific-candidate.json` |
| Device Profile population configuration/provenance | candidate build manifest and `IoT_auth_simulator/simulator/device_profiles/` |
| Semantic invariants | `IoT_auth_simulator/simulator/invariant_registry.py` |
| Hard-negative definitions and qualification | `IoT_auth_simulator/simulator/hard_negative.py` and `hard_negative_qualification.py` |
| O03 observations and fields | candidate JSON, dataset contract, and synchronized views |
| O03 derived features | `IoT_auth_simulator/ml/icds/feature_spec_icds_v1.json` |
| C0/C1/C2 definitions and splits | `artifacts/icds-ml-phase2/icds_experiment_manifest_v1.json` |
| Device-grouped sensitivity split | `artifacts/icds-ml-phase3/icds_grouped_cohorts_v1.json` |
| Preprocessing and baseline models | `IoT_auth_simulator/ml/icds/icds_pipeline_spec_v1.json` |
| Frozen predictions | `artifacts/icds-ml-phase4/*/predictions.jsonl` |
| Run provenance and model seeds | `artifacts/icds-ml-phase4/*/run-provenance.json` |
| Per-run and aggregate metrics | `artifacts/icds-ml-phase4/*/per-run-metrics.json` and `primary/aggregate-summary.json` |
| Frozen result interpretation | `docs/icds-phase4-scientific-results.md` and `claim-audit.json` |
| Semantic hard-negative figure | `artifacts/icds-paper-figures/figure-1-primary-hard-negative-effect.*` |
| Semantic error-profile matrix | `artifacts/icds-paper-figures/figure-2-semantic-error-profile.*` |
| Representation sensitivity figure | `artifacts/icds-paper-figures/figure-3-representation-sensitivity.*` |
| Device-disjoint sensitivity figure | `artifacts/icds-paper-figures/figure-4-device-grouped-sensitivity.*` |
| Paper tables A–D | `artifacts/icds-paper-figures/table-*.csv` and `table-*.tex` |
| Figure/table generation | `notebooks/icds_phase4_paper_artifacts.ipynb` and `paper_artifacts.py` |
| Export hashes and source evidence | `artifacts/icds-paper-figures/paper-artifact-manifest.json` |

The paper layer is a deterministic reporting transformation. Persisted Phase 4
predictions remain authoritative for all reported metrics.
