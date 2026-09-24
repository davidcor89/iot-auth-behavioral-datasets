# ICDS 2026 IoT Authentication Dataset Artifact

This repository contains the frozen research artifact for a controlled study of
semantic boundaries in synthetic IoT-authentication anomaly detection. It
provides the lifecycle-aware simulator, invariant validation, persistent Device
Profiles, hard-negative construction, the frozen Timing+Nonce candidate,
leakage-safe baseline evaluation, and paper-oriented figures and tables.

The artifact is a controlled synthetic methodology study. It is not a
general-purpose or production intrusion-detection system.

## Scientific scope

The generator separates persistent device context, authentication/session
context, semantic ground truth, detector observations, and privileged
provenance. Device Profiles introduce controlled behavioral heterogeneity
without redefining semantic validity. Independently validated hard negatives
exercise legitimate behavior close to invariant boundaries.

The frozen ICDS candidate is limited to two families:

- **Timing:** legitimate small positive causal margins and anomalous timestamp
  inconsistency;
- **Nonce:** legitimate same-attempt recurrence, fresh cross-attempt nonce use,
  and anomalous cross-attempt nonce reuse.

Token, access, identity, and binding transformations exist in the simulator but
are outside the frozen ICDS candidate and reported evaluation.

## Frozen candidate

The candidate contains 336 canonical scientific traces generated from 64
persistent Device Profiles.

| Scientific role | Count | Semantic truth |
|---|---:|---|
| Ordinary normal | 96 | Normal |
| `small_positive_causal_margin` | 48 | Normal / hard negative |
| `timestamp_inconsistency` | 48 | Anomaly |
| `legitimate_same_attempt_nonce_recurrence` | 48 | Normal / hard negative |
| `fresh_nonce_cross_attempt` | 48 | Normal / hard negative |
| `nonce_reuse` | 48 | Anomaly |
| **Total** | **336** | **240 normal / 96 anomaly** |

The authoritative candidate and build manifest are in
[`artifacts/icds-timing-nonce-candidate/`](artifacts/icds-timing-nonce-candidate/).
The manifest records all seeds, admission results, validation evidence, and
fingerprints. Two of 512 attempted base opportunities were canonically rejected
without replacement or resampling; 336/336 scheduled scientific samples were
admitted.

## Observation boundary

The frozen O03 detector-observation contract exposes only:

```text
event_type
observed_timestamp
observed_delay_since_previous_event
result
failure_reason
retry_count
token_id
nonce
topic
resource_id
requested_action
```

Scientific role, ground truth, invariant results, hard-negative qualification,
applicability/control membership, Device Profile parameters, scheduler state,
and transformation provenance remain privileged. The authoritative feature
manifest is
[`IoT_auth_simulator/ml/icds/feature_spec_icds_v1.json`](IoT_auth_simulator/ml/icds/feature_spec_icds_v1.json).

## Frozen evaluation

The complete-trace offline task compares three fixed training conditions:

- **C0:** ordinary-normal-only normal training data;
- **C1:** the same training size, class balance, anomalies, test set, and model
  settings, with 36 ordinary normals replaced by semantic hard negatives;
- **C2:** C1 membership with three predicate-equivalent semantic-relation
  features removed.

Five frozen trace-level repetitions evaluate Logistic Regression, Decision
Tree, and Random Forest. One controlled device-grouped partition is retained as
a sensitivity analysis, not as proof of population-level unseen-device
generalization. Exact cohorts, preprocessing, models, seeds, thresholds, and
metrics are machine-readable in the Phase 2–4 manifests.

The central frozen result is semantic rather than a model leaderboard:
ordinary-normal-only training can appear strong while systematically
misclassifying legitimate near-boundary Timing behavior. Nonce provides a
contrasting family where hard negatives were already largely handled.

## Repository structure

```text
IoT_auth_simulator/
  simulator/                 lifecycle generation, profiles, transformations,
                             composition, serialization, and validation
  ml/icds/                   O03 features, frozen manifests, baseline pipeline,
                             Phase 4 orchestration, and paper derivation
artifacts/
  icds-timing-nonce-candidate/  frozen candidate and build manifest
  icds-ml-phase1/               structural extraction audit
  icds-ml-phase2/               experimental manifest and persisted splits
  icds-ml-phase3/               device-grouped cohort definition
  icds-ml-phase4/               predictions, provenance, metrics, and summaries
  icds-paper-figures/            derived publication figures and tables
docs/                        public reproducibility/schema/artifact guides
notebooks/                   executable paper-artifact notebook
scripts/                     frozen verification and public-snapshot tooling
tests/                       semantic, generation, dataset, and ML tests
```

Historical ML notebooks, legacy generated datasets, internal handover notes,
and milestone diaries are deliberately excluded from the publication snapshot.

## Installation

The frozen environment was verified with Python 3.14.2. From a clean checkout:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-publication.txt
export PYTHONPATH="$PWD/IoT_auth_simulator"
```

Windows PowerShell users should activate with `.venv\Scripts\Activate.ps1` and
set `$env:PYTHONPATH="$PWD\IoT_auth_simulator"`.

## Quick reproducibility path

All commands below run from the repository root.

```bash
# 1. Verify checked-in candidate, Phase 1–4, and paper-artifact hashes/counts.
python scripts/verify_publication.py --verify-only

# 2. Run the complete test suite.
python -m unittest discover -s tests

# 3. Regenerate the candidate in a temporary directory and compare it exactly.
python scripts/verify_publication.py --reproduce-candidate

# 4. Rerun the frozen 45+9 Phase 4 runs in a temporary directory and compare
#    predictions, metrics, and normalized provenance.
python scripts/verify_publication.py --reproduce-phase4

# 5. Regenerate publication figures/tables from verified frozen Phase 4 data.
python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/icds_phase4_paper_artifacts.ipynb
python scripts/verify_publication.py --verify-only
```

Candidate regeneration takes several minutes because the complete build is
performed twice to assert canonical equality. Phase 4 provenance includes new
execution timestamps on every rerun; scientific predictions and metrics are
compared byte-for-byte, while provenance is compared after removing only those
execution-time fields. The original frozen Phase 4 artifact-tree fingerprint
remains unchanged and is never replaced by a rerun fingerprint.

See [the reproducibility guide](docs/reproducibility.md) for expected
fingerprints and detailed commands.

## Data and manifests

The candidate JSON keeps detector observations, semantic truth, qualification,
and privileged provenance distinct. A field is not detector-eligible merely
because it appears in the same serialized package. See:

- [dataset and manifest guide](docs/dataset-and-manifests.md);
- [paper artifact map](docs/paper-artifact-map.md);
- [frozen Phase 4 results](docs/icds-phase4-scientific-results.md).

## Frozen fingerprints

| Artifact | Fingerprint |
|---|---|
| Population | `sha256:ec38b07c7b1831a4e4e2daec1d63fb73adae17aadfd442d22032cdb9c255cb6f` |
| Opportunity pool | `sha256:09a0554012f77f3da8e5c4ec3766d56e0fa9cfd8504239447c9ce255b6945504` |
| Applicability | `sha256:a66eae00676dba4fd303b553e5a486c568124c87341a3f30fb8e049421529904` |
| Composition plan | `sha256:fdb19b628bd511a2415e2eb745637e81c6ef28b60d996b0f4c0023867d5dd154` |
| Candidate rows | `sha256:be34fcbce532d4791cf3f27f056146215fdd255d1e28415e00e3e77f248d3f5c` |
| Candidate manifest | `sha256:bc80a1a261cd095c1b2f02b499e816577e69fa581310316a225f3c0c4b1d737c` |
| Phase 2 experiment manifest | `sha256:8e11d3a31dec7d9ad57fe59a6a658f58bfb7919e7bd744fb1600b594bcf0b8ed` |
| Grouped cohort | `sha256:b111c2710848bda6e3e1cfec2b9a018b65e95b7276116490a2e41599446bdf53` |
| Phase 4 frozen tree | `sha256:cc100d1e234d8c0605639def25351d4c756e99a9b7b99acf3639c03065a97c11` |

## Limitations

This artifact does not establish:

- empirical realism or calibrated real-world Device Profile distributions;
- representativeness of real IoT populations or networks;
- operational IDS validity or production deployment readiness;
- universal observability or anomaly detectability;
- universal benefit from hard-negative training;
- generalization beyond the controlled Timing and Nonce families;
- comprehensive authentication-anomaly coverage.

The reported ML evidence is family-specific and restricted to O03,
complete-trace offline classification, and the frozen LR/DT/RF baselines.

## Citation and license

The final paper title, complete author list, DOI, ORCIDs, institutional GitLab
URL, and archival record have not yet been supplied in authoritative metadata.
Consequently, this repository does not invent a `CITATION.cff` or Zenodo
record.

No software/data license is currently present. Public redistribution and the
proposed `icds-2026-artifact-v1.0` release therefore require a researcher
license decision before publication.
