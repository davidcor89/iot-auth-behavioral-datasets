# IoT Authentication Simulator and Behavioral Dataset Artifact

This repository contains a controlled synthetic IoT-authentication research
generator, semantic validation, a frozen behavioral dataset, leakage-safe
baseline experiments, and reusable publication artifacts. The scientific core
is venue-neutral. Historical submission material is preserved separately under
[`submissions/`](submissions/).

The simulator is a **scenario-based sequential authentication-session
generator** with state-machine execution, persistent Device Profiles,
independent semantic validation, and controlled normal, hard-negative, and
anomaly transformations. It is not a global discrete-event simulator, a
packet-level network emulator, or evidence of operational realism.

Canonical repository:
<https://gitlab.sorbonne-paris-nord.fr/david.cordovamorales/iot-authentication-simulator>

Software/artifact developers:

- David A. Córdova Morales — primary developer and maintainer;
- Lamia Belhadj Sghaier — developer.

## Frozen scientific identities

| Alias | Meaning |
|---|---|
| `behavioral-model-v1` | Original synchronized Behavioral Model baseline |
| `sync-v1.0.0` | Original synchronized generator baseline at `40617b723e68357af7b9e4c597b99efdaf63f02c` |
| `behavioral-model-v1.1` | Scoped semantic refinement declared by the frozen candidate records |
| `sync-v1.1.0` | Refined synchronized release at `74c459f72d214f7eb8ca9ace5a47c25d81589b2f` |
| `dataset-v1.0` | The unchanged 336-trace Timing+Nonce candidate, governed by v1.1 references |
| `eval-v1.0` | The unchanged C0/C1/C2 evaluation and grouped sensitivity run |

These aliases are packaging references, not new scientific versions. The
original v1 baseline, the v1.1 synchronized release, the candidate source
checkpoint, and the candidate construction commit are distinct, complementary
provenance layers. Their verified relationships are recorded without rewriting frozen bytes in
[`metadata/scientific-version-registry.json`](metadata/scientific-version-registry.json).

## Dataset and evaluation

`dataset-v1.0` contains 336 canonical traces from 64 persistent Device
Profiles: 96 ordinary normals, 144 semantic hard negatives, and 96 anomalies.
The two represented semantic families are Timing and Nonce. All observations,
event ordering, roles, ground truth, device assignments, validation outcomes,
and provenance remain byte-identical to the submitted artifact.

The complete-trace offline evaluation preserves three conditions:

- **C0 / Ordinary:** ordinary-normal-only normal training data;
- **C1 / Boundary:** the same size, anomaly membership, test set, and model
  settings, with 36 ordinary normals replaced by semantic hard negatives;
- **C2 / Reduced:** C1 membership with three semantic-relation features
  removed.

Five primary repetitions evaluate frozen LR/DT/RF baselines. A single
device-disjoint partition remains a sensitivity analysis, not a generalization
claim.

## Data-role boundary

The serialized candidate keeps these roles distinct:

- generated/source observations;
- detector-facing fields under the frozen O03 contract;
- ground truth;
- semantic validation results;
- privileged/debug/internal fields;
- provenance and experiment metadata.

The detector contract exposes only the eleven recorded event fields listed in
the frozen feature manifest. Device identifiers remain available for grouping
and split enforcement even when excluded from model input. Privileged role,
profile, invariant, qualification, and transformation metadata never become
detector inputs.

## Repository map

```text
IoT_auth_simulator/
  simulator/                     generator, profiles, transformations, validation
  ml/evaluation/                 venue-neutral compatibility entry points
  ml/icds/                       historical import compatibility wrappers
datasets/dataset-v1.0/           venue-neutral dataset reference manifest
validation/dataset-v1.0/         venue-neutral validation reference manifest
experiments/eval-v1.0/           venue-neutral experiment reference manifest
publication-artifacts/eval-v1.0/ venue-neutral figure/table reference manifest
submissions/icds2026/            immutable historical submission snapshot manifest
metadata/                        path map, inventories, scientific version registry
artifacts/                       unchanged frozen scientific evidence
docs/                            methodology, migration, versioning, reproducibility
scripts/                         verification and snapshot tooling
tests/                           semantic, generation, dataset, ML, integrity tests
```

The reference manifests avoid duplicating large frozen objects. Their paths and
hashes are checked by `scripts/verify_artifacts.py`.

## Installation and verification

The frozen environment used CPython 3.14.2. From a fresh clone:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reproducibility.txt
export PYTHONPATH="$PWD/IoT_auth_simulator"

python scripts/verify_artifacts.py --verify-only
python -m unittest discover -s tests
```

Verification is read-only. Optional regeneration writes only to temporary
directories and never overwrites frozen evidence:

```bash
python scripts/verify_artifacts.py --reproduce-candidate
python scripts/verify_artifacts.py --reproduce-evaluation
```

See [reproducibility](docs/reproducibility.md), [dataset and manifests](docs/dataset-and-manifests.md),
[artifact versioning](docs/artifact-versioning.md), and the
[V2-01 migration report](docs/repository-migration-v2-01.md).

## Historical submissions

The annotated tag `icds-2026-artifact-v1.0` is immutable and resolves to
`952718ab4faa191c870a045dfa8b7b1c075c0b36`. Its file-level snapshot is in
[`submissions/icds2026/snapshot-manifest.json`](submissions/icds2026/snapshot-manifest.json).
The repository contains generated figures, tables, results, and original
venue-specific manifests, but no submitted manuscript source/PDF, reviewer
reports, or external submission-system metadata; those absences are explicit
in the snapshot manifest.

## Versioning rule

A new conference target never creates a scientific version. A new dataset or
evaluation version is required only when generated observations, semantic
families/invariants, ground truth, validation semantics, observation contract,
split protocol, condition definition, or model/evaluation protocol changes.
Organizational changes use ordinary Git commits or packaging revisions.

## Limitations

The artifact does not establish empirical realism, representative device or
network populations, operational IDS validity, universal observability,
universal benefit from hard-negative training, or generalization beyond the
controlled Timing and Nonce setting.

Software is Apache-2.0 ([LICENSE](LICENSE)); repository-generated scientific
materials are CC-BY-4.0 where legally applicable ([LICENSE-DATA](LICENSE-DATA)).
Citation metadata is in [CITATION.cff](CITATION.cff).
