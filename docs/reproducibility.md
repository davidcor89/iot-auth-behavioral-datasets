# ICDS 2026 artifact reproducibility

This guide distinguishes verification of the frozen evidence from regeneration
of that evidence. Regeneration never overwrites the checked-in artifacts.

## Environment

The publication checks were run on macOS arm64 with CPython 3.14.2 using the
exact direct dependency versions in `requirements-publication.txt`. The
repository does not claim byte-identical cross-platform estimator internals.
The scientific comparison is instead performed using persisted predictions,
metrics, cohorts, configurations, and normalized run provenance.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-publication.txt
export PYTHONPATH="$PWD/IoT_auth_simulator"
```

## Read-only verification

```bash
python scripts/verify_publication.py --verify-only
```

This checks:

- candidate and build-manifest file hashes;
- 336 samples and the six frozen role counts;
- 240 normal / 96 anomaly truth counts;
- 144 qualified hard negatives;
- 512 attempted, 510 admitted, and 2 rejected base opportunities;
- frozen Phase 1–3 input fingerprints;
- 45 primary runs and 3,780 primary predictions;
- 9 grouped runs and 819 grouped predictions;
- the Phase 4 tree fingerprint and key file hashes;
- every derived paper artifact against its manifest hash.

## Candidate reproduction

```bash
python scripts/verify_publication.py --reproduce-candidate
```

The wrapper invokes the real candidate entry point in a temporary directory.
The implementation independently performs two complete builds before writing.
The candidate JSON, build manifest, and characterization report must be
byte-identical to the checked-in artifacts.

Frozen configuration:

| Setting | Value |
|---|---:|
| Devices | 64 |
| Opportunities per device | 8 |
| Device-construction seed | 4700 |
| Population seed | 4701 |
| Pool seed | 4702 |
| Planning-root seed | 4703 |
| Role-assignment seed | 4704 |
| Transformation seed | 4705 |
| Semantic origin | `1710000000` |
| Trace spacing | 1,000 seconds |
| Calibration | controlled synthetic, non-final, uncalibrated |

Expected rejected opportunities, with no resampling:

- `population-device-00000037`, opportunity 3;
- `population-device-00000060`, opportunity 8.

Authoritative fingerprints:

```text
population          sha256:ec38b07c7b1831a4e4e2daec1d63fb73adae17aadfd442d22032cdb9c255cb6f
opportunity pool    sha256:09a0554012f77f3da8e5c4ec3766d56e0fa9cfd8504239447c9ce255b6945504
applicability       sha256:a66eae00676dba4fd303b553e5a486c568124c87341a3f30fb8e049421529904
composition plan    sha256:fdb19b628bd511a2415e2eb745637e81c6ef28b60d996b0f4c0023867d5dd154
composition build   sha256:bfcfd48e9e6b213325e12c3dd3cf25efad0677753eb9623e406859625b13e3ef
candidate rows      sha256:be34fcbce532d4791cf3f27f056146215fdd255d1e28415e00e3e77f248d3f5c
privileged metadata sha256:36a2714360b285ac6651e34c4b62333e1b915014a17470aeb21ab39f5c1e3cde
build manifest      sha256:bc80a1a261cd095c1b2f02b499e816577e69fa581310316a225f3c0c4b1d737c
```

## Test suite and semantic neutrality

```bash
python -m unittest discover -s tests
```

The suite includes the accepted semantic-neutrality hash:

```text
657cb57cb28515120b233704b618e64c8ec1a37d61ef129a1fbabd3060d7808c
```

The hash is not updated when code changes; a mismatch is a scientific blocker.

## Phase 4 reproduction

```bash
python scripts/verify_publication.py --reproduce-phase4
```

The wrapper invokes the frozen orchestration with explicit scientific-execution
authorization and a new temporary output directory. It verifies:

- all 45 primary and 9 grouped runs are present;
- predictions, metrics, summaries, figure data, and frozen tables are
  byte-identical;
- run provenance is identical after removing only `created_at_utc`;
- the execution manifest is identical after removing only start/completion
  timestamps.

The original frozen Phase 4 tree fingerprint is:

```text
sha256:cc100d1e234d8c0605639def25351d4c756e99a9b7b99acf3639c03065a97c11
```

A rerun necessarily has a different whole-tree fingerprint because timestamps
are provenance and `claim-audit.json` is a frozen post-execution review artifact
rather than an orchestrator output. The verifier reports that distinction and
never replaces the frozen fingerprint.

The expected sklearn 1.9 warning about the deprecated explicit Logistic
Regression `penalty` parameter is retained rather than changing the frozen
model configuration.

## Paper artifacts

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/icds_phase4_paper_artifacts.ipynb
python scripts/verify_publication.py --verify-only
```

The notebook first verifies the Phase 4 hashes, then derives four figures and
four tables. It does not fit models or change frozen evidence. PDF and SVG are
publication formats; PNG files are previews. CSV tables are machine-readable,
and TeX files are generated fragments.

## Dedicated public snapshot

The research repository contains internal history that is deliberately absent
from the public artifact. Create an allowlisted snapshot outside the repository:

```bash
python scripts/export_publication_snapshot.py /tmp/icds-2026-public
```

The command refuses to overwrite an existing directory. It creates
`PUBLICATION_SNAPSHOT.json`, recording every exported path, size, and SHA-256.
The snapshot can then be initialized as the dedicated USPN GitLab repository
after the repository URL, namespace, visibility, license, and authorization are
confirmed.

