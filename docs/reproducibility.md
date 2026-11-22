# Scientific artifact reproducibility

This guide separates read-only verification of frozen evidence from optional
regeneration. Regeneration writes only to temporary directories and never
overwrites checked-in data, predictions, metrics, or figures.

## Environment

The frozen artifact was verified on macOS arm64 with CPython 3.14.2. Exact
direct dependency versions are in `requirements-reproducibility.txt`.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reproducibility.txt
export PYTHONPATH="$PWD/IoT_auth_simulator"
```

## Read-only verification

```bash
python scripts/verify_artifacts.py --verify-only
```

The verifier checks:

- all 207 frozen files against the V2-01 baseline SHA-256 inventory;
- the 336 sample IDs and six scientific-role counts;
- the dataset/build-manifest hashes and semantic fingerprints;
- the frozen O03 feature contract and experiment inputs;
- 45 primary runs / 3,780 predictions;
- 9 device-grouped sensitivity runs / 819 predictions;
- the Phase-4 artifact-tree fingerprint and paper-artifact hashes;
- all venue-neutral reference-manifest targets;
- all 365 historical snapshot files directly from the immutable submitted tag.

Run the complete regression suite with the repository source root available:

```bash
python -m unittest discover -s tests
```

The accepted semantic-neutrality hash remains:

```text
657cb57cb28515120b233704b618e64c8ec1a37d61ef129a1fbabd3060d7808c
```

## Dataset reproduction

```bash
python scripts/verify_artifacts.py --reproduce-candidate
```

The candidate builder runs in a temporary directory and compares the candidate
JSON, build manifest, and characterization byte-for-byte with `dataset-v1.0`.
It does not replace the frozen files.

Frozen construction settings remain 64 devices, eight opportunities per
device, seeds 4700–4705, semantic origin 1710000000, and 1,000-second trace
spacing. The two rejected opportunities are retained without resampling.

## Evaluation reproduction

```bash
python scripts/verify_artifacts.py --reproduce-evaluation
```

This reruns the frozen 45 primary and 9 grouped jobs into a temporary output.
Predictions, metrics, summaries, tables, figure data, and scientific
configuration are compared exactly. Only execution timestamps and the current
repository commit are normalized in run provenance. The original frozen tree
fingerprint remains:

```text
sha256:cc100d1e234d8c0605639def25351d4c756e99a9b7b99acf3639c03065a97c11
```

## Figures and tables

The submitted notebook is historical and remains byte-preserved:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/icds_phase4_paper_artifacts.ipynb
python scripts/verify_artifacts.py --verify-only
```

For verification, execute the notebook only in a disposable clone because
`--inplace` changes notebook execution metadata. Figures/tables are derived
from frozen Phase-4 evidence; the notebook does not train models.

## Venue-neutral path resolution

Use the reference paths below rather than hard-coding historical artifact
directories:

- `datasets/dataset-v1.0/manifest.json`;
- `validation/dataset-v1.0/manifest.json`;
- `experiments/eval-v1.0/manifest.json`;
- `publication-artifacts/eval-v1.0/manifest.json`;
- `metadata/scientific-version-registry.json`.

Python callers may use `ml.artifact_paths.resolve_scientific_path()` and the
`ml.evaluation` namespace. The `ml.icds` wrappers and
`scripts/verify_publication.py` remain for historical import/command
compatibility; exact submitted bytes remain available from the immutable tag.

## Known portability distinction

The repository does not claim byte-identical estimator internals across
platforms. Persisted predictions, cohorts, configurations, metrics, and
normalized provenance are the frozen evidence. Any pre-existing auxiliary
cross-platform fingerprint difference must be reported separately and must not
be “fixed” by changing scientific artifacts.
