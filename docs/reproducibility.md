# Scientific artifact reproducibility

Start with archive verification and result reconstruction. Both use the
published evidence and require no model fitting. Experiment replay fits
classifiers on the frozen inputs. Candidate regeneration runs the simulator
and compares its outputs with the frozen dataset. Use isolated destinations
for any command that writes outputs.

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
python scripts/verify_release.py archive
python scripts/verify_release.py metrics
```

The `archive` command verifies:

- all 207 frozen files against the V2-01 baseline SHA-256 inventory;
- the 336 sample IDs and six scientific-role counts;
- the dataset/build-manifest hashes and semantic fingerprints;
- the frozen O03 feature contract and experiment inputs;
- 45 primary runs / 3,780 predictions;
- 9 device-grouped sensitivity runs / 819 predictions;
- the Phase-4 artifact-tree fingerprint and paper-artifact hashes;
- all venue-neutral reference-manifest targets;
- the R4.1 archive (566 inventoried files), the R4.2 archive (379 files),
  the 19-file R4.2 pre-execution package and references to its R4.1 anchors;
- the three V2 manuscript figures, generated previews and plot-data hashes;
- all 365 historical snapshot files directly from the immutable submitted tag.

Full verification requires a Git checkout with the historical annotated tag
and its objects. The command works at the release checkout and returns a
nonzero exit status on a failed required check. For a source archive without
Git metadata, use `python scripts/verify_release.py archive --files-only`.
That mode reports `PASS_FILE_CHECKS_ONLY` when successful and explicitly lists
the historical tag and Git-object checks it omits. Archive verification checks
the frozen semantic validation evidence and its hashes. It does not regenerate
the candidate or rerun semantic construction.

The `metrics` command reconstructs per-run metrics from saved predictions and
compares condition summaries, linear quartiles and ranges, paired differences
and the 360 manuscript plot values. Expected totals are 45 baseline runs
(3,780 predictions), nine historical grouped runs (819), 180 R4.1 runs
(15,120), and 120 new R4.2 runs (10,080). R4.2 also uses 120 existing R4.1
Full and historical Reduced runs as anchors. These are counted separately
from new executions. Add `--include-summaries` to print the reconstructed
distributions and paired summaries as JSON. Differences of condition-level
medians and medians of paired differences are separate quantities.

The legacy `python scripts/verify_artifacts.py --verify-only` and
`python scripts/verify_r4_1_freeze.py` commands remain available.

Run the complete regression suite with the repository source root available:

```bash
python -m unittest discover -s tests
```

The archived R4.2 execution handoff recorded a pytest collection issue in
`tests/test_dp2_profile_provenance.py` involving a helper collected as a test.
Use the supported unittest command above.

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

This tests whether the preserved implementation reproduces the three archived
files. The original candidate execution checkout remains **unverified**.
The candidate declares `behavioral-model-v1.1` and `sync-v1.1.0`.
The original v1 release, the v1.1 release, the feasibility checkpoint and
the candidate-construction commit have separate roles in
`metadata/scientific-version-registry.json`.

Frozen construction settings remain 64 devices, eight opportunities per
device, seeds 4700–4705, semantic origin 1710000000, and 1,000-second trace
spacing. The two rejected opportunities are retained without resampling.

## Detector feature inspection

```bash
python scripts/verify_release.py features
```

This reconstructs the frozen structural audit for 336 traces and prints the
ordered feature lists and fingerprints for Full (90), historical Reduced (87),
without Timing summary (89), and without summary or timing statistics (82).
Add `--sample SAMPLE_ID` to print one Full90 vector. Use a `sample_id` from
the frozen candidate linked by `datasets/dataset-v1.0/manifest.json`.

Each vector derives from the sample's ordered `detector_observations` at
complete-trace scope. O03 declares 11 fields, ten of which contribute to
Full90. `token_id` is declared and unused by this extractor. Sample IDs join
trace records and labels are held separately under privileged metadata.
The ordered contracts are in `metadata/r4.2-preexecution/feature-contracts.json`.
The expected feature-row fingerprints are
`sha256:4f3510f849c978dc30d89e4b661446133669503f51abff9e6517521dd747008f`
for Full90 and
`sha256:39c7a57dd4c989bfc70c88379235467254760255318b06b90ff5713de2a3368e`
for historical Reduced. These hash the full ordered row collections and are
distinct from the fingerprints of the ordered feature names.
For A and B, the archived contract hashes reproduce when the two
`historical_r4_1_feature_profile*` linkage fields are also excluded from
the hashed payload. The embedded canonicalization description does not list
those two exclusions. The release verifier checks the linkage independently
against the frozen historical feature profiles.

## Baseline evaluation replay

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

## R4.1 and R4.2 experimental replay

These commands fit classifiers. Use them only for experiment replay. Start
from a full repository clone that has the historical commits and annotated
tag. Run the setup from the root of the release checkout with the documented
environment activated. Each replay uses its original execution checkout and
the exact executable stored later in the corresponding archive. The original
checkpoint guards remain in force.

```bash
release_root="$PWD"
replay_root="$(mktemp -d)"
git clone --no-local "$release_root" "$replay_root/r41"
git clone --no-local "$release_root" "$replay_root/r42"
```

**R4.1.** The runner infers the checkout root from its module location, so
copy the archived source to that exact module path in the disposable clone.
The partition JSON and accepted-partition CSV are repository-relative mirrors
of the frozen inputs. This replays the exact 20 partitions and memberships,
including the matched cohorts. The partition manifest describes the original
seeded search, but the release does not provide a standalone seed-to-partition
enumerator.

```bash
git -C "$replay_root/r41" switch --detach 97003b1d9448caa105b6e206ceb5deb18c02a207
cp "$release_root/metadata/r4.1-freeze/execution-implementation/r4_1_execution.executed.py" \
  "$replay_root/r41/IoT_auth_simulator/ml/evaluation/r4_1_execution.py"
(
  cd "$replay_root/r41" || exit 1
  PYTHONPATH=IoT_auth_simulator python -m ml.evaluation.r4_1_execution \
    --partition-manifest "$release_root/metadata/r4.1-freeze/inputs/partition-manifest.json" \
    --accepted-partitions "$release_root/metadata/r4.1-freeze/inputs/accepted-partitions.csv" \
    --output "$replay_root/r41-results"
)
```

Expect 180 runs and 15,120 predictions in the new output directory.
Use the same replay invocation with `--validate-only` on that scratch output
to check its results. The historical runner writes an integrity report in the
output directory.

**R4.2.** The executor accepts a checkout root but requires its original
branch name, exact commit and a clean tree. Invoke the archived file from
the release checkout so the execution checkout stays clean. The checkout at
`bd00f09` already contains the R4.1 anchors and the R4.2 plan.

```bash
git -C "$replay_root/r42" switch -c research/r4.2-representation-sensitivity \
  bd00f098ec86902e5c01e5da1cf1eb4c6e92d77b
(
  cd "$replay_root/r42" || exit 1
  PYTHONPATH=IoT_auth_simulator python \
    "$release_root/artifacts/r4.2-representation-sensitivity/execution-implementation/r4_2_execution.executed.py" \
    --root . --output "$replay_root/r42-results"
)
```

Expect 120 new C/D runs, 10,080 predictions and 240 paired comparisons
including contrasts with the existing A/B anchors. Compare predictions,
metrics and summaries with the corresponding frozen files. Replay output
manifests can include different execution times or paths. Use the release
verifier to check the frozen artifacts rather than running the original
freeze validator against the release HEAD.

## Figures and tables

The final V2 figures are indexed in
`figures/publication-figure-manifest.json`. From a disposable release clone
with the documented environment, run
`python scripts/generate_paper_figures_2_3.py`. It reconstructs plot inputs
from frozen predictions and writes two PDFs, two PNG previews, the source
CSV and generation verification. Figure 1 is a supplied static illustration.
Rendering environments may produce different PDF or PNG bytes for the same
plot values and visual content.

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
