# V2-01 repository and artifact normalization report

## Status

V2-01 is an organizational migration on branch
`chore/v2-artifact-normalization`, based on submitted repository commit
`952718ab4faa191c870a045dfa8b7b1c075c0b36`. No scientific generator,
dataset, validation, feature, split, model, prediction, metric, or publication
source-data value was changed.

Workspaces 02 and 03 have resolved the apparent generator-reference mismatch
as complementary provenance layers. V2-01-P records that resolution without
changing any frozen scientific evidence.

## Development-continuity audit

The historical repository `PFE_simulator` and the curated USPN repository were
compared before normalization. All files common to both repositories were
byte-identical except the curated public README. The curated repository already
contains the complete scientific implementation, ML evaluation package,
candidate, Phase 1–4 evidence, figures/tables, and 46 test modules.

| Finding | Classification | Disposition |
|---|---|---|
| Three historical LR/DT/RF notebooks | Retain in archive | Historical exploratory evidence; not reusable pipeline authority |
| Historical `iot_auth_ml_events.json` | Retain in archive | Pre-canonical generated data; not part of the 336-trace candidate |
| Device Profile/HN/feasibility methodology audits | Candidate for selective integration | Useful internal provenance, but not required to execute or verify the curated artifact |
| Six C1 supervisor/ML handover documents | Requires researcher decision | Internal governance/handover material intentionally excluded from the public repository |
| Empty `pipeline/README.md` and old broad requirements | Retain in archive | Superseded by curated exact environment and ML package |
| Simulator, validation, profile, composition, ML and publication code | Already present in USPN | Common files are byte-identical |
| Complete frozen predictions/metrics/figures and public tests | Already present in USPN | Curated repository is more complete than tracked historical state |

Private/internal caches, virtual environments, local outputs, `.DS_Store`, and
untracked handover notes were not imported. No significant missing executable
scientific functionality was found, so no history merge or selective code
integration was required.

## Baseline capture

- Branch before migration: `main`.
- Baseline/immutable submission commit:
  `952718ab4faa191c870a045dfa8b7b1c075c0b36`.
- Immutable annotated tag: `icds-2026-artifact-v1.0` (tag object
  `650ffa5321836d903a6ed40fa42dbac602292570`).
- Remote: USPN GitLab canonical repository.
- Baseline working tree: clean.
- Frozen artifact inventory: 207 files under `artifacts/`, recorded individually
  in `metadata/scientific-artifact-inventory.json`.
- Historical public snapshot: 365 files, recorded in
  `submissions/icds2026/snapshot-manifest.json` and verified from the tag.
- Venue-reference inventory: `metadata/venue-reference-inventory.json`.

The repository contains four generated paper PDFs and four TeX table fragments,
but no submitted manuscript source/PDF, reviewer reports, or external
submission-system metadata. Missing entries are explicit in the snapshot.

## Venue-reference classification

All case-insensitive `ICDS`, `CODASPY`, and `IFIPSEC` variants were searched in
tracked paths and text. No CODASPY or IFIPSEC scientific assets were found.
Remaining ICDS references fall into these deliberate classes:

- frozen scientific evidence and embedded identifiers;
- the byte-preserved submitted `ml.icds` implementation namespace;
- historical submission scripts, notebook, documentation, and tests;
- `submissions/icds2026/` historical metadata.

Active, venue-neutral access is provided through `ml.evaluation`, versioned
reference manifests, and `scripts/verify_artifacts.py`. The machine inventory
records every affected tracked file and occurrence count.

## Path normalization

No frozen scientific object was relocated or duplicated. The complete mapping
is machine-readable in `metadata/path-mapping.json`.

| Original frozen path | Venue-neutral reference |
|---|---|
| `artifacts/icds-timing-nonce-candidate/` | `datasets/dataset-v1.0/manifest.json` |
| candidate validation evidence | `validation/dataset-v1.0/manifest.json` |
| Phase 1–4 definitions/results | `experiments/eval-v1.0/manifest.json` |
| submitted figures/tables | `publication-artifacts/eval-v1.0/manifest.json` |
| `IoT_auth_simulator/ml/icds/` | active code moved to `IoT_auth_simulator/ml/evaluation/`; historical wrappers retained |
| submitted public snapshot | `submissions/icds2026/snapshot-manifest.json` |

Compatibility aliases are explicit. They prevent duplicated sources of truth
and retain exact submission reproduction.

## Frozen objects explicitly unchanged

- all 336 candidate samples, sample IDs, observations, and event order;
- 64 profile assignments and persistent device relationships;
- 96 ordinary normals, 144 hard negatives, and 96 anomalies;
- invariant definitions, validation outcomes, GT, and privileged provenance;
- O03 observation eligibility and extracted feature contracts;
- all primary and grouped split assignments and seeds;
- all 45 primary and 9 grouped runs;
- all 3,780 primary and 819 grouped predictions;
- all metrics, figure/table source data, and submitted figures/tables;
- generator and Behavioral Model source code;
- semantic-neutrality hash and existing tests.

Baseline and post-migration SHA-256 equality is enforced for all 207 frozen
files by `scripts/verify_artifacts.py`.

## Version registry and verified provenance layers

`metadata/scientific-version-registry.json` introduces the packaging aliases
`behavioral-model-v1`, `behavioral-model-v1.1`, `sync-v1.0.0`, `sync-v1.1.0`,
`dataset-v1.0`, and `eval-v1.0`.

The verified provenance relationship is:

- `behavioral-model-v1` and `sync-v1.0.0` at `40617b…` identify the original
  synchronized baseline;
- `behavioral-model-v1.1` is the subsequent scoped semantic refinement;
- `sync-v1.1.0` at `74c459f…` is the refined synchronized release and is the
  generator release declared for the frozen candidate;
- `fc97fdc…` is the source checkpoint declared by the candidate builder and is
  historically the Timing+Nonce feasibility commit;
- `2a9883b…` is the candidate construction/publication commit that first adds
  the builder and the three frozen candidate artifacts.

The four commits form an ancestor chain in that order. This establishes their
historical relationship, but not the exact checked-out commit used for the
original generation process. Neither the candidate build manifest nor another
frozen provenance record contains an execution-commit field. The exact
generation execution commit therefore remains **unverified**; it is not
inferred as `fc97fdc…` or `2a9883b…` from ancestry. The referenced historical
objects are absent from the curated repository and remain external provenance.

Commit inspection also distinguishes their roles: `40617b…` repaired the
original synchronized generator; `74c459f…` froze v1.1 release metadata and
contracts; `fc97fdc…` added feasibility analysis, audit evidence, and tests;
and `2a9883b…` added the candidate builder, candidate artifacts, tests, and a
methodology-status update. The candidate files and builder in the curated
repository are byte-identical to those at `2a9883b…`.

V2-01-P verification retained the following concrete evidence:

- the annotated tags resolve to `sync-v1.0.0` → `40617b…` and
  `sync-v1.1.0` → `74c459f…`;
- Git confirms `40617b…` → `74c459f…` → `fc97fdc…` → `2a9883b…` as an
  ancestor chain;
- the frozen evaluation manifest declares `behavioral-model-v1.1`,
  `sync-v1.1.0`, source checkpoint `fc97fdc…`, and publication commit
  `2a9883b…`;
- the candidate build manifest declares `fc97fdc…` as its source checkpoint
  and does not declare an execution commit;
- the curated builder, build manifest, and candidate have the same SHA-256
  hashes as their historical `2a9883b…` versions: `5ab1b896…`, `6524346f…`,
  and `fa22706b…`, respectively.

## Commands and verification

Baseline commands:

```text
git status --short --branch
git remote -v
git tag --list
rg -n -i '(ICDS|CODASPY|IFIPSEC|IFIP[- _]?SEC)'
PYTHONPATH=IoT_auth_simulator python -m pytest -q
PYTHONPATH=IoT_auth_simulator python scripts/verify_publication.py --verify-only
PYTHONPATH=IoT_auth_simulator python -m unittest discover -s tests -q
```

Baseline results:

- frozen publication verification: passed (336 samples, 45+9 runs, 20 paper artifacts);
- pytest: 463 passed, 1 skipped, 1 pre-existing collection error because the
  helper `test_population` in `test_dp2_profile_provenance.py` is collected as
  a fixture-based test;
- unittest is the repository's supported full-suite runner;
- running pytest without `PYTHONPATH=IoT_auth_simulator` produces 46
  pre-existing import collection errors.

Final results are recorded in the final V2-01 handoff and checked by CI through
`scripts/verify_artifacts.py --verify-only` plus unittest discovery.

Post-migration verification completed locally:

- neutral integrity verifier: passed; 207/207 frozen scientific file hashes,
  365/365 immutable-tag snapshot records, all alias references, 336 sample IDs,
  and all six role counts verified;
- full supported regression suite: 468 tests passed, 1 skipped;
- candidate reproduction: 3/3 outputs byte-identical;
- evaluation reproduction: 13/13 scientific outputs equivalent and 56/56
  provenance files equivalent after the previously frozen normalization;
- frozen artifacts and `IoT_auth_simulator/simulator/`: no Git diff;
- baseline SHA-256 list: all 207 files unchanged;
- Markdown relative-link scan: no missing links;
- venue-neutral fresh snapshot export: 400 allowlisted files, completed outside
  the repository without absolute private paths;
- JSON parsing, Python compilation, and `git diff --check`: passed.

V2-01-P then reran the neutral integrity verifier and 18 focused registry,
manifest, and orchestration tests. All passed. The verifier reconfirmed 336
candidate samples, the six frozen role counts, 45 primary runs, 9 grouped runs,
3,780 primary predictions, 819 grouped predictions, 20 paper artifacts, and
the unchanged Phase 4 tree fingerprint. No candidate generation or model
execution was performed for this provenance-only correction.

## Compatibility and reproducibility

- Original artifact paths, the `ml.icds` compatibility namespace,
  `requirements-publication.txt`, and `scripts/verify_publication.py` are
  retained for submission compatibility; exact historical code bytes remain
  recoverable from the immutable tag.
- Neutral callers use `ml.evaluation`, `ml.artifact_paths`, and the versioned
  reference manifests.
- Optional reruns are isolated in temporary directories and cannot overwrite
  frozen evidence.
- The historical tag remains unchanged; no tag is created by V2-01.

## Remaining manual actions

1. Preserve the exact candidate generation execution commit as unverified
   unless an authoritative execution record is later recovered.
2. Add exact manuscript/reviewer/submission-system files only if authoritative
   copies become available; update the snapshot as a new packaging commit,
   never by moving the historical tag.
3. Review this migration before merging the branch into `main`.

No V2 scientific extension, new trace, new split, retraining, or new result is
part of this migration.
