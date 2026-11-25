# IoT Authentication Behavioral Datasets

This research artifact accompanies the manuscript *Security-Semantics-Driven
Behavioral Dataset Generation for IoT Authentication Anomaly Detection*.
It studies how classifiers distinguish valid authentication behavior near a
security-property boundary from behavior that violates the property. The
method constructs scenarios from explicit authentication semantics,
qualifies boundary-valid cases, and validates violations independently
before using the traces for offline evaluation.

The simulator generates sequential authentication-session scenarios with
persistent device profiles, state-machine execution, and controlled
transformations. The dataset and all reported experiments are synthetic.

## Frozen dataset

[Dataset v1.0](datasets/dataset-v1.0/manifest.json) contains **336 traces** from
**64 persistent device profiles**, covering the Timing and Nonce property
families.

| Behavioral role | Traces |
| --- | ---: |
| Ordinary-valid | 96 |
| Timing boundary-valid, small positive causal margin | 48 |
| Timing violation, observed timestamp inversion | 48 |
| Nonce boundary-valid, recurrence within one attempt | 48 |
| Nonce boundary-valid, fresh nonce across linked attempts | 48 |
| Nonce violation, reuse across attempts | 48 |

The [data dictionary](docs/dataset-and-manifests.md) explains the separation
between detector-visible observations, ground truth, device grouping, and
privileged provenance. The [validation manifest](validation/dataset-v1.0/manifest.json)
links to the independent qualification and invariant evidence.

## Start here

Use Git and CPython 3.14.2 with the pinned dependencies. From the repository
root in a full Git clone:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproducibility.txt

python scripts/verify_release.py archive
python scripts/verify_release.py metrics
python scripts/verify_release.py features
```

These commands verify frozen bytes and historical Git objects, reconstruct
metrics from saved predictions, and inspect the detector representation.
For a source archive with no Git objects, use `archive --files-only`. Its
output lists the historical checks that require a clone.
The [reproducibility guide](docs/reproducibility.md) gives the environment,
expected outputs, supported tests, and separate replay procedures.

## Evaluation and manuscript evidence

- [Baseline evaluation](experiments/eval-v1.0/manifest.json) compares Ordinary
  and Boundary training over five repetitions with matched anomaly examples
  and test membership.
- [Repeated device-disjoint evaluation](metadata/r4.1-freeze/freeze-manifest.json)
  preserves 20 exact partitions and their matched training cohorts.
- [Representation sensitivity](metadata/r4.2-freeze/freeze-manifest.json)
  compares Full90, historical Reduced, and two targeted Timing feature
  exclusions on those same device-disjoint partitions.
- [Manuscript figures](figures/publication-figure-manifest.json) identify the
  exact figure assets, source values, and rendering workflow.

The [manuscript-to-artifact map](docs/manuscript-artifact-map.md) links the
source code, splits, predictions, and historical comparisons. The
[release index](metadata/publication-release-index.json) records scientific
checkpoints. The [historical submission index](submissions/icds2026/snapshot-manifest.json)
identifies the original ICDS artifact.

## Scope and limitations

The detector features summarize complete traces for offline evaluation. The
device-disjoint experiments test the frozen candidate under an explicit
grouping protocol. Timing margins are controlled construction parameters,
and anomaly labels identify observed model-defined property violations.
The evidence does not establish real-world attack outcomes, calibrated timing
thresholds, representative deployment populations, online detection
performance, or generalization to other property families.

## Citation and licensing

Use [CITATION.cff](CITATION.cff) for the software citation and include the
scientific identifiers from the release index when citing results. The
associated manuscript credits David A. Cordova Morales, Lamia Belhadj Sghaier,
and Gina Gallegos-García. Its authorship is recorded separately from the
software citation.

Software is covered by [Apache-2.0](LICENSE). Repository-generated scientific
materials are covered by [CC BY 4.0](LICENSE-DATA) where legally applicable.
See the [publication review](docs/publication-release-review.md) for pending
asset and attribution approvals. The intended publication repository is
`davidcor89/iot-auth-behavioral-datasets`.
