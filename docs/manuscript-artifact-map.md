# Manuscript-to-artifact map

This map accompanies *Security-Semantics-Driven Behavioral Dataset Generation
for IoT Authentication Anomaly Detection*. The links point to authoritative
records in the release.

| Manuscript component | Authoritative evidence |
| --- | --- |
| Authentication semantics and dataset construction | [Scientific version registry](../metadata/scientific-version-registry.json), [candidate builder](../IoT_auth_simulator/simulator/composition/timing_nonce_candidate.py), [build manifest](../artifacts/icds-timing-nonce-candidate/build-manifest.json) |
| Timing and Nonce boundaries | [Timing construction](../IoT_auth_simulator/simulator/role_integration/direct_timing.py), [Nonce construction](../IoT_auth_simulator/simulator/role_integration/nonce_boundary.py), [boundary qualification](../IoT_auth_simulator/simulator/hard_negative_qualification.py), [semantic validator](../IoT_auth_simulator/simulator/validation.py), [frozen validation evidence](../validation/dataset-v1.0/manifest.json) |
| Frozen dataset and six role counts | [Dataset manifest](../datasets/dataset-v1.0/manifest.json), [candidate records](../artifacts/icds-timing-nonce-candidate/scientific-candidate.json), [role and field dictionary](dataset-and-manifests.md) |
| Detector observation contract | [O03 field and feature specification](../IoT_auth_simulator/ml/icds/feature_spec_icds_v1.json), [extractor](../IoT_auth_simulator/ml/evaluation/trace_features.py) |
| Full90 and feature exclusions | [Ordered 90/87/89/82 contracts](../metadata/r4.2-preexecution/feature-contracts.json), [seven generic timing statistics](../metadata/r4.2-preexecution/f2-generic-temporal-features.json) |
| Baseline five-repetition evaluation | [Partition and cohort assignments](../artifacts/icds-ml-phase2/icds_experiment_manifest_v1.json), [predictions](../artifacts/icds-ml-phase4/primary/predictions.jsonl), [aggregate summaries](../artifacts/icds-ml-phase4/primary/aggregate-summary.json) |
| Repeated device-disjoint evaluation | [Partition and cohort manifest](../metadata/r4.1-freeze/inputs/partition-manifest.json), [predictions](../artifacts/r4.1-device-disjoint-evaluation/predictions.jsonl), [paired comparisons](../artifacts/r4.1-device-disjoint-evaluation/paired-comparisons.json), [aggregate summaries](../artifacts/r4.1-device-disjoint-evaluation/aggregate-summary.json) |
| Representation sensitivity | [Frozen plan](../metadata/r4.2-preexecution/execution-manifest-cd.json), [new predictions](../artifacts/r4.2-representation-sensitivity/predictions.jsonl), [reused anchors](../artifacts/r4.2-representation-sensitivity/anchor-references.json), [paired comparisons](../artifacts/r4.2-representation-sensitivity/paired-comparisons.json), [aggregate summaries](../artifacts/r4.2-representation-sensitivity/aggregate-summary.json) |
| Reconstruct reported metrics | [Release verifier](../scripts/verify_release.py) and [reproduction guide](reproducibility.md). Use `python scripts/verify_release.py metrics --include-summaries`. |
| Figure 1 | [Supplied methodology illustration](../figures/methodology.png) |
| Figure 2 | [Five-repetition boundary exposure](../figures/figure2_boundary_exposure.pdf) |
| Figure 3 | [Device-disjoint boundary exposure and representation sensitivity](../figures/figure3_device_disjoint_representation.pdf) |

[Figure inventory](../figures/publication-figure-manifest.json) binds the three
manuscript assets. Figures 2 and 3 share the [plot-data CSV](../figures/figure2_figure3_plot_data.csv)
and [rendering script](../scripts/generate_paper_figures_2_3.py).

The [original single grouped sensitivity](../artifacts/icds-ml-phase4/grouped/grouped-sensitivity-table.json)
and [historical submission exports](../publication-artifacts/eval-v1.0/manifest.json)
are preserved separately. The five-repetition baseline also archives the
historical Reduced comparison. The repeated evaluation's Full90 and historical
Reduced runs supply the existing anchors for the representation-sensitivity
campaign. The latter adds 120 runs for the 89- and 82-feature representations.
