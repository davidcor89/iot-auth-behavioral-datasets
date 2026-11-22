# Scientific artifact versioning

## Principle

Scientific identity follows scientific content, not the publication venue or
repository layout. V2-01 introduces reference aliases and packaging paths only;
it changes no trace, label, invariant, split, prediction, metric, or figure
source value.

## Version layers

| Layer | Current alias/reference | Changes when |
|---|---|---|
| Behavioral model | `behavioral-model-v1` baseline; candidate governed by `behavioral-model-v1.1` | The authoritative behavioral semantics change |
| Generator | `sync-v1.0.0` baseline at `40617b…`; candidate governed by `sync-v1.1.0` at `74c459f…` | Generator semantics covered by the release change |
| Dataset | `dataset-v1.0` | Population, observation, role/GT, semantic family, invariant, or validation content changes |
| Evaluation | `eval-v1.0` | Observation/feature contract, conditions, splits, seeds, models, metrics, or evaluation protocol changes |
| Packaging | Git commit / `V2-01` | Paths, documentation, aliases, CI, or integrity tooling change without scientific content changing |

A conference name or new submission target never triggers a scientific version.

## Aliases and original identities

The machine-readable authority for repository aliases is
[`metadata/scientific-version-registry.json`](../metadata/scientific-version-registry.json).
Aliases resolve to original frozen paths and fingerprints. They do not rewrite
the embedded identifiers of frozen manifests.

The provenance layers are complementary. `behavioral-model-v1` and
`sync-v1.0.0` / `40617b…` identify the original synchronized baseline. The
frozen candidate declares the scoped `behavioral-model-v1.1` refinement and
the `sync-v1.1.0` release at `74c459f…`. Its build records separately name
`fc97fdc…` as the candidate source/feasibility checkpoint and `2a9883b…` as
the candidate construction/publication commit. No frozen record identifies an
exact generation execution commit, so that value remains explicitly
unverified rather than inferred from ancestry.

## Immutability

- `icds-2026-artifact-v1.0` must never move or be recreated.
- Frozen manifests are never edited to adopt an alias.
- Frozen datasets/results are verified by SHA-256 and scientific fingerprints.
- A rerun writes outside frozen paths and cannot replace submitted evidence.
- Historical snapshots bind to a tag/commit, never to moving `main`.

## Future extensions

The directory scheme admits additional datasets, observation contracts,
semantic families, repeated device-disjoint evaluations, and publication
outputs by adding versioned directories and registry entries. V2-01 does not
implement any of those scientific extensions.
