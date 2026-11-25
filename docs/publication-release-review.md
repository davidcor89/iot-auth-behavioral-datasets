# Publication release review

## Proposed publication refs

The intended destination is `davidcor89/iot-auth-behavioral-datasets`.
The proposed publication uses the curated lineage in this repository.

| Local source ref | Proposed destination ref | Contents |
| --- | --- | --- |
| `refs/heads/publication/release-preparation` | `refs/heads/main` | Reviewed publication checkpoint, after later release approval |
| `refs/tags/icds-2026-artifact-v1.0` | `refs/tags/icds-2026-artifact-v1.0` | Unchanged annotated object `650ffa5321836d903a6ed40fa42dbac602292570`, pointing to `952718ab4faa191c870a045dfa8b7b1c075c0b36` |

No other branches or tags are proposed. The execution and archive checkpoints
listed in the [release index](../metadata/publication-release-index.json) are
already ancestors of the release branch. The historical generator tags belong
to the separate development repository and are referenced in the
[version registry](../metadata/scientific-version-registry.json).
Review the tip and reachability of both proposed refs again before publication.

## File inventory and history findings

[PUBLICATION_RELEASE_FILES.txt](../PUBLICATION_RELEASE_FILES.txt) classifies
Git-tracked paths into required release artifacts, required historical
verification inputs, and supplementary material retained in the curated
lineage. The old [snapshot allowlist](../PUBLICATION_FILES.txt) still serves
the historical export utilities. It omits later experiments and figures and
should not be used to assemble the V2 release. A file inventory governs
packaging. Publishing a Git ref also distributes its reachable history.

The pre-edit review enumerated 1,239 distinct file blobs across the proposed
branch and annotated tag. It searched for private-key markers, common provider
credentials, credentials in URLs, bearer and credential literals, local and
absolute home paths, addresses, and private configuration or cache filenames.
The scan found no credential matches in those categories. Review also
covered tracked binary formats, figure metadata, notebook output types,
origin of the manuscript assets, and the separation from the development
repository. These checks are not a guarantee against all encoded secrets or
identifying data.

- Three frozen records contain absolute workstation paths.
  `artifacts/r4.2-representation-sensitivity/pre-execution-verification.json`
  records the Python interpreter.
  `metadata/r4.1-freeze/freeze-manifest.json` and
  `metadata/r4.1-freeze/inputs/partition-manifest.json` record local evidence
  storage and preservation locations. These are engineering provenance,
  contain identifying filesystem context, and are not required local
  dependencies for external verification. The repository has portable
  artifact paths. Approve disclosure of the frozen locators before release.
- Eleven tracked paths or reachable versions contain transient work
  locators under the system temporary directory. Their function is execution
  or audit provenance. Historical commands can retain these paths because
  the current guide gives portable alternatives.
- Git history exposes an author/committer address, and the historical lineage
  identifies its authors and institutional host. These refs are unsuitable
  for anonymous review. Confirm that public attribution and contact exposure
  are acceptable.
- The candidate uses synthetic identifiers and device groups. The selection
  contains no tracked virtual environment, local environment file, agent
  cache, private key file, supervision handover, or reviewer report.
  The historical index declares missing submission materials. Technical
  feasibility audits and the historical notebook are retained as evidence
  and supplementary context.
- Figure 1's PNG embeds editable diagram markup. Inspection found no home
  path, email address, or external link in the embedded diagram model.
  The supplied bitmap and its embedded material still require rights
  confirmation. Its legacy abbreviation is tracked separately as an
  editorial issue.

The review preserved the frozen files and history. If disclosure of a frozen
locator or other historical content is declined, block publication of the
affected refs. Consider an access-controlled original archive and a reviewed
public derivative with an explicit provenance and fingerprint record. That
would require a separate decision and would change which checks a public
clone can perform.

## License and attribution decisions

[LICENSE](../LICENSE) applies Apache-2.0 to software under the existing
repository policy. [LICENSE-DATA](../LICENSE-DATA) provides CC BY 4.0 for
repository-generated scientific materials where legally applicable. The
candidate, persisted evaluation outputs, and programmatic figures have
repository generation records. Dependency pins identify external libraries
installed by the user. No separate vendored library, model weight package,
or font archive was located. Embedded fonts and historical rendered assets
remain subject to applicable third-party terms.

The [software citation](../CITATION.cff) retains the two existing software
authors. The [README](../README.md#citation-and-licensing) separately records
the three provided manuscript authors. No acceptance, publication date, DOI,
or approval to change license scope is recorded.

Before public distribution, obtain approval for:

1. Disclosure of identifying Git metadata and the three frozen workstation
   locators for the exact proposed refs.
2. Attribution and redistribution rights for the supplied methodology
   illustration, including its embedded diagram, and any third-party or
   historical submission material retained with the release.
3. The final software and manuscript citation metadata and any extension
   of the existing license scope. Paper authorship does not determine
   software ownership or authorship.

## CI and remaining release checks

The [GitHub workflow](../.github/workflows/verify-publication.yml) requests a
full Git checkout with tags, installs the pinned environment, and verifies the
archive, metrics, and features. It runs a selection of existing unittest
modules for normalization, feature extraction, split contracts, and
publication evidence. Full unittest discovery includes model fitting and
candidate generation, so the publication workflow uses only the modules
compatible with verification. The existing GitLab workflow is retained.

The GitHub workflow has not run on GitHub. The Ubuntu runner's ability to
resolve the exact Python build and dependency pins remains to be checked.
The known pytest collection issue is documented in the
[reproducibility guide](reproducibility.md#read-only-verification).
Publication is pending the approvals above and the later clean-environment
verification. No public URL availability or release DOI is assumed.
