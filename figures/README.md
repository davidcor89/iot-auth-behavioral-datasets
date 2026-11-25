# Manuscript figure package

The [figure manifest](publication-figure-manifest.json) records the supplied
methodology illustration (Figure 1) and the programmatically generated
Figures 2 and 3. The historical submission figures remain under
`artifacts/icds-paper-figures/`.

To render Figures 2 and 3, set up the environment in
[the reproducibility guide](../docs/reproducibility.md) and run from the
repository root:

```bash
source .venv/bin/activate
python scripts/generate_paper_figures_2_3.py
```

Run in a disposable checkout when comparing with the archived exports.
PDF and PNG bytes may vary across rendering environments even when the
plotted values and visual content are equivalent.

The exact input paths, SHA-256 hashes, reconstructed summaries, checks, and
output hashes are recorded in `figure_generation_verification.json`. Exact
plotted values and their originating run IDs are retained in
`figure2_figure3_plot_data.csv`.

## Proposed captions

**Figure 2.** Five-repetition controlled comparison of Ordinary-only and
Boundary-exposed training. Points show repetition-level results; diamonds and
bars show medians and interquartile ranges. Boundary exposure reduces the
false-positive rate on small-positive-margin Timing boundary-valid cases,
while Timing-anomaly recall remains model dependent.

**Figure 3.** Stability and representation sensitivity over 20 predefined
device-disjoint partitions. (a) Paired false-positive rates on Timing
boundary-valid cases under Ordinary-only and Boundary-exposed training. Lines join results
from the same partition. (b) Timing-anomaly recall for Boundary-trained models
under the Full 90-feature representation, removal of the direct Timing-margin
summary (89 features), and additional removal of seven generic temporal
features (82 features). Points show partition-level results; diamonds and bars
show medians and interquartile ranges. The 82-feature representation retains
Timing-related structural evidence, including the relation-count feature.
