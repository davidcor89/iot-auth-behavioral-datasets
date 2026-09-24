"""Read-only publication artifacts derived from the frozen ICDS Phase-4 results.

This module does not fit models, create predictions, or alter scientific inputs.
It verifies the frozen Phase-4 evidence and deterministically transforms the
already-recorded metrics into paper-oriented figures and tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHASE4_TREE_FINGERPRINT = (
    "sha256:cc100d1e234d8c0605639def25351d4c756e99a9b7b99acf3639c03065a97c11"
)
EXPECTED_HASHES = {
    "primary/predictions.jsonl": "04fc533592a50e23dab67ba22d3bd7ebf23239702de4a9daa149d586fbdc5539",
    "primary/run-provenance.json": "9a27f9ad469502b4dcd7d0408eff2788a877d51b96dd96751e434fe1343b2594",
    "primary/per-run-metrics.json": "50ee928098644035869dfb48306255b2f749c614111338f8b43445cbb79404aa",
    "primary/aggregate-summary.json": "d33f909862039afd727250b67705c6359d974cde75849455c967efb8b70206ca",
    "grouped/predictions.jsonl": "d76cb118f7d37082d2ee26e9cc90e79bd864875e65fdaca6b8820f04f931fb25",
    "grouped/run-provenance.json": "bd560a4b8eac22e7778f11787449d0c57c32f6af3cf665e93525bf70d7089c13",
    "grouped/per-run-metrics.json": "7568feb63040978278cd202d63e8853197698054060cd25c011e4f780a25a687",
    "grouped/grouped-sensitivity-table.json": "ef69601c1b57492b6fbb1471f1e113a7ddfd1fe2610699707ee091d8a7e4aeb0",
    "claim-audit.json": "b22a62e7eb014c7c492d4031be7d1ec97d5d16b65d55bb96026fa33b7ef33293",
}

MODELS = ("LR", "DT", "RF")
CONDITIONS = ("C0", "C1", "C2")
NORMAL_ROLES = (
    "ordinary_normal",
    "hard_negative:small_positive_causal_margin",
    "hard_negative:legitimate_same_attempt_nonce_recurrence",
    "hard_negative:fresh_nonce_cross_attempt",
)
ROLE_LABELS = {
    "ordinary_normal": "Ordinary normal",
    "hard_negative:small_positive_causal_margin": "Timing HN",
    "hard_negative:legitimate_same_attempt_nonce_recurrence": "Same-attempt nonce HN",
    "hard_negative:fresh_nonce_cross_attempt": "Fresh cross-attempt nonce HN",
}
TIMING_RECALL = "anomaly_recall:anomaly:timestamp_inconsistency"
NONCE_RECALL = "anomaly_recall:anomaly:nonce_reuse"
TIMING_FPR = "normal_fpr:hard_negative:small_positive_causal_margin"


class FrozenArtifactError(RuntimeError):
    """Raised before derivation when frozen Phase-4 evidence does not match."""


@dataclass(frozen=True)
class FrozenEvidence:
    root: Path
    primary_metrics: list[dict[str, Any]]
    primary_summary: dict[str, Any]
    grouped_metrics: list[dict[str, Any]]
    grouped_table: list[dict[str, Any]]
    primary_prediction_count: int
    grouped_prediction_count: int
    source_hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def verify_and_load(phase4_root: Path) -> FrozenEvidence:
    """Verify the frozen evidence byte-for-byte, then load reporting inputs."""
    phase4_root = phase4_root.resolve()
    actual: dict[str, str] = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = phase4_root / relative
        if not path.is_file():
            raise FrozenArtifactError(f"missing frozen Phase-4 artifact: {path}")
        actual[relative] = sha256_file(path)
        if actual[relative] != expected:
            raise FrozenArtifactError(
                f"frozen Phase-4 hash mismatch for {relative}: "
                f"expected {expected}, found {actual[relative]}"
            )

    fingerprints = _read_json(phase4_root / "fingerprints.json")
    if fingerprints.get("artifact_tree_fingerprint") != PHASE4_TREE_FINGERPRINT:
        raise FrozenArtifactError(
            "frozen Phase-4 tree fingerprint mismatch: "
            f"expected {PHASE4_TREE_FINGERPRINT}, "
            f"found {fingerprints.get('artifact_tree_fingerprint')!r}"
        )

    primary_metrics = _read_json(phase4_root / "primary/per-run-metrics.json")
    grouped_metrics = _read_json(phase4_root / "grouped/per-run-metrics.json")
    primary_count = _jsonl_count(phase4_root / "primary/predictions.jsonl")
    grouped_count = _jsonl_count(phase4_root / "grouped/predictions.jsonl")
    if (len(primary_metrics), primary_count) != (45, 3780):
        raise FrozenArtifactError(
            f"primary execution-count mismatch: {len(primary_metrics)} runs, "
            f"{primary_count} predictions"
        )
    if (len(grouped_metrics), grouped_count) != (9, 819):
        raise FrozenArtifactError(
            f"grouped execution-count mismatch: {len(grouped_metrics)} runs, "
            f"{grouped_count} predictions"
        )

    expected_primary = {(model, condition, repetition) for model in MODELS
                        for condition in CONDITIONS for repetition in range(1, 6)}
    actual_primary = {(row["model"], row["condition"], row["repetition"])
                      for row in primary_metrics}
    if actual_primary != expected_primary:
        raise FrozenArtifactError("primary model/condition/repetition grid is incomplete")
    expected_grouped = {(model, condition) for model in MODELS for condition in CONDITIONS}
    actual_grouped = {(row["model"], row["condition"]) for row in grouped_metrics}
    if actual_grouped != expected_grouped:
        raise FrozenArtifactError("grouped model/condition grid is incomplete")

    return FrozenEvidence(
        root=phase4_root,
        primary_metrics=primary_metrics,
        primary_summary=_read_json(phase4_root / "primary/aggregate-summary.json"),
        grouped_metrics=grouped_metrics,
        grouped_table=_read_json(phase4_root / "grouped/grouped-sensitivity-table.json"),
        primary_prediction_count=primary_count,
        grouped_prediction_count=grouped_count,
        source_hashes=actual,
    )


def _summary(evidence: FrozenEvidence, model: str, condition: str, metric: str) -> dict[str, Any]:
    return evidence.primary_summary["by_model_condition"][model][condition][metric]


def _metric(row: Mapping[str, Any], metric: str) -> float:
    metrics = row["metrics"]
    if metric == "anomaly_f1" or metric == "balanced_accuracy":
        return float(metrics[metric])
    prefix, role = metric.split(":", 1)
    if prefix == "normal_fpr":
        return float(metrics["normal_role_diagnostics"][role]["false_positive_rate"])
    if prefix == "anomaly_recall":
        return float(metrics["anomaly_role_diagnostics"][role]["recall"])
    raise KeyError(metric)


def _metric_rows(evidence: FrozenEvidence, model: str, condition: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in evidence.primary_metrics
         if row["model"] == model and row["condition"] == condition),
        key=lambda row: row["repetition"],
    )


def _fmt_summary(summary: Mapping[str, float]) -> str:
    return f'{summary["median"]:.3f} ({summary["iqr"]:.3f})'


def build_table_a(evidence: FrozenEvidence) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            f1 = _summary(evidence, model, condition, "anomaly_f1")
            ba = _summary(evidence, model, condition, "balanced_accuracy")
            timing = _summary(evidence, model, condition, TIMING_RECALL)
            nonce = _summary(evidence, model, condition, NONCE_RECALL)
            rows.append({
                "model": model, "condition": condition,
                "anomaly_f1_median": f1["median"], "anomaly_f1_iqr": f1["iqr"],
                "anomaly_f1_median_iqr": _fmt_summary(f1),
                "balanced_accuracy_median": ba["median"],
                "balanced_accuracy_iqr": ba["iqr"],
                "balanced_accuracy_median_iqr": _fmt_summary(ba),
                "timing_recall_median": timing["median"],
                "timing_recall_iqr": timing["iqr"],
                "nonce_recall_median": nonce["median"],
                "nonce_recall_iqr": nonce["iqr"],
            })
    return rows


def build_table_b(evidence: FrozenEvidence) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        row: dict[str, Any] = {"model": model}
        for condition in ("C0", "C1"):
            fpr = _summary(evidence, model, condition, TIMING_FPR)
            recall = _summary(evidence, model, condition, TIMING_RECALL)
            row.update({
                f"{condition.lower()}_timing_hn_fpr_median": fpr["median"],
                f"{condition.lower()}_timing_hn_fpr_iqr": fpr["iqr"],
                f"{condition.lower()}_timing_hn_fpr_range": f'{fpr["minimum"]:.3f}–{fpr["maximum"]:.3f}',
                f"{condition.lower()}_timing_recall_median": recall["median"],
                f"{condition.lower()}_timing_recall_iqr": recall["iqr"],
                f"{condition.lower()}_timing_recall_range": f'{recall["minimum"]:.3f}–{recall["maximum"]:.3f}',
            })
        rows.append(row)
    return rows


def build_table_c(evidence: FrozenEvidence) -> list[dict[str, Any]]:
    rows = []
    metrics = {
        "timing_hn_fpr": TIMING_FPR,
        "timing_recall": TIMING_RECALL,
        "anomaly_f1": "anomaly_f1",
        "balanced_accuracy": "balanced_accuracy",
    }
    for model in MODELS:
        row: dict[str, Any] = {"model": model}
        for condition in ("C1", "C2"):
            for label, metric in metrics.items():
                summary = _summary(evidence, model, condition, metric)
                row[f"{condition.lower()}_{label}_median"] = summary["median"]
                row[f"{condition.lower()}_{label}_iqr"] = summary["iqr"]
        rows.append(row)
    return rows


def build_table_d(evidence: FrozenEvidence) -> list[dict[str, Any]]:
    rows = []
    by_key = {(row["model"], row["condition"]): row for row in evidence.grouped_table}
    for model in MODELS:
        for condition in CONDITIONS:
            source = by_key[(model, condition)]
            rows.append({
                "model": model,
                "condition": condition,
                "anomaly_f1": source["anomaly_f1"],
                "balanced_accuracy": source["balanced_accuracy"],
                "timing_hn_fp": source["timing_hn_fp"],
                "timing_hn_count": source["timing_hn_count"],
                "timing_hn_fpr": source["timing_hn_fpr"],
                "timing_anomaly_tp": source["timing_anomaly_tp"],
                "timing_anomaly_count": source["timing_anomaly_count"],
                "timing_anomaly_recall": source["timing_anomaly_recall"],
                "nonce_anomaly_tp": source["nonce_anomaly_tp"],
                "nonce_anomaly_count": source["nonce_anomaly_count"],
                "nonce_anomaly_recall": source["nonce_anomaly_recall"],
            })
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot export empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _escape_latex(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def _write_latex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(rows[0])
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\toprule"]
    lines.append(" & ".join(_escape_latex(column) for column in columns) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.3f}" if isinstance(value, float) else _escape_latex(value))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
    })


def _save_figure(fig: Any, stem: Path) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{suffix}")
        metadata = {"Creator": "ICDS Phase-4 frozen paper-artifact layer"}
        if suffix == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        elif suffix == "svg":
            metadata = {"Date": None, "Creator": metadata["Creator"]}
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
        outputs.append(path)
    return outputs


def figure_hard_negative_effect(evidence: FrozenEvidence, stem: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    _plot_style()
    colors = {"C0": "#D55E00", "C1": "#0072B2"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        for role_index, role in enumerate(NORMAL_ROLES):
            metric = f"normal_fpr:{role}"
            paired = []
            for condition in ("C0", "C1"):
                values = [_metric(row, metric) for row in _metric_rows(evidence, model, condition)]
                paired.append(values)
                x = role_index + (-0.13 if condition == "C0" else 0.13)
                offsets = np.linspace(-0.035, 0.035, len(values))
                ax.scatter(x + offsets, values, s=18, color=colors[condition], alpha=0.65,
                           zorder=3, label=condition if role_index == 0 else None)
                summary = _summary(evidence, model, condition, metric)
                ax.errorbar(x, summary["median"],
                            yerr=[[summary["median"] - summary["q1"]],
                                  [summary["q3"] - summary["median"]]],
                            fmt="D", ms=5, color=colors[condition], capsize=3, zorder=4)
            for repetition in range(5):
                ax.plot([role_index - 0.13, role_index + 0.13],
                        [paired[0][repetition], paired[1][repetition]],
                        color="#777777", alpha=0.24, linewidth=0.7, zorder=1)
        ax.set_title(model)
        ax.set_xticks(range(len(NORMAL_ROLES)), [ROLE_LABELS[r] for r in NORMAL_ROLES],
                      rotation=28, ha="right")
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("Normal semantic subgroup")
    axes[0].set_ylabel("False-positive rate (lower is better)")
    axes[-1].legend(title="Training condition", frameon=False, loc="upper right")
    fig.suptitle("Semantic hard-negative effect: C0 versus C1\n"
                 "Points are five frozen repetitions; diamonds show median and IQR", y=1.03)
    fig.tight_layout()
    paths = _save_figure(fig, stem)
    plt.close(fig)
    return paths


def figure_semantic_error_matrix(evidence: FrozenEvidence, stem: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    _plot_style()
    fpr_metrics = [
        ("Ordinary\nFPR", "normal_fpr:ordinary_normal"),
        ("Timing HN\nFPR", TIMING_FPR),
        ("Same-attempt\nnonce HN FPR", "normal_fpr:hard_negative:legitimate_same_attempt_nonce_recurrence"),
        ("Fresh cross-attempt\nnonce HN FPR", "normal_fpr:hard_negative:fresh_nonce_cross_attempt"),
    ]
    recall_metrics = [("Timing anomaly\nrecall", TIMING_RECALL),
                      ("Nonce anomaly\nrecall", NONCE_RECALL)]
    rows = [(model, condition) for model in MODELS for condition in CONDITIONS]
    fig, (ax_fpr, ax_recall) = plt.subplots(
        1, 2, figsize=(10.8, 5.1), sharey=True,
        gridspec_kw={"width_ratios": [4, 2], "wspace": 0.05},
    )
    for ax, metrics, cmap, title in (
        (ax_fpr, fpr_metrics, "Oranges", "False-positive rates — lower is better ↓"),
        (ax_recall, recall_metrics, "Blues", "Anomaly recall — higher is better ↑"),
    ):
        matrix = np.array([[_summary(evidence, m, c, metric)["median"]
                            for _, metric in metrics] for m, c in rows])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(metrics)), [label for label, _ in metrics])
        ax.tick_params(axis="x", labelrotation=25)
        ax.set_title(title)
        ax.grid(False)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center",
                        color="white" if matrix[y, x] > 0.58 else "#222222", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    ax_fpr.set_yticks(range(len(rows)), [f"{m}–{c}" for m, c in rows])
    ax_recall.tick_params(axis="y", left=False)
    fig.suptitle("Semantic error profile (median across five frozen repetitions)", y=1.01)
    paths = _save_figure(fig, stem)
    plt.close(fig)
    return paths


def figure_representation_sensitivity(evidence: FrozenEvidence, stem: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    _plot_style()
    metrics = [
        ("Timing HN FPR\n↓", TIMING_FPR),
        ("Timing recall\n↑", TIMING_RECALL),
        ("Anomaly F1\n↑", "anomaly_f1"),
        ("Balanced accuracy\n↑", "balanced_accuracy"),
    ]
    colors = {"C1": "#0072B2", "C2": "#CC79A7"}
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1), sharey=True)
    for ax, model in zip(axes, MODELS):
        for index, (_, metric) in enumerate(metrics):
            values = {}
            for condition in ("C1", "C2"):
                values[condition] = [_metric(row, metric)
                                     for row in _metric_rows(evidence, model, condition)]
                x = index + (-0.13 if condition == "C1" else 0.13)
                offsets = np.linspace(-0.035, 0.035, 5)
                ax.scatter(x + offsets, values[condition], s=18, color=colors[condition],
                           alpha=0.7, zorder=3, label=condition if index == 0 else None)
                summary = _summary(evidence, model, condition, metric)
                ax.errorbar(x, summary["median"],
                            yerr=[[summary["median"] - summary["q1"]],
                                  [summary["q3"] - summary["median"]]],
                            fmt="D", ms=5, color=colors[condition], capsize=3, zorder=4)
            for repetition in range(5):
                ax.plot([index - 0.13, index + 0.13],
                        [values["C1"][repetition], values["C2"][repetition]],
                        color="#777777", alpha=0.24, linewidth=0.7)
        ax.set_title(model)
        ax.set_xticks(range(len(metrics)), [label for label, _ in metrics], rotation=22, ha="right")
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("Frozen metric")
    axes[0].set_ylabel("Rate / score")
    axes[-1].legend(title="Representation", frameon=False)
    fig.suptitle("C1 → C2 representation sensitivity\n"
                 "Five paired repetitions; diamonds show median and IQR", y=1.03)
    fig.tight_layout()
    paths = _save_figure(fig, stem)
    plt.close(fig)
    return paths


def figure_grouped_sensitivity(evidence: FrozenEvidence, stem: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    _plot_style()
    rows = {(row["model"], row["condition"]): row for row in evidence.grouped_table}
    outcomes = [
        ("Timing HN\nfalse positives", "timing_hn_fp", "timing_hn_count"),
        ("Timing anomaly\ntrue positives", "timing_anomaly_tp", "timing_anomaly_count"),
        ("Nonce anomaly\ntrue positives", "nonce_anomaly_tp", "nonce_anomaly_count"),
    ]
    colors = {"C0": "#D55E00", "C1": "#0072B2", "C2": "#CC79A7"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    for ax, model in zip(axes, MODELS):
        for condition_index, condition in enumerate(CONDITIONS):
            xs = np.arange(len(outcomes)) + (condition_index - 1) * 0.22
            values = []
            for _, numerator, denominator in outcomes:
                row = rows[(model, condition)]
                values.append(row[numerator] / row[denominator])
            bars = ax.bar(xs, values, width=0.20, color=colors[condition], alpha=0.85,
                          label=condition)
            for bar, (_, numerator, denominator) in zip(bars, outcomes):
                row = rows[(model, condition)]
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                        f'{row[numerator]}/{row[denominator]}', ha="center", va="bottom",
                        fontsize=7, rotation=90)
        ax.set_title(model)
        ax.set_xticks(range(len(outcomes)), [label for label, _, _ in outcomes])
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("Frozen semantic outcome")
    axes[0].set_ylabel("Observed proportion (raw count shown)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Condition", frameon=False,
               loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=3)
    fig.suptitle("Device-disjoint sensitivity — one controlled grouped partition", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    paths = _save_figure(fig, stem)
    plt.close(fig)
    return paths


def _write_manifest(output_root: Path, evidence: FrozenEvidence, artifacts: Iterable[Path]) -> Path:
    purposes = {
        "figure-1-primary-hard-negative-effect": "C0/C1 normal-subgroup false-positive behavior with repetitions",
        "figure-2-semantic-error-profile": "direction-aware semantic FPR and anomaly-recall matrix",
        "figure-3-representation-sensitivity": "C1/C2 representation sensitivity with paired repetitions",
        "figure-4-device-grouped-sensitivity": "one controlled device-disjoint partition with raw counts",
        "table-a-primary-performance": "primary median/IQR performance summary",
        "table-b-semantic-boundary": "C0/C1 Timing boundary behavior",
        "table-c-representation-sensitivity": "C1/C2 Timing and aggregate metric comparison",
        "table-d-device-grouped-sensitivity": "device-grouped sensitivity with raw counts",
    }
    entries = []
    for path in sorted(artifacts, key=lambda item: item.name):
        stem = path.stem
        entries.append({
            "filename": path.name,
            "artifact_type": path.suffix.lstrip("."),
            "sha256": sha256_file(path),
            "source_frozen_files": sorted(evidence.source_hashes),
            "source_hashes": evidence.source_hashes,
            "generation_notebook": "notebooks/icds_phase4_paper_artifacts.ipynb",
            "generation_script": "IoT_auth_simulator/ml/icds/paper_artifacts.py",
            "scientific_condition": (
                "C0/C1" if stem in {"figure-1-primary-hard-negative-effect", "table-b-semantic-boundary"}
                else "C1/C2" if stem in {"figure-3-representation-sensitivity", "table-c-representation-sensitivity"}
                else "C0/C1/C2"
            ),
            "purpose": purposes[stem],
        })
    manifest = {
        "schema_version": "icds-paper-artifact-manifest-v1",
        "source_phase4_tree_fingerprint": PHASE4_TREE_FINGERPRINT,
        "source_execution_counts": {
            "primary_runs": len(evidence.primary_metrics),
            "primary_predictions": evidence.primary_prediction_count,
            "grouped_runs": len(evidence.grouped_metrics),
            "grouped_predictions": evidence.grouped_prediction_count,
        },
        "generation_policy": "deterministic read-only transformation; no model fitting",
        "generated_at_utc": None,
        "artifacts": entries,
    }
    path = output_root / "paper-artifact-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_paper_artifacts(repository_root: Path) -> dict[str, Any]:
    """Verify frozen inputs and generate the complete paper-artifact layer."""
    repository_root = repository_root.resolve()
    evidence = verify_and_load(repository_root / "artifacts/icds-ml-phase4")
    output_root = repository_root / "artifacts/icds-paper-figures"
    output_root.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []
    artifacts += figure_hard_negative_effect(
        evidence, output_root / "figure-1-primary-hard-negative-effect")
    artifacts += figure_semantic_error_matrix(
        evidence, output_root / "figure-2-semantic-error-profile")
    artifacts += figure_representation_sensitivity(
        evidence, output_root / "figure-3-representation-sensitivity")
    artifacts += figure_grouped_sensitivity(
        evidence, output_root / "figure-4-device-grouped-sensitivity")

    tables = {
        "table-a-primary-performance": build_table_a(evidence),
        "table-b-semantic-boundary": build_table_b(evidence),
        "table-c-representation-sensitivity": build_table_c(evidence),
        "table-d-device-grouped-sensitivity": build_table_d(evidence),
    }
    for name, rows in tables.items():
        csv_path = output_root / f"{name}.csv"
        tex_path = output_root / f"{name}.tex"
        _write_csv(csv_path, rows)
        _write_latex(tex_path, rows)
        artifacts.extend([csv_path, tex_path])

    manifest = _write_manifest(output_root, evidence, artifacts)
    return {
        "output_root": output_root.relative_to(repository_root).as_posix(),
        "manifest": manifest.relative_to(repository_root).as_posix(),
        "figure_count": 4,
        "figure_export_count": 12,
        "table_count": 4,
        "table_export_count": 8,
        "primary_runs": len(evidence.primary_metrics),
        "primary_predictions": evidence.primary_prediction_count,
        "grouped_runs": len(evidence.grouped_metrics),
        "grouped_predictions": evidence.grouped_prediction_count,
        "phase4_tree_fingerprint": PHASE4_TREE_FINGERPRINT,
    }
