#!/usr/bin/env python3
"""Generate manuscript Figures 2 and 3 from frozen prediction artifacts.

The script is intentionally read-only with respect to all frozen experiment
artifacts.  It reconstructs subgroup metrics from persisted test predictions,
checks them against persisted run-level and aggregate records, and writes only
derived publication outputs under ``figures/`` (or ``--output-dir``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np


TIMING_HN = "hard_negative:small_positive_causal_margin"
TIMING_ANOMALY = "anomaly:timestamp_inconsistency"
MODELS = ("LR", "DT", "RF")

NAVY = "#274C8E"
TEAL = "#1B9E8A"
SLATE = "#667785"
GRAY = "#8B8E91"
LINE_GRAY = "#59636C"
GRID = "#D8DCE0"

INPUTS = (
    "artifacts/icds-ml-phase4/primary/predictions.jsonl",
    "artifacts/icds-ml-phase4/primary/per-run-metrics.json",
    "artifacts/icds-ml-phase4/primary/aggregate-summary.json",
    "artifacts/r4.1-device-disjoint-evaluation/predictions.jsonl",
    "artifacts/r4.1-device-disjoint-evaluation/run-level-metrics.json",
    "artifacts/r4.1-device-disjoint-evaluation/aggregate-summary.json",
    "artifacts/r4.1-device-disjoint-evaluation/paired-comparisons.json",
    "artifacts/r4.1-device-disjoint-evaluation/fingerprints.json",
    "artifacts/r4.2-representation-sensitivity/predictions.jsonl",
    "artifacts/r4.2-representation-sensitivity/run-level-metrics.json",
    "artifacts/r4.2-representation-sensitivity/aggregate-summary.json",
    "artifacts/r4.2-representation-sensitivity/paired-comparisons.json",
    "artifacts/r4.2-representation-sensitivity/feature-contracts.json",
    "artifacts/r4.2-representation-sensitivity/fingerprints.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def grouped_by_run(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "test":
            raise AssertionError(f"Unexpected non-test prediction: {row['run_id']}")
        grouped[row["run_id"]].append(row)
    return dict(grouped)


def subgroup_metric(rows: list[dict[str, Any]], role: str) -> float:
    selected = [row for row in rows if row["scientific_role"] == role]
    if not selected:
        raise AssertionError(f"No prediction rows for role {role}")
    expected_target = 0 if role == TIMING_HN else 1
    if {int(row["true_target"]) for row in selected} != {expected_target}:
        raise AssertionError(f"Unexpected target for role {role}")
    return float(np.mean([int(row["predicted_class"]) for row in selected]))


def reconstructed_runs(
    predictions: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = grouped_by_run(predictions)
    metrics_by_run = {row["run_id"]: row for row in metrics}
    if set(grouped) != set(metrics_by_run):
        raise AssertionError("Prediction and metric run IDs differ")

    output: list[dict[str, Any]] = []
    for run_id, rows in grouped.items():
        identity = {
            key: rows[0].get(key)
            for key in ("condition", "model", "repetition", "partition_id")
            if key in rows[0]
        }
        for key, value in identity.items():
            if {row.get(key) for row in rows} != {value}:
                raise AssertionError(f"Inconsistent {key} within run {run_id}")
        fpr = subgroup_metric(rows, TIMING_HN)
        recall = subgroup_metric(rows, TIMING_ANOMALY)
        frozen = metrics_by_run[run_id]["metrics"]
        frozen_fpr = frozen["normal_role_diagnostics"][TIMING_HN]["false_positive_rate"]
        frozen_recall = frozen["anomaly_role_diagnostics"][TIMING_ANOMALY]["recall"]
        if not np.isclose(fpr, frozen_fpr) or not np.isclose(recall, frozen_recall):
            raise AssertionError(f"Prediction reconstruction disagrees for {run_id}")
        output.append(
            {
                **identity,
                "run_id": run_id,
                "timing_hn_fpr": fpr,
                "timing_anomaly_recall": recall,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row.get("condition")),
            str(row.get("model")),
            int(row.get("repetition") or 0),
            str(row.get("partition_id") or ""),
        ),
    )


def select_one(
    rows: list[dict[str, Any]], **criteria: Any
) -> dict[str, Any]:
    selected = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row for {criteria}, found {len(selected)}")
    return selected[0]


def values(
    rows: list[dict[str, Any]], metric: str, **criteria: Any
) -> list[float]:
    selected = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    return [float(row[metric]) for row in selected]


def distribution(items: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(items), dtype=float)
    if array.size == 0:
        raise AssertionError("Cannot summarize an empty distribution")
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75], method="linear")
    return {"q1": float(q1), "median": float(median), "q3": float(q3)}


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=1e-12, rtol=0.0):
        raise AssertionError(f"{label}: reconstructed {actual}, expected {expected}")


def verify_aggregate_values(
    baseline: list[dict[str, Any]],
    r41: list[dict[str, Any]],
    r42: list[dict[str, Any]],
    baseline_aggregate: dict[str, Any],
    r41_aggregate: dict[str, Any],
    r42_aggregate: dict[str, Any],
) -> dict[str, Any]:
    checks = 0
    baseline_expected = {
        ("LR", "C0", "timing_hn_fpr"): 1.0,
        ("LR", "C1", "timing_hn_fpr"): 0.75,
        ("DT", "C0", "timing_hn_fpr"): 1.0,
        ("DT", "C1", "timing_hn_fpr"): 0.0,
        ("RF", "C0", "timing_hn_fpr"): 0.75,
        ("RF", "C1", "timing_hn_fpr"): 0.0,
        ("LR", "C0", "timing_anomaly_recall"): 1.0,
        ("LR", "C1", "timing_anomaly_recall"): 11.0 / 12.0,
        ("DT", "C0", "timing_anomaly_recall"): 1.0,
        ("DT", "C1", "timing_anomaly_recall"): 1.0,
        ("RF", "C0", "timing_anomaly_recall"): 1.0,
        ("RF", "C1", "timing_anomaly_recall"): 1.0,
    }
    aggregate_metric_names = {
        "timing_hn_fpr": f"normal_fpr:{TIMING_HN}",
        "timing_anomaly_recall": f"anomaly_recall:{TIMING_ANOMALY}",
    }
    for (model, condition, metric), expected in baseline_expected.items():
        reconstructed = distribution(values(baseline, metric, model=model, condition=condition))["median"]
        persisted = baseline_aggregate["by_model_condition"][model][condition][aggregate_metric_names[metric]]["median"]
        assert_close(reconstructed, persisted, f"baseline aggregate {model}/{condition}/{metric}")
        assert_close(reconstructed, expected, f"baseline expected {model}/{condition}/{metric}")
        checks += 2

    paired_r41: dict[str, Any] = {}
    r41_expected_delta = {"LR": -1.0 / 6.0, "DT": -1.0, "RF": -2.0 / 3.0}
    r41_expected_recall_losses = {"LR": 14, "DT": 2, "RF": 1}
    for model in MODELS:
        partitions = sorted({row["partition_id"] for row in r41 if row["model"] == model})
        deltas_fpr: list[float] = []
        deltas_recall: list[float] = []
        for partition in partitions:
            ordinary = select_one(r41, model=model, condition="C0", partition_id=partition)
            boundary = select_one(r41, model=model, condition="C1", partition_id=partition)
            deltas_fpr.append(boundary["timing_hn_fpr"] - ordinary["timing_hn_fpr"])
            deltas_recall.append(boundary["timing_anomaly_recall"] - ordinary["timing_anomaly_recall"])
        median_delta = float(np.median(deltas_fpr))
        loss_count = sum(delta < 0 for delta in deltas_recall)
        if not all(delta < 0 for delta in deltas_fpr):
            raise AssertionError(f"Not all R4.1 FPR deltas are negative for {model}")
        assert_close(median_delta, r41_expected_delta[model], f"R4.1 median delta {model}")
        if loss_count != r41_expected_recall_losses[model]:
            raise AssertionError(f"R4.1 recall-loss count {model}: {loss_count}")
        persisted = r41_aggregate["paired_C0_to_C1"][model]
        assert_close(median_delta, persisted["delta_fpr"]["median"], f"R4.1 aggregate delta {model}")
        if loss_count != sum(value < 0 for value in persisted["delta_recall"]["values"]):
            raise AssertionError(f"R4.1 aggregate recall-loss mismatch for {model}")
        paired_r41[model] = {
            "median_delta_timing_hn_fpr": median_delta,
            "recall_loss_partitions": loss_count,
            "partition_count": len(partitions),
            "all_fpr_deltas_negative": True,
        }
        checks += 4

    expected_ac = {
        "LR": (-0.25, 8),
        "DT": (0.0, 1),
        "RF": (0.0, 15),
    }
    expected_cd = {
        "LR": (-5.0 / 12.0, -19.0 / 24.0),
        "DT": (1.0 / 6.0, -19.0 / 24.0),
        "RF": (1.0 / 12.0, -0.75),
    }
    paired_r42: dict[str, Any] = {}
    for model in MODELS:
        partitions = sorted({row["partition_id"] for row in r42 if row["model"] == model})
        ac_fpr: list[float] = []
        ac_recall: list[float] = []
        cd_fpr: list[float] = []
        cd_recall: list[float] = []
        for partition in partitions:
            full = select_one(r41, model=model, condition="C1", partition_id=partition)
            no_direct = select_one(r42, model=model, condition="C", partition_id=partition)
            no_generic = select_one(r42, model=model, condition="D", partition_id=partition)
            ac_fpr.append(no_direct["timing_hn_fpr"] - full["timing_hn_fpr"])
            ac_recall.append(no_direct["timing_anomaly_recall"] - full["timing_anomaly_recall"])
            cd_fpr.append(no_generic["timing_hn_fpr"] - no_direct["timing_hn_fpr"])
            cd_recall.append(no_generic["timing_anomaly_recall"] - no_direct["timing_anomaly_recall"])
        ac_fpr_median = float(np.median(ac_fpr))
        ac_loss_count = sum(delta < 0 for delta in ac_recall)
        cd_fpr_median = float(np.median(cd_fpr))
        cd_recall_median = float(np.median(cd_recall))
        cd_loss_count = sum(delta < 0 for delta in cd_recall)
        assert_close(ac_fpr_median, expected_ac[model][0], f"R4.2 A→C FPR {model}")
        if ac_loss_count != expected_ac[model][1]:
            raise AssertionError(f"R4.2 A→C recall-loss count {model}: {ac_loss_count}")
        assert_close(cd_fpr_median, expected_cd[model][0], f"R4.2 C→D FPR {model}")
        assert_close(cd_recall_median, expected_cd[model][1], f"R4.2 C→D recall {model}")
        if cd_loss_count != 20:
            raise AssertionError(f"R4.2 C→D recall does not fall in all partitions for {model}")
        persisted = r42_aggregate["paired_comparisons"][model]
        assert_close(ac_fpr_median, persisted["A_to_C"]["delta_fpr"]["median"], f"R4.2 aggregate A→C {model}")
        assert_close(cd_fpr_median, persisted["C_to_D"]["delta_fpr"]["median"], f"R4.2 aggregate C→D FPR {model}")
        assert_close(cd_recall_median, persisted["C_to_D"]["delta_recall"]["median"], f"R4.2 aggregate C→D recall {model}")
        paired_r42[model] = {
            "full_to_no_direct": {
                "median_delta_timing_hn_fpr": ac_fpr_median,
                "recall_loss_partitions": ac_loss_count,
            },
            "no_direct_to_no_direct_plus_generic": {
                "median_delta_timing_hn_fpr": cd_fpr_median,
                "median_delta_timing_anomaly_recall": cd_recall_median,
                "recall_loss_partitions": cd_loss_count,
            },
        }
        checks += 8

    return {
        "check_count": checks,
        "baseline_expected_medians_verified": True,
        "r4_1": paired_r41,
        "r4_2": paired_r42,
    }


def verify_paired_records(
    r41: list[dict[str, Any]],
    r42: list[dict[str, Any]],
    r41_paired: list[dict[str, Any]],
    r42_paired: list[dict[str, Any]],
) -> int:
    checks = 0
    r41_index = {(row["partition_id"], row["model"]): row for row in r41_paired}
    r42_index = {
        (row["contrast"], row["partition_id"], row["model"]): row
        for row in r42_paired
    }
    for model in MODELS:
        partitions = sorted({row["partition_id"] for row in r42 if row["model"] == model})
        for partition in partitions:
            a = select_one(r41, model=model, condition="C1", partition_id=partition)
            ordinary = select_one(r41, model=model, condition="C0", partition_id=partition)
            c = select_one(r42, model=model, condition="C", partition_id=partition)
            d = select_one(r42, model=model, condition="D", partition_id=partition)
            frozen_r41 = r41_index[(partition, model)]
            assert_close(
                a["timing_hn_fpr"] - ordinary["timing_hn_fpr"],
                frozen_r41["delta_fpr_C1_minus_C0"],
                f"R4.1 paired record {partition}/{model}",
            )
            for contrast, source, target in (("A_to_C", a, c), ("C_to_D", c, d)):
                frozen = r42_index[(contrast, partition, model)]
                assert_close(
                    target["timing_hn_fpr"] - source["timing_hn_fpr"],
                    frozen["delta_fpr"],
                    f"R4.2 paired FPR {contrast}/{partition}/{model}",
                )
                assert_close(
                    target["timing_anomaly_recall"] - source["timing_anomaly_recall"],
                    frozen["delta_recall"],
                    f"R4.2 paired recall {contrast}/{partition}/{model}",
                )
            checks += 5
    return checks


def style_axes(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_ylim(-0.015, 1.025)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(width=0.7, length=3, labelsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)


def plot_distribution(
    ax: plt.Axes,
    x: float,
    items: list[float],
    color: str,
    jitter_width: float,
) -> None:
    offsets = np.linspace(-jitter_width, jitter_width, len(items))
    ax.scatter(
        x + offsets,
        items,
        s=14,
        color=color,
        alpha=0.52,
        edgecolors="none",
        zorder=3,
    )
    summary = distribution(items)
    ax.errorbar(
        [x],
        [summary["median"]],
        yerr=[[summary["median"] - summary["q1"]], [summary["q3"] - summary["median"]]],
        color=color,
        linewidth=1.25,
        capsize=3,
        zorder=4,
    )
    ax.scatter(
        [x],
        [summary["median"]],
        marker="D",
        s=38,
        color=color,
        edgecolors="white",
        linewidths=0.45,
        zorder=5,
    )


def generate_figure2(baseline: list[dict[str, Any]], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), sharey=True)
    panels = (
        ("timing_hn_fpr", "Timing boundary FPR"),
        ("timing_anomaly_recall", "Timing-anomaly recall"),
    )
    conditions = (("C0", "Ordinary", NAVY, -0.15), ("C1", "Boundary", TEAL, 0.15))
    for panel_index, (ax, (metric, title)) in enumerate(zip(axes, panels)):
        style_axes(ax, "Rate" if panel_index == 0 else None)
        for model_index, model in enumerate(MODELS):
            for condition, _label, color, offset in conditions:
                vals = values(baseline, metric, model=model, condition=condition)
                if len(vals) != 5:
                    raise AssertionError(f"Expected five baseline repetitions for {model}/{condition}")
                plot_distribution(ax, model_index + offset, vals, color, 0.035)
        ax.set_xlim(-0.55, 2.55)
        ax.set_xticks(range(3), MODELS)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=6)
        ax.text(-0.12, 1.075, f"({chr(97 + panel_index)})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, markersize=5.5, label=label)
        for _, label, color, _ in conditions
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.015),
               ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.6)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.79, wspace=0.16)
    fig.savefig(output_dir / "figure2_boundary_exposure.pdf",
                bbox_inches=Bbox.from_bounds(0.07827951388888901, 0.1730555555555557, 7.100220486111111, 2.662638888888888))
    fig.savefig(output_dir / "figure2_boundary_exposure.png", dpi=450,
                bbox_inches=Bbox.from_bounds(0.07719444444444445, 0.17305555555555566, 7.101305555555555, 2.6626388888888886))
    plt.close(fig)


def generate_figure3(
    r41: list[dict[str, Any]],
    r42: list[dict[str, Any]],
    verification: dict[str, Any],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.25, 4.15))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.35], wspace=0.25)
    ax_left = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(3, 1, hspace=0.12)
    right_axes = [fig.add_subplot(right[i, 0]) for i in range(3)]

    style_axes(ax_left, "Timing boundary FPR")
    for model_index, model in enumerate(MODELS):
        x0, x1 = model_index - 0.14, model_index + 0.14
        partitions = sorted({row["partition_id"] for row in r41 if row["model"] == model})
        ordinary_vals: list[float] = []
        boundary_vals: list[float] = []
        for partition in partitions:
            ordinary = select_one(r41, model=model, condition="C0", partition_id=partition)
            boundary = select_one(r41, model=model, condition="C1", partition_id=partition)
            y0, y1 = ordinary["timing_hn_fpr"], boundary["timing_hn_fpr"]
            ordinary_vals.append(y0)
            boundary_vals.append(y1)
            ax_left.plot([x0, x1], [y0, y1], color=LINE_GRAY, alpha=0.24,
                         linewidth=0.65, zorder=1)
            ax_left.scatter([x0], [y0], s=10, color=NAVY, alpha=0.40,
                            edgecolors="none", zorder=2)
            ax_left.scatter([x1], [y1], s=10, color=TEAL, alpha=0.40,
                            edgecolors="none", zorder=2)
        for x, vals, color in ((x0, ordinary_vals, NAVY), (x1, boundary_vals, TEAL)):
            summary = distribution(vals)
            ax_left.scatter([x], [summary["median"]], marker="D", s=45, color=color,
                            edgecolors="white", linewidths=0.45, zorder=5)
        # Compact tick text: model / median ΔFPR · recall-loss partitions.
    ax_left.set_xlim(-0.5, 2.5)
    ax_left.set_xticks(
        range(3),
        [
            f"{model}\n{verification['r4_1'][model]['median_delta_timing_hn_fpr']:+.3f} · "
            f"{verification['r4_1'][model]['recall_loss_partitions']}/20"
            for model in MODELS
        ],
        fontsize=6.6,
    )
    ax_left.set_xlabel("median ΔFPR · recall-loss partitions", fontsize=6.6, labelpad=5)
    ax_left.set_title("Boundary exposure", loc="left", fontsize=9,
                      fontweight="bold", y=1.18, pad=0)
    ax_left.text(-0.15, 1.18, "(a)", transform=ax_left.transAxes,
                 fontsize=9, fontweight="bold", va="bottom")
    ax_left.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color=NAVY, markersize=5, label="Ordinary"),
            Line2D([0], [0], marker="o", linestyle="none", color=TEAL, markersize=5, label="Boundary"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0,
        fontsize=7,
    )

    representation_specs = (
        ("Full", 90, "A", TEAL),
        ("Without Timing summary", 89, "C", SLATE),
        ("Without summary or timing statistics", 82, "D", GRAY),
    )
    for row_index, (ax, model) in enumerate(zip(right_axes, MODELS)):
        style_axes(ax, "Recall" if row_index == 1 else None)
        ax.set_yticks([0.0, 0.5, 1.0])
        partitions = sorted({row["partition_id"] for row in r42 if row["model"] == model})
        per_representation: list[list[float]] = [[], [], []]
        for partition in partitions:
            full = select_one(r41, model=model, condition="C1", partition_id=partition)
            no_direct = select_one(r42, model=model, condition="C", partition_id=partition)
            no_generic = select_one(r42, model=model, condition="D", partition_id=partition)
            vals = [
                full["timing_anomaly_recall"],
                no_direct["timing_anomaly_recall"],
                no_generic["timing_anomaly_recall"],
            ]
            for index, value in enumerate(vals):
                per_representation[index].append(value)
            ax.plot([0, 1, 2], vals, color=LINE_GRAY, alpha=0.20, linewidth=0.60, zorder=1)
            for x, value, (_, _, _, color) in zip(range(3), vals, representation_specs):
                ax.scatter([x], [value], s=9, color=color, alpha=0.36,
                           edgecolors="none", zorder=2)
        for x, (vals, (_, _, _, color)) in enumerate(zip(per_representation, representation_specs)):
            summary = distribution(vals)
            ax.errorbar(
                [x], [summary["median"]],
                yerr=[[summary["median"] - summary["q1"]], [summary["q3"] - summary["median"]]],
                color=color, linewidth=1.15, capsize=2.5, zorder=4,
            )
            ax.scatter([x], [summary["median"]], marker="D", s=36, color=color,
                       edgecolors="white", linewidths=0.4, zorder=5)
        ax.set_xlim(-0.2, 2.2)
        ax.text(0.012, 0.78, model, transform=ax.transAxes, fontsize=7.2,
                fontweight="bold", va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8})
        if row_index < 2:
            ax.set_xticks([])
        else:
            ax.set_xticks(
                [0, 1, 2],
                ["Full\n(90)", "Without Timing summary\n(89)",
                 "Without summary or\ntiming statistics (82)"],
                fontsize=6.5,
            )
    right_axes[0].set_title("Representation sensitivity", loc="left",
                            fontsize=9, fontweight="bold", y=1.55, pad=0)
    right_axes[0].text(-0.11, 1.55, "(b)", transform=right_axes[0].transAxes,
                       fontsize=9, fontweight="bold", va="bottom")
    right_axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color=color, markersize=5,
            label=label.replace(" or timing statistics", " or\ntiming statistics"))
            for label, count, _, color in representation_specs
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.25),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.6,
        borderaxespad=0,
        fontsize=6.2,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.14, top=0.79)
    fig.savefig(output_dir / "figure3_device_disjoint_representation.pdf",
                bbox_inches=Bbox.from_bounds(0.08677951388888888, 0.009559027777777881, 7.341891309501957, 3.979490972222223))
    fig.savefig(output_dir / "figure3_device_disjoint_representation.png", dpi=450,
                bbox_inches=Bbox.from_bounds(0.08569444444444438, 0.009575303819444533, 7.346218566446402, 3.9800595486111114))
    plt.close(fig)


def plot_data_csv(
    baseline: list[dict[str, Any]],
    r41: list[dict[str, Any]],
    r42: list[dict[str, Any]],
) -> str:
    rows: list[dict[str, Any]] = []
    for item in baseline:
        if item["condition"] not in {"C0", "C1"}:
            continue
        for metric in ("timing_hn_fpr", "timing_anomaly_recall"):
            rows.append(
                {
                    "figure": "Figure 2",
                    "panel": "a" if metric == "timing_hn_fpr" else "b",
                    "source": "frozen Phase 4 predictions",
                    "condition_or_representation": "Ordinary" if item["condition"] == "C0" else "Boundary",
                    "model": item["model"],
                    "repetition": item["repetition"],
                    "partition_id": "",
                    "metric": metric,
                    "value": item[metric],
                    "run_id": item["run_id"],
                }
            )
    for item in r41:
        if item["condition"] in {"C0", "C1"}:
            rows.append(
                {
                    "figure": "Figure 3",
                    "panel": "a",
                    "source": "frozen R4.1 predictions",
                    "condition_or_representation": "Ordinary" if item["condition"] == "C0" else "Boundary",
                    "model": item["model"],
                    "repetition": item["repetition"],
                    "partition_id": item["partition_id"],
                    "metric": "timing_hn_fpr",
                    "value": item["timing_hn_fpr"],
                    "run_id": item["run_id"],
                }
            )
        if item["condition"] == "C1":
            rows.append(
                {
                    "figure": "Figure 3",
                    "panel": "b",
                    "source": "frozen R4.1 anchor predictions",
                    "condition_or_representation": "Full (90)",
                    "model": item["model"],
                    "repetition": item["repetition"],
                    "partition_id": item["partition_id"],
                    "metric": "timing_anomaly_recall",
                    "value": item["timing_anomaly_recall"],
                    "run_id": item["run_id"],
                }
            )
    for item in r42:
        rows.append(
            {
                "figure": "Figure 3",
                "panel": "b",
                "source": "frozen R4.2 predictions",
                "condition_or_representation": "No direct Timing (89)" if item["condition"] == "C" else "No direct + generic (82)",
                "model": item["model"],
                "repetition": item["repetition"],
                "partition_id": item["partition_id"],
                "metric": "timing_anomaly_recall",
                "value": item["timing_anomaly_recall"],
                "run_id": item["run_id"],
            }
        )
    fieldnames = [
        "figure", "panel", "source", "condition_or_representation", "model",
        "repetition", "partition_id", "metric", "value", "run_id",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {relative: root / relative for relative in INPUTS}
    before_hashes = {relative: sha256(path) for relative, path in input_paths.items()}

    baseline_predictions = read_jsonl(input_paths[INPUTS[0]])
    baseline_metrics = read_json(input_paths[INPUTS[1]])
    baseline_aggregate = read_json(input_paths[INPUTS[2]])
    r41_predictions = read_jsonl(input_paths[INPUTS[3]])
    r41_metrics = read_json(input_paths[INPUTS[4]])
    r41_aggregate = read_json(input_paths[INPUTS[5]])
    r41_paired = read_json(input_paths[INPUTS[6]])
    r42_predictions = read_jsonl(input_paths[INPUTS[8]])
    r42_metrics = read_json(input_paths[INPUTS[9]])
    r42_aggregate = read_json(input_paths[INPUTS[10]])
    r42_paired = read_json(input_paths[INPUTS[11]])
    feature_contracts = read_json(input_paths[INPUTS[12]])

    baseline = reconstructed_runs(baseline_predictions, baseline_metrics)
    r41 = reconstructed_runs(r41_predictions, r41_metrics)
    r42 = reconstructed_runs(r42_predictions, r42_metrics)

    if len(baseline) != 45 or len(r41) != 180 or len(r42) != 120:
        raise AssertionError("Unexpected frozen run counts")
    feature_counts = {
        name: len(contract["ordered_features"])
        for name, contract in feature_contracts["conditions"].items()
    }
    if {name: feature_counts[name] for name in ("A", "B", "C", "D")} != {"A": 90, "B": 87, "C": 89, "D": 82}:
        raise AssertionError(f"Unexpected feature-contract counts: {feature_counts}")

    verification = verify_aggregate_values(
        baseline, r41, r42, baseline_aggregate, r41_aggregate, r42_aggregate
    )
    paired_check_count = verify_paired_records(r41, r42, r41_paired, r42_paired)
    verification["paired_record_check_count"] = paired_check_count

    data_path = output_dir / "figure2_figure3_plot_data.csv"
    expected_data = plot_data_csv(baseline, r41, r42)
    if not data_path.exists():
        data_path.write_bytes(expected_data.encode("utf-8"))
    saved_data = data_path.read_bytes()
    if saved_data != expected_data.encode("utf-8"):
        raise AssertionError("Saved plot data differs from verified experiment values or run identities")
    runs = {row["run_id"]: row for row in baseline + r41 + r42}
    for row in csv.DictReader(io.StringIO(saved_data.decode("utf-8"))):
        runs[row["run_id"]][row["metric"]] = float(row["value"])
    verification["saved_plot_data_matches_reconstruction"] = True
    generate_figure2(baseline, output_dir)
    generate_figure3(r41, r42, verification, output_dir)

    after_hashes = {relative: sha256(path) for relative, path in input_paths.items()}
    if before_hashes != after_hashes:
        raise AssertionError("At least one frozen input artifact changed during figure generation")

    outputs = [
        output_dir / "figure2_boundary_exposure.pdf",
        output_dir / "figure2_boundary_exposure.png",
        output_dir / "figure3_device_disjoint_representation.pdf",
        output_dir / "figure3_device_disjoint_representation.png",
        data_path,
    ]
    report = {
        "schema_version": "paper-figures-2-3-verification-v1",
        "source_policy": "metrics reconstructed from persisted test predictions and checked against frozen run-level and aggregate artifacts",
        "input_artifacts": [
            {"path": relative, "sha256": digest}
            for relative, digest in before_hashes.items()
        ],
        "input_artifacts_unchanged": True,
        "run_counts": {"phase4_baseline": len(baseline), "r4_1": len(r41), "r4_2": len(r42)},
        "feature_contract_counts": {name: feature_counts[name] for name in ("A", "B", "C", "D")},
        "verification": verification,
        "disagreements": [],
        "outputs": [
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
    }
    verification_path = output_dir / "figure_generation_verification.json"
    verification_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(output_dir),
        "aggregate_checks": verification["check_count"],
        "paired_record_checks": paired_check_count,
        "disagreements": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
