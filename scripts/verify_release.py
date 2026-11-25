#!/usr/bin/env python3
"""Verify publication evidence or reconstruct results from frozen inputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import runpy
import subprocess
import sys

# Running the verifier must not create bytecode caches in the checkout.
sys.dont_write_bytecode = True

from verify_artifacts import (
    ROOT, assert_equal, load_legacy_verifier, read_json, sha256_file,
    verify_dataset_identity, verify_inventory, verify_reference_manifests,
    verify_submission_snapshot,
)
from verify_r4_1_freeze import canonical_hash


PHASE4 = ROOT / "artifacts/icds-ml-phase4"
R41 = ROOT / "artifacts/r4.1-device-disjoint-evaluation"
R42 = ROOT / "artifacts/r4.2-representation-sensitivity"
PREEXEC = ROOT / "metadata/r4.2-preexecution"
FREEZE = ROOT / "metadata/r4.2-freeze"
FIGURES = ROOT / "figures"
FIGURE_MANIFEST_SHA = "4a62b208c6b932ffa50925c632f71b8c02c2fda5027fbb67b25b59b16ac405be"
FIGURE_SCRIPT_SHA = "5cab1309f772ccecf4c4a3e16c806d22664882499f72aaef3a33f0046ccd4f67"


def artifact_path(relative, base=ROOT):
    path = Path(relative)
    candidate = (base / path).resolve()
    if path.is_absolute() or not candidate.is_relative_to(base.resolve()) or not candidate.is_file():
        raise ValueError(f"unresolved repository artifact: {relative}")
    return candidate


def verify_file(base, relative, digest, size=None):
    path = artifact_path(relative, base)
    assert_equal(sha256_file(path), digest.removeprefix("sha256:"), relative)
    if size is not None:
        assert_equal(path.stat().st_size, size, f"{relative} bytes")
    return path


def verify_records(base, records):
    for record in records:
        verify_file(base, record["path"], record["sha256"], record.get("bytes"))
    return len(records)


def archive_phase4():
    from ml.evaluation.phase4_experiment import _artifact_hashes
    from ml.artifact_paths import resolve_scientific_path

    evidence = load_legacy_verifier().verify_frozen_artifacts()
    fingerprint = read_json(PHASE4 / "fingerprints.json")
    # This is the original Phase 4 tree enumeration and canonicalization.
    actual = _artifact_hashes(PHASE4)
    assert_equal(actual["artifact_tree_fingerprint"], fingerprint["artifact_tree_fingerprint"],
                 "Phase 4 reconstructed tree fingerprint")
    dataset = resolve_scientific_path("dataset-v1.0", ROOT)
    return {**evidence, "fingerprinted_files": len(fingerprint["files"]),
            "dataset_file_sha256": sha256_file(dataset),
            "dataset_candidate_fingerprint": read_json(dataset)["candidate_fingerprint"],
            "frozen_inventory": verify_inventory(), "dataset": verify_dataset_identity()}


def archive_r42(r41):
    from ml.artifact_paths import resolve_scientific_path

    freeze = read_json(FREEZE / "freeze-manifest.json")
    artifact_tree = read_json(R42 / "fingerprints.json")
    pre = read_json(PREEXEC / "durable-artifact-inventory.json")
    for record, field, expected in (
        (artifact_tree, "artifact_tree_fingerprint", freeze["final_aggregate_r4_2_fingerprint"]),
        (pre, "inventory_fingerprint", freeze["preexecution_fingerprint"]),
    ):
        assert_equal(record[field], expected, field)
        assert_equal(canonical_hash({k: v for k, v in record.items() if k != field}), expected, field)
    assert_equal(verify_records(ROOT, pre["artifacts"]), pre["artifact_count_excluding_inventory"],
                 "R4.2 pre-execution payload count")
    assert_equal(verify_records(R42, freeze["artifact_inventory"]), freeze["artifact_inventory_count"],
                 "R4.2 archive count")
    assert_equal(verify_records(R42, artifact_tree["files"]), artifact_tree["artifact_count"],
                 "R4.2 execution payload count")
    assert_equal({p.relative_to(R42).as_posix() for p in R42.rglob("*") if p.is_file()},
                 {r["path"] for r in freeze["artifact_inventory"]}, "R4.2 archive paths")

    upstream = freeze["upstream"]
    for field, actual in (
        ("r4_1_aggregate_fingerprint", r41["aggregate_r4_1_fingerprint"]),
        ("r4_1_artifact_tree_fingerprint", r41["execution_output_fingerprint"]),
        ("dataset_file_sha256", sha256_file(resolve_scientific_path("dataset-v1.0", ROOT))),
        ("historical_phase4_fingerprint", read_json(PHASE4 / "fingerprints.json")["artifact_tree_fingerprint"]),
    ):
        assert_equal(actual, upstream[field], f"R4.2 upstream {field}")
    for name, field in (("partition-manifest.json", "partition_manifest_sha256"),
                        ("accepted-partitions.csv", "accepted_partitions_sha256")):
        verify_file(ROOT / "metadata/r4.1-freeze/inputs", name, upstream[field])
    execution = freeze["execution"]
    verify_file(ROOT, execution["execution_implementation_path"], execution["execution_implementation_sha256"])
    validation = freeze["validation"]
    for name, digest in (("evidence_path", "evidence_sha256"),
                         ("execution_independent_validation_path", "execution_independent_validation_sha256")):
        report = read_json(verify_file(ROOT, validation[name], validation[digest]))
        assert_equal(report["status"], "PASS", validation[name])
        assert_equal(all(check["passed"] for check in report["checks"]), True, validation[name])
    anchors = read_json(R42 / "anchor-references.json")
    assert_equal(len(anchors["anchors"]), anchors["anchor_count"], "R4.2 anchor count")
    for anchor in anchors["anchors"]:
        for record in anchor["artifacts"]:
            verify_file(ROOT, record["repository_path"], record["sha256"], record["bytes"])
    plan = read_json(PREEXEC / "execution-manifest-cd.json")
    for key in ("anchors_path", "feature_contracts_path"):
        artifact_path(plan[key])
    contracts = read_json(artifact_path(plan["feature_contracts_path"]))["conditions"]
    for condition, record in freeze["feature_contract_fingerprints"].items():
        assert_equal({key: contracts[condition][key] for key in record}, record,
                     f"R4.2 {condition} frozen feature identity")
    assert_equal(read_json(R42 / "feature-contracts.json"),
                 read_json(artifact_path(plan["feature_contracts_path"])),
                 "R4.2 executed and pre-execution contracts")
    for run in plan["plans"]:
        assert_equal(run["feature_contract_reference"],
                     f"{plan['feature_contracts_path']}#/conditions/{run['condition']}",
                     "R4.2 feature contract reference")
        assert_equal(run["feature_contract_fingerprint"],
                     contracts[run["condition"]]["contract_fingerprint"], "R4.2 planned feature contract")
    return {
        "artifact_count": freeze["artifact_inventory_count"],
        "execution_payload_count": artifact_tree["artifact_count"],
        "preexecution_file_count": len(pre["artifacts"]) + 1,
        "reused_anchor_count": anchors["anchor_count"],
        "fingerprint": artifact_tree["artifact_tree_fingerprint"],
        "preexecution_fingerprint": pre["inventory_fingerprint"],
    }


def archive_figures():
    manifest_file = verify_file(ROOT, "figures/publication-figure-manifest.json", FIGURE_MANIFEST_SHA)
    manifest = read_json(manifest_file)
    for figure in manifest["figures"]:
        verify_records(ROOT, [figure])
        for relative in figure["scientific_artifact_references"]:
            artifact_path(relative)
        if figure["kind"] == "programmatically_generated":
            verify_file(ROOT, figure["source"], FIGURE_SCRIPT_SHA)
            verify_records(ROOT, [figure["preview"]])
    verify_records(ROOT, [manifest["plot_data"], manifest["generation_verification"]])
    verification = read_json(artifact_path(manifest["generation_verification"]["path"]))
    return {
        "figures": [{key: f[key] for key in ("number", "path", "sha256")} for f in manifest["figures"]],
        "manifest_sha256": FIGURE_MANIFEST_SHA,
        "generation_inputs": verify_records(ROOT, verification["input_artifacts"]),
        "generation_outputs": verify_records(ROOT, verification["outputs"]),
    }


def archive_r41(files_only):
    if not files_only:
        # The historical entry point is itself in the frozen R4.1 inventory.
        return json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/verify_r4_1_freeze.py")], text=True,
            stderr=subprocess.PIPE,
        ))
    manifest = read_json(ROOT / "metadata/r4.1-freeze/freeze-manifest.json")
    inventory = manifest["authoritative_inventory"]
    fingerprint = read_json(R41 / "fingerprints.json")
    count = verify_records(R41, fingerprint["files"])
    assert_equal(count, fingerprint["artifact_count"], "R4.1 execution output count")
    paths = {p.relative_to(R41).as_posix() for p in R41.rglob("*") if p.is_file()}
    assert_equal(paths, {"fingerprints.json", *(r["path"] for r in fingerprint["files"])},
                 "R4.1 output paths")
    aggregate = canonical_hash(inventory)
    output = canonical_hash(fingerprint["files"])
    assert_equal(aggregate, manifest["aggregate_r4_1_fingerprint"], "R4.1 aggregate fingerprint")
    assert_equal(output, manifest["execution_output_fingerprint"], "R4.1 freeze output fingerprint")
    assert_equal(output, fingerprint["artifact_tree_fingerprint"], "R4.1 output fingerprint")
    return {"inventory_count": verify_records(ROOT, inventory),
            "execution_output_artifact_count": count,
            "aggregate_r4_1_fingerprint": aggregate, "execution_output_fingerprint": output}


def archive(files_only):
    if not files_only:
        git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                                           text=True, stderr=subprocess.PIPE).strip()
        assert_equal(Path(git_root).resolve(), ROOT, "release Git checkout root")
    result = {
        "phase4_and_dataset": archive_phase4(),
        "reference_hashes_checked": verify_reference_manifests(),
        "r4_1": archive_r41(files_only),
        "figures": archive_figures(),
    }
    result["r4_2"] = archive_r42(result["r4_1"])
    if files_only:
        result["unsupported_checks"] = [
            "Historical annotated tag object and target",
            "Historical submission file hashes from Git objects",
        ]
        result["status"] = "PASS_FILE_CHECKS_ONLY"
    else:
        result["historical_snapshot"] = verify_submission_snapshot()
        result["status"] = "PASS"
    return result


def unique_index(rows, key):
    index = {row[key]: row for row in rows}
    assert_equal(len(index), len(rows), f"unique {key}")
    return index


def compare_records(actual, expected, label):
    """Bound mismatch diagnostics even for complete campaign summaries."""
    if actual != expected:
        if isinstance(actual, dict) and isinstance(expected, dict):
            assert_equal(sorted(actual), sorted(expected), f"{label} keys")
            for key, value in actual.items():
                compare_records(value, expected[key], f"{label}/{key}")
        elif isinstance(actual, list) and isinstance(expected, list):
            assert_equal(len(actual), len(expected), f"{label} length")
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                compare_records(left, right, f"{label}/{index}")
        else:
            assert_equal(actual, expected, label)


def rebuild_runs(root, count, predictions_per_run, metric_file, role_index,
                 *, plan=None, plan_key="run_id", extra_metrics=None):
    from ml.evaluation.metrics import reconstruct_metrics

    archived = unique_index(read_json(root / metric_file), "run_id")
    provenance = unique_index(read_json(root / "run-provenance.json"), "run_id")
    predictions = defaultdict(list)
    with (root / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            predictions[record["run_id"]].append(record)

    assert_equal(len(archived), count, f"{root.name} run count")
    assert_equal(sorted(predictions), sorted(archived), f"{root.name} prediction run IDs")
    assert_equal(sorted(provenance), sorted(archived), f"{root.name} provenance run IDs")
    expected = unique_index(plan, plan_key) if plan is not None else provenance
    assert_equal(sorted(expected), sorted(archived), f"{root.name} expected run IDs")
    rebuilt = []
    for run_id, stored in archived.items():
        rows, run_plan = predictions[run_id], expected[run_id]
        run_provenance = provenance[run_id]
        assert_equal(len(rows), predictions_per_run, f"{run_id} prediction count")
        sample_ids = {row["trace_id"] for row in rows}
        assert_equal(sample_ids, set(run_plan["test_trace_ids"]), f"{run_id} test membership")
        assert_equal(sample_ids, set(run_provenance["test_trace_ids"]), f"{run_id} provenance membership")
        if plan is not None:
            assert_equal(run_provenance["training_trace_ids"], run_plan["training_trace_ids"],
                         f"{run_id} training membership")
        for key in ("model", "condition", "repetition", "partition_id"):
            if key in run_plan:
                assert_equal({row[key] for row in rows}, {run_plan[key]}, f"{run_id} {key}")
            if key in stored:
                assert_equal({row[key] for row in rows}, {stored[key]}, f"{run_id} metric {key}")
        for row in rows:
            role, target = role_index[row["trace_id"]]
            assert_equal((row["scientific_role"], row["true_target"]), (role, target),
                         f"{run_id} {row['trace_id']} ground truth")
            assert_equal(row["split"], "test", f"{run_id} prediction split")
            assert_equal(row["predicted_class"],
                         int(row["predicted_anomaly_probability"] >= run_provenance["decision_threshold"]),
                         f"{run_id} decision threshold")
        metrics = reconstruct_metrics(rows)
        if extra_metrics:
            metrics["subgroup_confusion_counts"] = extra_metrics(rows)
        assert_equal(metrics, stored["metrics"], f"{run_id} metrics")
        rebuilt.append({**stored, "metrics": metrics})
    return rebuilt


def paired_index(rows):
    return {(r["partition_id"], r["model"], r["condition"]): r["metrics"] for r in rows}


def reconstruction():
    from ml.artifact_paths import resolve_scientific_path
    from ml.evaluation import phase4_experiment, r4_1_execution

    archived_code = read_json(FREEZE / "freeze-manifest.json")["execution"]
    r42 = runpy.run_path(str(verify_file(ROOT, archived_code["execution_implementation_path"],
                                       archived_code["execution_implementation_sha256"])))
    candidate = read_json(resolve_scientific_path("dataset-v1.0", ROOT))
    truth = {sample["sample_id"]: (sample["privileged_metadata"]["role"]["key"],
                                 int(sample["privileged_metadata"]["detector_truth"] == "anomaly"))
             for sample in candidate["samples"]}
    baseline = rebuild_runs(PHASE4 / "primary", 45, 84, "per-run-metrics.json", truth)
    grouped = rebuild_runs(PHASE4 / "grouped", 9, 91, "per-run-metrics.json", truth)
    baseline_summary = phase4_experiment.summarize_primary(baseline)
    compare_records(baseline_summary, read_json(PHASE4 / "primary/aggregate-summary.json"), "baseline summaries")

    repeated = rebuild_runs(R41, 180, 84, "run-level-metrics.json", truth,
                            plan=read_json(R41 / "run-matrix.json"))
    grouped_metrics = defaultdict(dict)
    for (partition, model, condition), metric in paired_index(repeated).items():
        grouped_metrics[partition, model][condition] = metric
    r41_pairs = r4_1_execution.make_paired_comparisons(grouped_metrics)
    compare_records(r41_pairs, read_json(R41 / "paired-comparisons.json"), "R4.1 paired results")
    r41_summary = r4_1_execution.aggregate(repeated, r41_pairs)
    compare_records(r41_summary, read_json(R41 / "aggregate-summary.json"), "R4.1 summaries")

    plan = read_json(PREEXEC / "execution-manifest-cd.json")["plans"]
    compare_records(read_json(R42 / "run-matrix.json"), plan, "R4.2 executed plan")
    new = rebuild_runs(R42, 120, 84, "run-level-metrics.json", truth,
                       plan=plan, plan_key="planned_run_id", extra_metrics=r42["subgroup_confusions"])
    repeated_by_id = unique_index(repeated, "run_id")
    anchor_refs = unique_index(read_json(R42 / "anchor-references.json")["anchors"], "run_id")
    assert_equal(set(anchor_refs), {r["run_id"] for r in repeated if r["condition"] in {"C1", "C2"}},
                 "R4.2 exact R4.1 anchor membership")
    anchors = []
    for row in repeated:
        if row["run_id"] not in anchor_refs:
            continue
        ref = anchor_refs[row["run_id"]]
        condition = {"C1": "A", "C2": "B"}[row["condition"]]
        assert_equal((ref["partition_id"], ref["model"], ref["anchor_condition"], ref["r4_1_condition"]),
                     (row["partition_id"], row["model"], condition, row["condition"]), "R4.2 anchor identity")
        anchors.append({**{k: v for k, v in row.items() if k != "split_seed"},
                        "condition": condition, "source": "immutable R4.1 anchor"})
    for row in plan:
        for label, expected_condition in (("A", "C1"), ("B", "C2")):
            ref = repeated_by_id[row[f"frozen_anchor_{label}_run_id"]]
            assert_equal((ref["partition_id"], ref["model"], ref["condition"]),
                         (row["partition_id"], row["model"], expected_condition), "R4.2 paired anchor")
    combined = anchors + new
    compare_records(unique_index(combined, "run_id"),
                    unique_index(read_json(R42 / "combined-evaluation-index.json"), "run_id"),
                    "R4.2 combined results")
    pairs = r42["make_pairs"](paired_index(combined))
    compare_records(pairs, read_json(R42 / "paired-comparisons.json"), "R4.2 paired results")
    summary = r42["aggregate"](combined, pairs)
    compare_records(summary, read_json(R42 / "aggregate-summary.json"), "R4.2 summaries")

    plot = runpy.run_path(str(artifact_path("scripts/generate_paper_figures_2_3.py")))
    inputs = plot["INPUTS"]
    plot_runs = [
        plot["reconstructed_runs"](plot["read_jsonl"](ROOT / inputs[index]), rows)
        for index, rows in ((0, baseline), (3, repeated), (8, new))
    ]
    plot_data = plot["plot_data_csv"](*plot_runs)
    expected_plot = FIGURES / "figure2_figure3_plot_data.csv"
    assert_equal(plot_data.encode() == expected_plot.read_bytes(), True, "publication plot data")
    return {
        "status": "PASS",
        "baseline": {"runs": len(baseline), "predictions": 3780, "summary": baseline_summary},
        "historical_grouped": {"runs": len(grouped), "predictions": 819},
        "r4_1": {"runs": len(repeated), "predictions": 15120, "summary": r41_summary,
                 "recall_loss_counts": dict(sorted(Counter(
                     r["model"] for r in r41_pairs if r["delta_recall_C1_minus_C0"] < 0
                 ).items()))},
        "r4_2": {"new_runs": len(new), "new_predictions": 10080, "reused_anchor_runs": len(anchors),
                 "paired_comparisons": len(pairs), "summary": summary},
        "publication_plot_data": {"rows": len(plot_data.splitlines()) - 1, "sha256": sha256_file(expected_plot)},
    }


def inspect_features(sample_reference):
    from ml.artifact_paths import resolve_scientific_path
    from ml.evaluation.pipeline import feature_profile_fingerprint
    from ml.evaluation.trace_features import (
        C1_FEATURE_NAMES, C2_FEATURE_NAMES, audit_candidate, extract_candidate_feature_rows,
    )

    candidate = read_json(resolve_scientific_path("dataset-v1.0", ROOT))
    audit = audit_candidate(candidate)
    assert_equal(audit, read_json(ROOT / "artifacts/icds-ml-phase1/structural-extraction-audit.json"),
                 "frozen feature audit")
    spec = read_json(ROOT / "IoT_auth_simulator/ml/icds/feature_spec_icds_v1.json")
    fields = spec["source_o03_fields"]
    used = sorted({field for feature in spec["features"] for field in feature["source_o03_fields"]})
    contracts = read_json(PREEXEC / "feature-contracts.json")["conditions"]
    names = {key: [row["feature_name"] for row in contract["ordered_features"]]
             for key, contract in contracts.items()}
    assert_equal(names["A"], list(C1_FEATURE_NAMES), "Full90 extractor order")
    assert_equal(names["B"], list(C2_FEATURE_NAMES), "Reduced extractor order")
    f2 = read_json(PREEXEC / "f2-generic-temporal-features.json")["ordered_feature_names"]
    assert_equal(names["C"], [name for name in names["A"]
                             if name != "semantic_timing_min_signed_causal_margin"], "89-feature projection")
    assert_equal(names["D"], [name for name in names["C"] if name not in f2], "82-feature projection")
    order_audit = read_json(PREEXEC / "audit/candidate-feature-lists.json")
    audited_orders = {row["canonical_audit_list_fingerprint"]: row["retained_features"]
                      for row in order_audit["candidates"]}
    for key, ordered in names.items():
        contract = contracts[key]
        assert_equal(len(ordered), contract["feature_count"], f"{key} feature count")
        assert_equal(ordered, audited_orders[contract["ordered_list_fingerprint"]],
                     f"{key} audited feature order")
        excluded = {"contract_fingerprint", "ordered_list_fingerprint", "canonicalization"}
        if key in {"A", "B"}:
            # The archived A/B digest predates the historical anchor linkage fields.
            profile = {"A": "C1", "B": "C2"}[key]
            assert_equal(contract["historical_r4_1_feature_profile"], profile, f"{key} anchor profile")
            assert_equal(contract["historical_r4_1_feature_profile_fingerprint"],
                         feature_profile_fingerprint(profile), f"{key} anchor feature fingerprint")
            excluded.update({"historical_r4_1_feature_profile", "historical_r4_1_feature_profile_fingerprint"})
        core = {k: v for k, v in contract.items() if k not in excluded}
        assert_equal(canonical_hash(core), contract["contract_fingerprint"], f"{key} contract fingerprint")
    result = {
        "status": "PASS",
        "scope": spec["prediction_horizon"],
        "declared_o03_fields": fields,
        "used_o03_fields": used,
        "traces": audit["traces_processed"],
        "frozen_audit_fingerprint": audit["audit_fingerprint"],
        "feature_row_fingerprints": audit["determinism"],
        "ordered_contracts": {key: {
            "count": len(names[key]), "names": names[key],
            "ordered_list_fingerprint": contract["ordered_list_fingerprint"],
        } for key, contract in contracts.items()},
    }
    if sample_reference:
        rows = unique_index(extract_candidate_feature_rows(candidate), "sample_reference")
        if sample_reference not in rows:
            raise ValueError(f"unknown sample reference: {sample_reference}")
        result["sample"] = rows[sample_reference]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("archive").add_argument(
        "--files-only", action="store_true", help="explicitly omit Git object checks")
    metric_parser = commands.add_parser("metrics")
    metric_parser.add_argument("--include-summaries", action="store_true",
                               help="print condition-level distributions and paired summaries")
    commands.add_parser("features").add_argument("--sample", help="include one vector by sample ID")
    args = parser.parse_args()
    try:
        if args.command == "archive":
            result = archive(args.files_only)
        elif args.command == "metrics":
            result = reconstruction()
            if not args.include_summaries:
                for key in ("baseline", "r4_1", "r4_2"):
                    del result[key]["summary"]
        else:
            result = inspect_features(args.sample)
    except (Exception, SystemExit) as error:
        message = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            message += f": {error.stderr.strip()}"
        print(json.dumps({"status": "FAIL", "command": args.command, "error": message}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
