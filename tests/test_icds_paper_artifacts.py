"""Integrity checks for the read-only ICDS Phase-4 paper-artifact layer."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ml.icds.paper_artifacts import (
    CONDITIONS,
    EXPECTED_HASHES,
    MODELS,
    PHASE4_TREE_FINGERPRINT,
    TIMING_FPR,
    TIMING_RECALL,
    build_table_a,
    build_table_b,
    build_table_c,
    build_table_d,
    sha256_file,
    verify_and_load,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT / "artifacts/icds-ml-phase4"


class ICDSPaperArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = verify_and_load(PHASE4)

    def test_frozen_hashes_and_execution_grid(self):
        self.assertEqual(self.evidence.source_hashes, EXPECTED_HASHES)
        self.assertEqual(self.evidence.primary_prediction_count, 3780)
        self.assertEqual(self.evidence.grouped_prediction_count, 819)
        self.assertEqual(len(self.evidence.primary_metrics), 45)
        self.assertEqual(len(self.evidence.grouped_metrics), 9)
        fingerprints = json.loads((PHASE4 / "fingerprints.json").read_text())
        self.assertEqual(fingerprints["artifact_tree_fingerprint"], PHASE4_TREE_FINGERPRINT)

    def test_tables_have_complete_frozen_model_condition_coverage(self):
        table_a = build_table_a(self.evidence)
        table_b = build_table_b(self.evidence)
        table_c = build_table_c(self.evidence)
        table_d = build_table_d(self.evidence)
        self.assertEqual({(row["model"], row["condition"]) for row in table_a},
                         {(model, condition) for model in MODELS for condition in CONDITIONS})
        self.assertEqual({row["model"] for row in table_b}, set(MODELS))
        self.assertEqual({row["model"] for row in table_c}, set(MODELS))
        self.assertEqual({(row["model"], row["condition"]) for row in table_d},
                         {(model, condition) for model in MODELS for condition in CONDITIONS})

    def test_primary_tables_copy_frozen_summaries_exactly(self):
        summary = self.evidence.primary_summary["by_model_condition"]
        table_a = {(row["model"], row["condition"]): row
                   for row in build_table_a(self.evidence)}
        self.assertEqual(table_a[("LR", "C0")]["anomaly_f1_median"],
                         summary["LR"]["C0"]["anomaly_f1"]["median"])
        self.assertEqual(table_a[("RF", "C2")]["timing_recall_iqr"],
                         summary["RF"]["C2"][TIMING_RECALL]["iqr"])
        table_b = {row["model"]: row for row in build_table_b(self.evidence)}
        self.assertEqual(table_b["DT"]["c1_timing_hn_fpr_median"],
                         summary["DT"]["C1"][TIMING_FPR]["median"])
        table_c = {row["model"]: row for row in build_table_c(self.evidence)}
        self.assertEqual(table_c["LR"]["c2_timing_recall_median"],
                         summary["LR"]["C2"][TIMING_RECALL]["median"])

    def test_frozen_timing_claim_values_are_not_transformed(self):
        table_b = {row["model"]: row for row in build_table_b(self.evidence)}
        self.assertEqual(table_b["LR"]["c0_timing_hn_fpr_median"], 1.0)
        self.assertEqual(table_b["DT"]["c0_timing_hn_fpr_median"], 1.0)
        self.assertEqual(table_b["RF"]["c0_timing_hn_fpr_median"], 0.75)
        self.assertEqual(table_b["DT"]["c1_timing_hn_fpr_median"], 0.0)
        self.assertEqual(table_b["RF"]["c1_timing_hn_fpr_median"], 0.0)

    def test_grouped_table_preserves_raw_counts(self):
        table = {(row["model"], row["condition"]): row
                 for row in build_table_d(self.evidence)}
        expected = {
            ("LR", "C0"): (16, 16, 12, 12, 11, 11),
            ("LR", "C1"): (11, 16, 7, 12, 11, 11),
            ("DT", "C0"): (16, 16, 12, 12, 11, 11),
            ("DT", "C1"): (0, 16, 12, 12, 11, 11),
            ("RF", "C0"): (14, 16, 12, 12, 11, 11),
            ("RF", "C1"): (0, 16, 12, 12, 11, 11),
        }
        for key, counts in expected.items():
            row = table[key]
            self.assertEqual(
                (row["timing_hn_fp"], row["timing_hn_count"],
                 row["timing_anomaly_tp"], row["timing_anomaly_count"],
                 row["nonce_anomaly_tp"], row["nonce_anomaly_count"]),
                counts,
            )

    def test_exports_match_in_memory_tables_and_upstream_hashes_remain_frozen(self):
        output = ROOT / "artifacts/icds-paper-figures"
        if not output.exists():
            self.skipTest("paper artifacts not generated yet")
        with (output / "table-a-primary-performance.csv").open(newline="") as stream:
            exported = list(csv.DictReader(stream))
        in_memory = build_table_a(self.evidence)
        self.assertEqual(len(exported), len(in_memory))
        self.assertEqual(float(exported[0]["anomaly_f1_median"]),
                         in_memory[0]["anomaly_f1_median"])
        manifest = json.loads((output / "paper-artifact-manifest.json").read_text())
        self.assertEqual(manifest["source_phase4_tree_fingerprint"], PHASE4_TREE_FINGERPRINT)
        self.assertEqual(len(manifest["artifacts"]), 20)
        for entry in manifest["artifacts"]:
            self.assertEqual(sha256_file(output / entry["filename"]), entry["sha256"])
        for relative, expected in EXPECTED_HASHES.items():
            self.assertEqual(sha256_file(PHASE4 / relative), expected)


if __name__ == "__main__":
    unittest.main()
