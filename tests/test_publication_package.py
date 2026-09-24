"""Publication-only verification and allowlisted snapshot tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_INTERNAL_DOCS = {
    "docs/ml-data-analysis-handover.md",
    "docs/ml-evaluation-work-package.md",
    "docs/supervisor-simulator-walkthrough.md",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PublicationPackageTests(unittest.TestCase):
    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        source = str(ROOT / "IoT_auth_simulator")
        current = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = source + (os.pathsep + current if current else "")
        return environment

    def test_read_only_frozen_verifier(self):
        result = subprocess.run(
            [sys.executable, "scripts/verify_publication.py", "--verify-only"],
            cwd=ROOT, env=self.environment(), check=True,
            capture_output=True, text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["verification"]["candidate_samples"], 336)
        self.assertEqual(payload["verification"]["primary_runs"], 45)
        self.assertEqual(payload["verification"]["grouped_runs"], 9)
        self.assertEqual(payload["verification"]["paper_artifacts"], 20)

    def test_publication_comparison_normalizes_only_execution_provenance(self):
        from scripts.verify_publication import _without_execution_metadata

        payload = {
            "repository_commit": "publication-commit",
            "created_at_utc": "execution-time",
            "prediction": 1,
            "nested": {"completed_at_utc": "execution-time", "metric": 0.75},
        }
        self.assertEqual(
            _without_execution_metadata(payload),
            {"prediction": 1, "nested": {"metric": 0.75}},
        )

    def test_publication_snapshot_is_allowlisted_hashable_and_path_neutral(self):
        with tempfile.TemporaryDirectory(prefix="icds-publication-test-") as directory:
            destination = Path(directory) / "snapshot"
            result = subprocess.run(
                [sys.executable, "scripts/export_publication_snapshot.py", str(destination)],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            payload = json.loads(result.stdout)
            self.assertGreater(payload["file_count"], 0)
            self.assertTrue((destination / "README.md").is_file())
            self.assertTrue((destination / "artifacts/icds-ml-phase4/fingerprints.json").is_file())
            self.assertTrue((destination / "notebooks/icds_phase4_paper_artifacts.ipynb").is_file())
            self.assertFalse(
                (destination / "IoT_auth_simulator/simulator/data/output/iot_auth_ml_events.json").exists()
            )
            for relative in PROTECTED_INTERNAL_DOCS:
                self.assertFalse((destination / relative).exists())
            for legacy_notebook in (
                "IoT_auth_simulator/ml/decision_tree_pipeline.ipynb",
                "IoT_auth_simulator/ml/logistic_regression_pipeline.ipynb",
                "IoT_auth_simulator/ml/random_forest_pipeline.ipynb",
                "IoT_auth_simulator/simulator/data/output/iot_auth_features_analysis.ipynb",
            ):
                self.assertFalse((destination / legacy_notebook).exists())

            manifest = json.loads((destination / "PUBLICATION_SNAPSHOT.json").read_text())
            self.assertEqual(manifest["file_count"], payload["file_count"])
            unix_home_prefix = "/" + "Users/"
            windows_home_prefix = "C:" + "\\" + "Users" + "\\"
            for record in manifest["files"]:
                path = destination / record["path"]
                self.assertEqual(path.stat().st_size, record["size"])
                self.assertEqual(sha256_file(path), record["sha256"])
                if path.suffix.lower() in {".py", ".md", ".txt", ".yml", ".json", ".ipynb"}:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    self.assertNotIn(unix_home_prefix, text, record["path"])
                    self.assertNotIn(windows_home_prefix, text, record["path"])


if __name__ == "__main__":
    unittest.main()
