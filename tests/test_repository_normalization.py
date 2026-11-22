"""Integrity checks for the V2-01 venue-neutral packaging layer."""

from __future__ import annotations

import hashlib
import json
import random
import unittest
from pathlib import Path

from ml.artifact_paths import load_version_registry, resolve_scientific_path
from ml.evaluation.pipeline import model_seed
from simulator.anomaly_contract import SYNCHRONIZED_GENERATION_PROFILE


ROOT = Path(__file__).resolve().parents[1]


class RepositoryNormalizationTests(unittest.TestCase):
    def test_scientific_aliases_resolve(self) -> None:
        registry = load_version_registry(ROOT)
        self.assertTrue(
            {"behavioral-model-v1", "dataset-v1.0", "eval-v1.0", "sync-v1.0.0"}
            <= set(registry["aliases"])
        )
        self.assertEqual(
            resolve_scientific_path("dataset-v1.0", ROOT),
            ROOT / "artifacts/icds-timing-nonce-candidate/scientific-candidate.json",
        )
        self.assertEqual(
            resolve_scientific_path("eval-v1.0", ROOT),
            ROOT / "artifacts/icds-ml-phase4/fingerprints.json",
        )

    def test_reference_manifest_hashes(self) -> None:
        manifests = [
            ROOT / "datasets/dataset-v1.0/manifest.json",
            ROOT / "validation/dataset-v1.0/manifest.json",
            ROOT / "experiments/eval-v1.0/manifest.json",
            ROOT / "publication-artifacts/eval-v1.0/manifest.json",
        ]
        manifests += sorted((ROOT / "experiments/eval-v1.0").glob("*/manifest.json"))
        manifests += sorted(
            (ROOT / "publication-artifacts/eval-v1.0").glob("*/manifest.json")
        )

        def visit(value: object) -> None:
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    path = ROOT / str(value["path"])
                    self.assertTrue(path.is_file())
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), value["sha256"])
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for manifest in manifests:
            visit(json.loads(manifest.read_text(encoding="utf-8")))

    def test_neutral_imports_do_not_change_global_rng(self) -> None:
        before = random.getstate()
        self.assertEqual(model_seed(1, "LR"), model_seed(1, "LR"))
        self.assertTrue(SYNCHRONIZED_GENERATION_PROFILE)
        self.assertEqual(before, random.getstate())

    def test_frozen_artifact_inventory_is_current(self) -> None:
        inventory = json.loads(
            (ROOT / "metadata/scientific-artifact-inventory.json").read_text()
        )
        self.assertEqual(inventory["file_count"], 207)
        for record in inventory["files"]:
            path = ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
