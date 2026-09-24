"""DP.2R staged publication, companion binding and same-stem failure checks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from simulator.config.settings import cfg
from simulator.data.dataset_contract import field_manifest
from simulator.data.publication import load_dataset_profile_association
from simulator.data.synchronized_views import SynchronizedOutputViews
from simulator.provenance import build_generation_provenance
from simulator.runner import run_simulation_with_population


STEM = "dp2r-bundle"


def snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


class ProfilePublicationTests(unittest.TestCase):
    def setUp(self):
        with ExitStack() as stack:
            for name, value in {
                "random_seed": 12345, "num_devices": 1,
                "num_sessions_normal": 3, "num_sessions_attack": 0,
                "wire_entities": False,
            }.items():
                stack.enter_context(patch.object(cfg.simulation, name, value))
            stack.enter_context(patch(
                "simulator.runner.time.time", return_value=1_710_000_000.2,
            ))
            self.sequences, self.population = run_simulation_with_population()
            self.provenance = build_generation_provenance(
                cfg, run_id="dp2r-old-run", generator_git_dirty=False,
            )
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.views = SynchronizedOutputViews(self.directory)

    def save(self, *, profile: bool, run_id: str = "dp2r-old-run"):
        return self.views.save(
            self.sequences, stem=STEM,
            provenance=replace(self.provenance, run_id=run_id),
            run_population=self.population if profile else None,
        )

    def test_present_companion_is_declared_bound_and_reloadable(self):
        paths = self.save(profile=True)
        manifest = json.loads(paths["dataset_manifest"].read_text())
        declaration = manifest["device_profile_provenance"]
        companion = paths["device_profile_provenance"]
        self.assertEqual("present", declaration["status"])
        self.assertEqual(companion.name, declaration["filename"])
        self.assertTrue(companion.is_file())
        self.assertEqual(
            "sha256:" + hashlib.sha256(companion.read_bytes()).hexdigest(),
            declaration["content_digest"],
        )
        association = load_dataset_profile_association(self.directory, STEM)
        self.assertEqual("present", association["status"])
        self.assertEqual(
            declaration["population_fingerprint"],
            association["profile_provenance"]["population"]["population_fingerprint"],
        )
        self.assertEqual(manifest["run_id"], association["profile_provenance"]["run_id"])
        self.assertNotIn("device_profile_provenance", field_manifest())
        for view in ("detector_observations", "ground_truth_events", "debug"):
            self.assertNotIn("population_fingerprint", paths[view].read_text())
            self.assertNotIn("behavioral_profile_id", paths[view].read_text())

    def test_profile_stage_write_failure_preserves_previous_valid_bundle(self):
        self.save(profile=True)
        previous = snapshot(self.directory)
        original = Path.write_bytes

        def fail_profile(path: Path, content: bytes):
            if path.name.endswith("_device_profile_provenance.json") and "-stage-" in path.parent.name:
                raise OSError("injected staged profile write failure")
            return original(path, content)

        with patch.object(Path, "write_bytes", fail_profile):
            with self.assertRaisesRegex(OSError, "staged profile write failure"):
                self.save(profile=True, run_id="dp2r-new-run")
        self.assertEqual(previous, snapshot(self.directory))
        self.assertEqual([], list(self.directory.glob(f".{STEM}-stage-*")))
        self.assertEqual("present", load_dataset_profile_association(self.directory, STEM)["status"])

    def test_successful_nonprofile_same_stem_replacement_removes_stale_companion(self):
        old = self.save(profile=True)
        old_companion = old["device_profile_provenance"]
        self.assertTrue(old_companion.exists())
        new = self.save(profile=False, run_id="dp2r-new-run")
        self.assertFalse(old_companion.exists())
        manifest = json.loads(new["dataset_manifest"].read_text())
        self.assertEqual("absent", manifest["device_profile_provenance"]["status"])
        self.assertEqual("absent", load_dataset_profile_association(self.directory, STEM)["status"])

    def test_failed_nonprofile_stage_does_not_delete_previous_companion(self):
        self.save(profile=True)
        previous = snapshot(self.directory)
        original = Path.write_bytes

        def fail_observation(path: Path, content: bytes):
            if path.name.endswith("_observations.csv") and "-stage-" in path.parent.name:
                raise OSError("injected staged observation write failure")
            return original(path, content)

        with patch.object(Path, "write_bytes", fail_observation):
            with self.assertRaisesRegex(OSError, "staged observation write failure"):
                self.save(profile=False, run_id="dp2r-new-run")
        self.assertEqual(previous, snapshot(self.directory))
        self.assertEqual([], list(self.directory.glob(f".{STEM}-stage-*")))
        self.assertEqual("present", load_dataset_profile_association(self.directory, STEM)["status"])

    def test_failed_final_manifest_replacement_rolls_back_previous_bundle(self):
        self.save(profile=True)
        previous = snapshot(self.directory)
        original = os.replace
        failed = False

        def fail_once(source, destination):
            nonlocal failed
            if Path(destination).name == f"{STEM}_dataset_manifest.json" and not failed:
                failed = True
                raise OSError("injected final manifest replacement failure")
            return original(source, destination)

        with patch("simulator.data.publication.os.replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "manifest replacement failure"):
                self.save(profile=False, run_id="dp2r-new-run")
        self.assertTrue(failed)
        self.assertEqual(previous, snapshot(self.directory))
        self.assertEqual("present", load_dataset_profile_association(self.directory, STEM)["status"])

    def test_missing_and_mismatched_required_companion_are_rejected(self):
        paths = self.save(profile=True)
        companion = paths["device_profile_provenance"]
        moved = self.directory / "temporarily-moved-profile.json"
        companion.rename(moved)
        with self.assertRaisesRegex(ValueError, "required Device Profile companion is missing"):
            load_dataset_profile_association(self.directory, STEM)
        moved.rename(companion)
        altered = json.loads(companion.read_text())
        altered["population"]["population_fingerprint"] = "sha256:wrong-population"
        companion.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            load_dataset_profile_association(self.directory, STEM)

    def test_declared_population_identity_mismatch_is_rejected(self):
        paths = self.save(profile=True)
        manifest = json.loads(paths["dataset_manifest"].read_text())
        provenance = json.loads(paths["provenance"].read_text())
        wrong = "sha256:unrelated-population"
        manifest["device_profile_provenance"]["population_fingerprint"] = wrong
        provenance["dataset"]["capabilities"]["device_profile_provenance"] = (
            manifest["device_profile_provenance"]
        )
        paths["dataset_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        paths["provenance"].write_text(json.dumps(provenance), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint does not match declaration"):
            load_dataset_profile_association(self.directory, STEM)

    def test_unexpected_companion_for_absent_status_is_rejected(self):
        old = self.save(profile=True)
        retained = self.directory / "retained-old-profile.json"
        shutil.copy2(old["device_profile_provenance"], retained)
        self.save(profile=False, run_id="dp2r-new-run")
        shutil.copy2(retained, self.directory / f"{STEM}_device_profile_provenance.json")
        with self.assertRaisesRegex(ValueError, "unexpected same-stem"):
            load_dataset_profile_association(self.directory, STEM)


if __name__ == "__main__":
    unittest.main()
