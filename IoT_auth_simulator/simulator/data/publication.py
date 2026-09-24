"""Staged synchronized bundle publication and profile-companion association.

Each file replacement is atomic on its own filesystem, but the complete
multi-file bundle is not an all-file transaction. The dataset manifest is
published last; best-effort rollback restores an older same-stem bundle if a
final replacement fails. All staging writes and association checks complete
before any final file is replaced.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from simulator.device_profiles.provenance import (
    SUPPORTED_PROFILE_PROVENANCE_SCHEMA_VERSIONS,
    load_device_profile_provenance,
)


PROFILE_ASSOCIATION_VERSION = "dp2r-profile-companion-association-v1"


def profile_companion_name(stem: str) -> str:
    return f"{stem}_device_profile_provenance.json"


def profile_artifact_bytes(artifact: Mapping[str, Any]) -> bytes:
    """Serialize the dedicated companion without generic dataset flattening."""
    return json.dumps(artifact, indent=2, allow_nan=False).encode("utf-8")


def profile_association_status(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    """In-memory capability status before a particular filename is published."""
    if artifact is None:
        return {"association_version": PROFILE_ASSOCIATION_VERSION, "status": "absent"}
    return {
        "association_version": PROFILE_ASSOCIATION_VERSION,
        "status": "present",
        "provenance_schema_version": artifact["provenance_schema_version"],
        "population_fingerprint": artifact["population"]["population_fingerprint"],
        "run_id": artifact["run_id"],
    }


def profile_publication_declaration(
    *, stem: str, artifact: Mapping[str, Any] | None,
    artifact_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Declare companion absence or bind a required file to this dataset."""
    if artifact is None:
        return profile_association_status(None)
    if artifact_bytes is None:
        raise ValueError("a present profile companion requires serialized bytes")
    return {
        **profile_association_status(artifact),
        "filename": profile_companion_name(stem),
        "content_digest": "sha256:" + hashlib.sha256(artifact_bytes).hexdigest(),
    }


def load_dataset_profile_association(output_dir: str | Path, stem: str) -> dict[str, Any]:
    """Interpret the dataset manifest, not ambient file presence, as authority.

    The standalone profile loader remains usable independently. This loader
    additionally binds the declared companion to the dataset's run provenance
    and trace device IDs. An unexpected same-stem companion is rejected when
    the manifest declares absence.
    """
    output = Path(output_dir)
    manifest = json.loads((output / f"{stem}_dataset_manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / f"{stem}_provenance.json").read_text(encoding="utf-8"))
    declaration = manifest.get("device_profile_provenance")
    if not isinstance(declaration, dict):
        raise ValueError("dataset manifest lacks explicit Device Profile status")
    if declaration.get("association_version") != PROFILE_ASSOCIATION_VERSION:
        raise ValueError("unsupported Device Profile association version")
    if provenance["dataset"]["capabilities"].get("device_profile_provenance") != declaration:
        raise ValueError("dataset manifest and run provenance disagree about profile companion")
    run_id = manifest["run_id"]
    if provenance["dataset"]["generation"]["run_id"] != run_id:
        raise ValueError("dataset manifest and run provenance have different run IDs")

    companion_path = output / profile_companion_name(stem)
    if declaration.get("status") == "absent":
        if set(declaration) != {"association_version", "status"}:
            raise ValueError("invalid absent Device Profile declaration")
        if companion_path.exists():
            raise ValueError("unexpected same-stem Device Profile companion for absent status")
        return {"status": "absent", "profile_provenance": None}

    if declaration.get("status") != "present":
        raise ValueError("unknown Device Profile companion status")
    if set(declaration) != {
        "association_version", "status", "filename", "provenance_schema_version",
        "population_fingerprint", "run_id", "content_digest",
    }:
        raise ValueError("invalid present Device Profile declaration")
    if declaration["filename"] != companion_path.name:
        raise ValueError("declared Device Profile filename does not match the bundle stem")
    if declaration["run_id"] != run_id:
        raise ValueError("declared profile run ID does not match dataset run ID")
    if declaration["provenance_schema_version"] not in SUPPORTED_PROFILE_PROVENANCE_SCHEMA_VERSIONS:
        raise ValueError("declared profile provenance schema is unsupported")
    if not companion_path.is_file():
        raise ValueError("required Device Profile companion is missing")
    content = companion_path.read_bytes()
    if declaration["content_digest"] != "sha256:" + hashlib.sha256(content).hexdigest():
        raise ValueError("Device Profile companion content digest mismatch")
    artifact = load_device_profile_provenance(companion_path)
    if artifact["run_id"] != run_id:
        raise ValueError("profile companion and dataset have different run IDs")
    if artifact["provenance_schema_version"] != declaration["provenance_schema_version"]:
        raise ValueError("profile companion schema does not match declaration")
    if artifact["population"]["population_fingerprint"] != declaration["population_fingerprint"]:
        raise ValueError("profile population fingerprint does not match declaration")
    profile_ids = {record["device_id"] for record in artifact["devices"]}
    for trace in provenance["traces"]:
        if trace["device_id"] not in profile_ids:
            raise ValueError("published runner trace has no Device Profile record")
    for event in provenance["events"]:
        if event["device_id"] not in profile_ids:
            raise ValueError("published runner event has no Device Profile record")
    return {"status": "present", "profile_provenance": artifact}


def publish_staged_bundle(
    output_dir: str | Path, *, stem: str, outputs: Mapping[str, bytes],
    profile_expected: bool,
) -> dict[str, Path]:
    """Stage, verify, replace, and best-effort rollback exact same-stem files."""
    output = Path(output_dir)
    manifest_name = f"{stem}_dataset_manifest.json"
    companion_name = profile_companion_name(stem)
    if manifest_name not in outputs:
        raise ValueError("dataset manifest is required as the final publication marker")
    if (companion_name in outputs) != profile_expected:
        raise ValueError("staged Device Profile presence disagrees with declared status")
    if len(outputs) != len(set(outputs)) or any(Path(name).name != name for name in outputs):
        raise ValueError("publication output names must be unique bare filenames")

    with tempfile.TemporaryDirectory(prefix=f".{stem}-stage-", dir=output) as temporary:
        staging = Path(temporary)
        for name, content in outputs.items():
            (staging / name).write_bytes(content)
        # This reads the staged manifest, provenance and (where present) the
        # staged companion. Failure here has not touched the previous finals.
        load_dataset_profile_association(staging, stem)

        previous = {}
        all_targets = set(outputs) | ({companion_name} if not profile_expected else set())
        backup_dir = staging / "previous"
        backup_dir.mkdir()
        for name in all_targets:
            target = output / name
            if target.exists():
                backup = backup_dir / name
                shutil.copy2(target, backup)
                previous[name] = backup

        replace_order = ([companion_name] if profile_expected else []) + [
            name for name in outputs if name not in {companion_name, manifest_name}
        ] + [manifest_name]
        touched = []
        try:
            for name in replace_order[:-1]:
                os.replace(staging / name, output / name)
                touched.append(name)
            # On an absent publication, remove stale companion only after all
            # staging/verification succeeds and immediately before manifest-last.
            if not profile_expected and (output / companion_name).exists():
                (output / companion_name).unlink()
                touched.append(companion_name)
            os.replace(staging / manifest_name, output / manifest_name)
            touched.append(manifest_name)
        except Exception as publication_error:
            rollback_errors = []
            for name in reversed(touched):
                try:
                    target = output / name
                    if name in previous:
                        os.replace(previous[name], target)
                    elif target.exists():
                        target.unlink()
                except Exception as rollback_error:
                    rollback_errors.append(f"{name}: {rollback_error}")
            if rollback_errors:
                raise RuntimeError(
                    "publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from publication_error
            raise
    return {name: output / name for name in outputs}
