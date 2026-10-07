"""Build model catalog entries from exported ONNX artifacts and training metadata."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import onnx

from training.dataset import load


def tensor_locations(message):
    if (
        isinstance(message, onnx.TensorProto)
        and message.data_location == onnx.TensorProto.EXTERNAL
    ):
        yield onnx.external_data_helper.ExternalDataInfo(message).location
    for field, value in message.ListFields():
        if field.message_type is not None:
            for child in value if field.is_repeated else (value,):
                yield from tensor_locations(child)


def artifact(directory, relative_path):
    if (
        not isinstance(relative_path, str)
        or "\\" in relative_path
        or Path(relative_path).is_absolute()
        or any(part in ("", ".", "..") for part in relative_path.split("/"))
    ):
        raise ValueError(f"Invalid artifact path: {relative_path}")
    directory = Path(directory).resolve(strict=True)
    path = (directory / relative_path).resolve(strict=True)
    if not path.is_relative_to(directory):
        raise ValueError(f"Artifact is outside the release directory: {relative_path}")
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": relative_path, "sha256": checksum}


def model_entry(
    directory,
    training_table,
    *,
    version,
    model_file,
    evaluation_file,
    keep_threshold,
    source_revision,
    status,
):
    table, contract = load(training_table)
    if (
        not isinstance(version, str)
        or not version.strip()
        or not isinstance(source_revision, str)
        or not source_revision.strip()
        or type(keep_threshold) not in (int, float)
        or not math.isfinite(keep_threshold)
        or not 0 <= keep_threshold <= 1
        or status not in ("experimental", "approved")
    ):
        raise ValueError("Invalid model release identity, threshold, or status")
    model_artifact = artifact(directory, model_file)
    graph_path = Path(directory) / model_file
    graph = onnx.load(graph_path, load_external_data=False)
    tensor_files = set()
    for location in tensor_locations(graph):
        if (
            Path(location).is_absolute()
            or "\\" in location
            or any(part in ("", ".", "..") for part in location.split("/"))
        ):
            raise ValueError(f"Invalid external tensor location: {location}")
        tensor_files.add((Path(model_file).parent / location).as_posix())
    paths = [model_file, evaluation_file, *sorted(tensor_files)]
    if len(set(paths)) != len(paths):
        raise ValueError("Model, evaluation, and tensor artifacts must be distinct")
    files = [model_artifact, *(artifact(directory, path) for path in paths[1:])]
    model_parent = Path(model_file).parent
    if any(not Path(path).is_relative_to(model_parent) for path in tensor_files):
        raise ValueError("Tensor files must be inside the model directory")
    onnx.checker.check_model(str(Path(directory) / model_file))
    revisions = sorted(set(table["engine_revision"].to_pylist()))
    if any(
        not isinstance(revision, str) or not revision.strip() for revision in revisions
    ):
        raise ValueError("Missing training extraction provenance")
    return {
        "kind": contract["kind"],
        "version": version,
        "input_schema": contract["input_schema"],
        "feature_names": contract["feature_names"],
        "runtime": "onnx",
        "model_file": model_file,
        "evaluation_file": evaluation_file,
        "keep_threshold": keep_threshold,
        "status": status,
        "provenance": {
            "source_revision": source_revision,
            "engine_revisions": revisions,
        },
        "files": files,
    }


def write_manifest(directory, version, models):
    if not isinstance(version, str) or not version.strip() or not models:
        raise ValueError("Model catalog needs a version and at least one entry")
    identities = set()
    hashes = {}
    for model in models:
        identity = (model["kind"], model["version"])
        if identity in identities:
            raise ValueError("Duplicate model kind and version")
        identities.add(identity)
        for file in model["files"]:
            actual = artifact(directory, file["path"])
            if actual != file:
                raise ValueError(f"Artifact hash mismatch: {file['path']}")
            if file["path"] in hashes and hashes[file["path"]] != file["sha256"]:
                raise ValueError("Conflicting artifact hashes")
            hashes[file["path"]] = file["sha256"]
    text = (
        json.dumps(
            {"schema_version": 1, "version": version, "models": models},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    directory = Path(directory)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "manifest.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
