"""Read prepared Parquet data without invoking replay extraction."""

import json
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from training import bang_dataset, bump_dataset, context_dataset, overcommitting_dataset

CONTEXT_READERS = {
    "bumping_teammate": bump_dataset,
    "bang_with_time": bang_dataset,
    "overcommitting": overcommitting_dataset,
}


def load(path):
    metadata = pq.read_schema(path).metadata
    if metadata and context_dataset.METADATA_KEY in metadata:
        if b"rl_mistake_analysis" in metadata:
            raise ValueError("A table must declare exactly one feature contract")
        kind = json.loads(metadata[context_dataset.METADATA_KEY])["kind"]
        if kind not in CONTEXT_READERS:
            raise ValueError(f"Unsupported context kind: {kind}")
        table, contract = CONTEXT_READERS[kind].load(path, labeled=True)
        contract = {
            **contract,
            "feature_names": [d["name"] for d in contract["feature_definitions"]],
        }
        return table, contract
    if metadata is None or b"rl_mistake_analysis" not in metadata:
        raise ValueError("Missing rl_mistake_analysis table metadata")
    table = pq.read_table(path)
    contract = json.loads(metadata[b"rl_mistake_analysis"])
    names = contract.get("feature_names")
    if (
        contract.get("schema_version") != 1
        or not isinstance(contract.get("kind"), str)
        or not contract["kind"].strip()
        or not isinstance(contract.get("input_schema"), str)
        or not contract["input_schema"].strip()
        or not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
        or table.num_rows == 0
    ):
        raise ValueError("Invalid prepared feature contract or empty table")
    required = [
        "replay_sha256",
        "frame",
        "initiator",
        "victim",
        "raw_time",
        "label",
        "engine_revision",
        *names,
    ]
    if any(name not in table.column_names for name in required):
        raise ValueError("Prepared table is missing required columns")
    if any(table[name].null_count for name in required):
        raise ValueError("Required training columns contain null values")
    if not pa.types.is_integer(table["label"].type):
        raise ValueError("Labels must be integer 0 or 1")
    if not all(label in (0, 1) for label in table["label"].to_pylist()):
        raise ValueError("Labels must be integer 0 or 1")
    if not pa.types.is_integer(table["frame"].type) or any(
        frame < 0 for frame in table["frame"].to_pylist()
    ):
        raise ValueError("Frames must be nonnegative integers")
    for name in ["raw_time", *names]:
        if not (
            pa.types.is_floating(table[name].type)
            or pa.types.is_integer(table[name].type)
        ):
            raise ValueError(f"Non-numeric training column: {name}")
        if not np.isfinite(table[name].to_numpy()).all():
            raise ValueError(f"Nonfinite training column: {name}")
    identities = set()
    for row in table.select(
        ["replay_sha256", "frame", "initiator", "victim"]
    ).to_pylist():
        checksum = row["replay_sha256"]
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("Invalid replay fingerprint")
        initiator = json.loads(row["initiator"])
        victim = json.loads(row["victim"])
        if initiator is None or victim is None or initiator == victim:
            raise ValueError("Invalid incident participants")
        identity = (
            checksum,
            row["frame"],
            json.dumps(initiator, sort_keys=True),
            json.dumps(victim, sort_keys=True),
        )
        if identity in identities:
            raise ValueError("Duplicate training incident")
        identities.add(identity)
    return table, contract


def arrays(table, contract):
    if any(
        reader.supports_schema(contract["input_schema"])
        for reader in CONTEXT_READERS.values()
    ):
        return context_dataset.arrays(table, contract)
    features = np.column_stack(
        [table[name].to_numpy() for name in contract["feature_names"]]
    )
    if (
        not np.isfinite(features).all()
        or (np.abs(features) > np.finfo(np.float32).max).any()
    ):
        raise ValueError("Features cannot be represented as finite float32 values")
    return (
        features.astype(np.float32),
        table["label"].to_numpy(),
        table["replay_sha256"].to_numpy(),
    )


def select_features(contract, names, input_schema):
    """Select ordered context inputs without changing their definitions or sampling."""
    if not any(
        reader.supports_schema(contract["input_schema"])
        and reader.supports_schema(input_schema)
        for reader in CONTEXT_READERS.values()
    ):
        raise ValueError("Feature selection requires supported context schemas")
    available = {
        feature["name"]: feature for feature in contract["feature_definitions"]
    }
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or name not in available for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "Selected features must be unique names in the source contract"
        )
    return {
        **contract,
        "input_schema": input_schema,
        "feature_names": names,
        "feature_definitions": [available[name] for name in names],
    }
