"""Validate typed context tables without invoking replay extraction."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

METADATA_KEY = b"rl_mistake_analysis_features"
DATA_TYPES = {"boolean": pa.bool_(), "f32": pa.float32(), "u32": pa.uint32()}


def load(
    path,
    *,
    kind,
    supports_schema,
    identity_fields,
    json_fields,
    identity,
    duplicate_message,
    labeled=False,
    source_id_field="source_event_id",
    multiple_source_ids=False,
    validation_fields=(),
):
    """Validate the context contract, preserving declared feature order and nulls."""
    table = pq.read_table(path)
    table.validate(full=True)
    if not table.schema.metadata or METADATA_KEY not in table.schema.metadata:
        raise ValueError("Missing context metadata")
    contract = json.loads(table.schema.metadata[METADATA_KEY])
    if (
        contract["schema_version"] != 1
        or contract["kind"] != kind
        or not supports_schema(contract["input_schema"])
        or not table.num_rows
    ):
        raise ValueError("Unsupported context contract or empty table")
    definitions = contract["feature_definitions"]
    names = [feature["name"] for feature in definitions]
    required = [
        *identity_fields,
        "raw_time",
        "engine_revision",
        source_id_field,
        *validation_fields,
    ]
    if labeled:
        required.append("label")
    if (
        not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
        or set(names)
        & {*required, "label", "annotation_engine_revision", "reject_reason"}
        or len(set(table.column_names)) != table.num_columns
        or any(name not in table.column_names for name in [*required, *names])
    ):
        raise ValueError("Invalid context feature names or missing required columns")
    if any(table[name].null_count for name in required):
        raise ValueError(
            "Required identities, timestamps, revisions and labels cannot be null"
        )
    if not pa.types.is_integer(table["frame"].type):
        raise ValueError("Frames must be integers")
    if not pa.types.is_floating(table["raw_time"].type):
        raise ValueError("Replay timestamps must be floating-point numbers")
    if not np.isfinite(table["raw_time"].to_numpy()).all():
        raise ValueError("Replay timestamps must be finite")
    if multiple_source_ids:
        for value in table[source_id_field].to_pylist():
            ids = json.loads(value)
            if (
                not isinstance(ids, list)
                or not ids
                or any(not isinstance(value, str) or not value for value in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("Source event IDs must be distinct nonempty strings")
    for feature in definitions:
        name = feature["name"]
        if (
            feature["data_type"] not in DATA_TYPES
            or table[name].type != DATA_TYPES[feature["data_type"]]
        ):
            raise ValueError(f"Feature type differs from its contract: {name}")
        if type(feature["nullable"]) is not bool:
            raise ValueError(f"Invalid feature nullability: {name}")
        if table[name].null_count and not feature["nullable"]:
            raise ValueError(f"Non-nullable feature contains nulls: {name}")
        if not np.isfinite(table[name].drop_null().to_numpy()).all():
            raise ValueError(f"Observed feature values must be finite: {name}")
    keys = []
    for row in table.select(identity_fields).to_pylist():
        for name in json_fields:
            row[name] = json.loads(row[name])
        keys.append(identity(row))
    if len(set(keys)) != len(keys):
        raise ValueError(duplicate_message)
    if labeled and (
        not pa.types.is_integer(table["label"].type)
        or any(value not in (0, 1) for value in table["label"].to_pylist())
    ):
        raise ValueError("Labels must be integer 0 or 1")
    return table, contract


def arrays(table, contract):
    """Convert boolean and numeric measurements to float32; nulls become NaN."""
    features = np.column_stack(
        [
            table[feature["name"]].cast(pa.float32()).to_numpy()
            for feature in contract["feature_definitions"]
        ]
    )
    return features, table["label"].to_numpy(), table["replay_sha256"].to_numpy()


def prepare(
    features,
    annotations,
    output,
    *,
    kind,
    load,
    identity_fields,
    json_fields,
    identity,
    source_id_field="source_event_id",
    multiple_source_ids=False,
    require_source_ids=False,
    validation_fields=(),
    validate_review=None,
):
    """Select reviewed subjects by exact replay, incident, player and timestamp."""
    output = Path(output)
    if output.suffix != ".parquet":
        raise ValueError("The labeled table must use a .parquet extension")
    if output.resolve() in (Path(features).resolve(), Path(annotations).resolve()):
        raise ValueError("The labeled table must not replace its inputs")
    table, _ = load(features)
    document = json.loads(Path(annotations).read_text())
    if document["schema_version"] != 1 or not document["annotations"]:
        raise ValueError(
            "Provide a version-1 annotation document with reviewed players"
        )
    rows = table.select(
        [*identity_fields, "raw_time", source_id_field, *validation_fields]
    ).to_pylist()
    lookup = {}
    for index, row in enumerate(rows):
        for name in json_fields:
            row[name] = json.loads(row[name])
        lookup[identity(row)] = index, row
    indices, labels, revisions, reasons = [], [], [], []
    seen = set()
    for review in document["annotations"]:
        if review["kind"] != kind:
            raise ValueError(f"Annotations must review {kind}")
        if "player_id" not in review:
            raise ValueError("Each review requires an explicit judged player_id")
        key = identity(review)
        if key in seen:
            raise ValueError("Duplicate player-specific annotation")
        seen.add(key)
        if key not in lookup:
            raise ValueError(f"Reviewed incident is absent from features: {key}")
        index, row = lookup[key]
        if review["raw_time"] != row["raw_time"]:
            raise ValueError("Review and feature replay timestamps differ")
        if require_source_ids and "source_event_ids" not in review:
            raise ValueError("Each review requires its source event IDs")
        if "source_event_ids" in review:
            source_ids = (
                json.loads(row[source_id_field])
                if multiple_source_ids
                else [row[source_id_field]]
            )
            if review["source_event_ids"] != source_ids:
                raise ValueError("Review and feature source events differ")
        if validate_review is not None:
            validate_review(review, row)
        if type(review["label"]) is not int or review["label"] not in (0, 1):
            raise ValueError("Review labels must be integer 0 or 1")
        if (
            not isinstance(review["engine_revision"], str)
            or not review["engine_revision"].strip()
        ):
            raise ValueError(
                "Review engine_revision must identify the reviewed extraction"
            )
        indices.append(index)
        labels.append(review["label"])
        revisions.append(review["engine_revision"])
        reasons.append(review.get("reject_reason"))
    prepared = table.take(indices)
    for name, values, data_type in (
        ("label", labels, pa.uint8()),
        ("annotation_engine_revision", revisions, pa.string()),
        ("reject_reason", reasons, pa.string()),
    ):
        if name in prepared.column_names:
            raise ValueError(
                f"Extracted table already contains a review column: {name}"
            )
        prepared = prepared.append_column(name, pa.array(values, type=data_type))
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(prepared, output, compression="snappy")
    return {
        "reviewed_rows": len(indices),
        "unreviewed_rows": table.num_rows - len(indices),
    }


def supports_schema(schema, base):
    return isinstance(schema, str) and (
        schema == base
        or re.fullmatch(rf"{re.escape(base)}\.[A-Za-z0-9_]+\.v1", schema) is not None
    )


def main(prepare, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("features", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.features, args.annotations, args.output)))
