"""Join reviewed overcommitting intervals by exact native boundaries and player."""

import json
import math
import re

import pyarrow as pa

from training import context_dataset

METADATA_KEY = context_dataset.METADATA_KEY
CONTEXT_SCHEMA = "overcommitting.context.v1"
IDENTITY_FIELDS = ("replay_sha256", "frame", "end_frame", "player_id")
arrays = context_dataset.arrays


def identity(row):
    checksum = row["replay_sha256"]
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Invalid replay fingerprint")
    if any(
        type(row[name]) is not int or row[name] < 0 for name in ("frame", "end_frame")
    ):
        raise ValueError("Interval frames must be nonnegative integers")
    if row["end_frame"] <= row["frame"]:
        raise ValueError("The interval must end after its opening frame")
    player = row["player_id"]
    if not isinstance(player, dict) or not player:
        raise ValueError("An explicit player identity is required")
    return checksum, row["frame"], row["end_frame"], json.dumps(player, sort_keys=True)


def supports_schema(schema):
    return context_dataset.supports_schema(schema, CONTEXT_SCHEMA)


def load(path, *, labeled=False):
    table, contract = context_dataset.load(
        path,
        kind="overcommitting",
        supports_schema=supports_schema,
        identity_fields=IDENTITY_FIELDS,
        json_fields=("player_id",),
        identity=identity,
        duplicate_message="Duplicate player-specific interval",
        labeled=labeled,
        source_id_field="source_event_ids",
        multiple_source_ids=True,
        validation_fields=("passage_frame", "samples"),
    )
    for name in ("end_frame", "passage_frame"):
        if not pa.types.is_integer(table[name].type):
            raise ValueError("Interval and passage frames must be integers")
    if any(
        row["passage_frame"] < row["end_frame"]
        for row in table.select(["end_frame", "passage_frame"]).to_pylist()
    ):
        raise ValueError("Ball passage cannot precede the approach end")
    for row in table.select(
        ["frame", "end_frame", "passage_frame", "raw_time", "samples"]
    ).to_pylist():
        samples = json.loads(row["samples"])
        required = (
            "start_frame",
            "start_time",
            "end_frame",
            "end_time",
            "passage_frame",
            "passage_time",
        )
        if not isinstance(samples, dict) or any(
            name not in samples for name in required
        ):
            raise ValueError("Interval samples require exact boundary frames and times")
        if any(
            type(samples[name]) is not int
            for name in required
            if name.endswith("_frame")
        ):
            raise ValueError("Interval sample frames must be integers")
        for field, column in (
            ("start_frame", "frame"),
            ("end_frame", "end_frame"),
            ("passage_frame", "passage_frame"),
            ("start_time", "raw_time"),
        ):
            if samples[field] != row[column]:
                raise ValueError(f"Interval samples and columns differ: {field}")
        for field in ("start_time", "end_time", "passage_time"):
            if type(samples[field]) not in (int, float) or not math.isfinite(
                samples[field]
            ):
                raise ValueError("Interval sample times must be finite numbers")
        if (
            samples["end_time"] < samples["start_time"]
            or samples["passage_time"] < samples["end_time"]
        ):
            raise ValueError("Interval boundary times cannot run backward")
    return table, contract


def validate_review(review, row):
    evidence = review.get("evidence")
    samples = json.loads(row["samples"])
    required = (
        "start_frame",
        "end_frame",
        "passage_frame",
        "start_time",
        "end_time",
        "passage_time",
        "player",
    )
    if not isinstance(evidence, dict) or any(name not in evidence for name in required):
        raise ValueError("Interval reviews require exact boundary and player evidence")
    if any(
        type(evidence[name]) is not int for name in required if name.endswith("_frame")
    ):
        raise ValueError("Interval evidence frames must be integers")
    for field in required[:-1]:
        if evidence[field] != samples[field]:
            raise ValueError(f"Review and feature interval evidence differ: {field}")
    if evidence["player"] != row["player_id"]:
        raise ValueError("Review and feature interval evidence differ: player")


def prepare(features, annotations, output):
    return context_dataset.prepare(
        features,
        annotations,
        output,
        kind="overcommitting",
        load=load,
        identity_fields=IDENTITY_FIELDS,
        json_fields=("player_id",),
        identity=identity,
        source_id_field="source_event_ids",
        multiple_source_ids=True,
        require_source_ids=True,
        validation_fields=("passage_frame", "samples"),
        validate_review=validate_review,
    )


if __name__ == "__main__":
    context_dataset.main(prepare, __doc__)
