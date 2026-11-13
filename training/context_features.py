"""Write typed context features with replay provenance and atomic publication."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def compute(
    paths,
    engine,
    output,
    *,
    kind,
    command,
    expected_sampling,
    participant_fields,
    integer_fields=(),
    source_id_field="source_event_id",
    multiple_source_ids=False,
    incident_fields=("source_event_id",),
):
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("No replay analyses selected")
    if len({path.absolute() for path in paths}) != len(paths):
        raise ValueError("Duplicate replay analysis selection")
    output = Path(output)
    if output.suffix != ".parquet":
        raise ValueError("Feature table must use a .parquet extension")
    records = []
    contract = None
    replays = []
    checksums = {}
    identities = set()
    for path in paths:
        with Path(path).open("rb") as analysis:
            result = subprocess.run(
                [str(engine), *(command(path) if callable(command) else command)],
                stdin=analysis,
                capture_output=True,
                check=False,
            )
        if result.returncode:
            raise ValueError(f"{path}: {result.stderr.decode().strip()}")
        batch = json.loads(result.stdout)
        current_contract = {
            "input_schema": batch["input_schema"],
            "sampling": batch["sampling"],
            "feature_definitions": batch["feature_definitions"],
        }
        if batch["sampling"] != expected_sampling:
            raise ValueError(f"Sampling configuration mismatch: {path}")
        if contract is None:
            contract = current_contract
        elif contract != current_contract:
            raise ValueError(f"Feature contract mismatch: {path}")
        checksum = batch["replay_sha256"]
        if checksum in checksums:
            previous, reference = checksums[checksum]
            if previous != batch:
                raise ValueError(f"Conflicting analyses for the same replay: {path}")
            reference["analysis_files"].append(str(path))
            reference["replay_ids"].append(path.stem)
            continue
        reference = {
            "analysis_files": [str(path)],
            "replay_ids": [path.stem],
            "replay_sha256": checksum,
            "engine_revision": batch["engine_revision"],
            "rows": len(batch["rows"]),
        }
        checksums[checksum] = batch, reference
        replays.append(reference)
        for row in batch["rows"]:
            player_id = json.dumps(row["player_id"], sort_keys=True)
            identity = (checksum, *(row[field] for field in incident_fields), player_id)
            if identity in identities:
                raise ValueError(f"Duplicate feature row identity: {path}")
            identities.add(identity)
            records.append(
                {
                    "replay_sha256": checksum,
                    "engine_revision": batch["engine_revision"],
                    source_id_field: (
                        json.dumps(row[source_id_field])
                        if multiple_source_ids
                        else row[source_id_field]
                    ),
                    "frame": row["frame"],
                    "raw_time": row["raw_time"],
                    "player_id": player_id,
                    **{
                        field: json.dumps(row[field], sort_keys=True)
                        for field in participant_fields
                    },
                    **{field: row[field] for field in integer_fields},
                    "samples": json.dumps(row["samples"], sort_keys=True),
                    **row["features"],
                }
            )
    if not records:
        raise ValueError("Selected analyses contain no feature rows")
    data_types = {"boolean": pa.bool_(), "f32": pa.float32(), "u32": pa.uint32()}
    fields = [
        pa.field(name, pa.string(), nullable=False)
        for name in ("replay_sha256", "engine_revision", source_id_field)
    ]
    fields.extend(
        [
            pa.field("frame", pa.uint64(), nullable=False),
            pa.field("raw_time", pa.float64(), nullable=False),
        ]
    )
    fields.extend(
        pa.field(name, pa.uint64(), nullable=False) for name in integer_fields
    )
    fields.extend(
        pa.field(name, pa.string(), nullable=False)
        for name in ("player_id", *participant_fields, "samples")
    )
    fields.extend(
        pa.field(
            feature["name"],
            data_types[feature["data_type"]],
            nullable=feature["nullable"],
        )
        for feature in contract["feature_definitions"]
    )
    metadata = {
        "schema_version": 1,
        "kind": kind,
        **contract,
        "replays": replays,
    }
    schema = pa.schema(
        fields,
        metadata={b"rl_mistake_analysis_features": json.dumps(metadata).encode()},
    )
    table = pa.Table.from_pylist(records, schema=schema)
    table.validate(full=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / "features.parquet"
        pq.write_table(table, temporary, compression="snappy")
        with temporary.open("rb") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, output)
    return {
        "analyses": len(paths),
        "replays": len(replays),
        "rows": len(records),
        "features": len(contract["feature_definitions"]),
    }


def main(compute, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("analyses", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--replay-ids", type=Path)
    args = parser.parse_args()
    if args.replay_ids:
        ids = json.loads(args.replay_ids.read_text())
        if not isinstance(ids, list) or any(
            not isinstance(value, str) or not value or Path(value).name != value
            for value in ids
        ):
            raise ValueError(
                "Replay IDs must be a JSON array of filenames without paths"
            )
        paths = [args.analyses / f"{value}.json" for value in ids]
    else:
        paths = sorted(args.analyses.glob("*.json"))
    print(json.dumps(compute(paths, args.engine, args.sampling, args.output)))
