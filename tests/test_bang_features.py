import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training import bang_dataset, dataset
from training.bang_dataset import prepare
from training.bang_features import compute

SAMPLING = Path(__file__).resolve().parents[1] / "examples/bang_sampling.json"


def extract(engine, path, *options):
    with path.open("rb") as source:
        return json.loads(
            subprocess.run(
                [str(engine), "bang-features", str(SAMPLING), *options],
                stdin=source,
                capture_output=True,
                check=True,
            ).stdout
        )


def test_real_replay_preserves_native_candidate_identity_and_complete_contract(
    engine, combined_analysis, tmp_path
):
    batch = extract(engine, combined_analysis)
    document = json.loads(combined_analysis.read_text())
    selection = json.loads(
        subprocess.run(
            [str(engine), "candidates", "bang_with_time"],
            input=json.dumps(document).encode(),
            capture_output=True,
            check=True,
        ).stdout
    )
    assert len(batch["rows"]) == len(selection[0]["candidates"]) == 57
    assert len(batch["feature_definitions"]) == 163
    frames = document["outputs"]["replay_data"]["frame_data"]
    events = document["outputs"]["events"]["events"]["events"]
    touches = {
        event["meta"]["id"]: event["payload"]["payload"]
        for event in events
        if event["meta"]["stream"] == "touch"
    }
    fifties = [
        event["payload"]["payload"]
        for event in events
        if event["meta"]["stream"] == "fifty_fifty"
    ]
    players = {json.dumps(player): data for player, data in frames["players"]}
    for candidate, row in zip(selection[0]["candidates"], batch["rows"], strict=True):
        assert row["source_event_id"] == candidate["source_event_ids"][0]
        for key in ("frame", "raw_time", "player_id"):
            assert row[key] == candidate[key]
        assert row["touch_id"] == candidate["evidence"]["touch_id"]
        assert len(row["features"]) == 163
        samples = row["samples"]
        assert row["features"]["first_touch_opponent"] is True
        checked_touches = [
            touches[source_id]
            for source_id in [
                row["source_event_id"],
                *samples["first_touch"]["source_event_ids"],
            ]
        ]
        for touch in checked_touches:
            assert not any(
                fifty["start_frame"] <= touch["frame"] <= fifty["resolve_frame"]
                for fifty in fifties
            )
            assert {"group": "contested", "value": "contested"} not in touch["tags"]
        opponent = samples["closest_opponent_id"]
        if opponent is not None:
            before = samples["before_frame"]
            car = players[json.dumps(opponent)]["frames"][before]["Data"]
            ball = frames["ball_data"]["frames"][before]["Data"]["rigid_body"]
            own_team_zero = any(
                player["remote_id"] == row["player_id"]
                for player in document["outputs"]["events"]["replay_meta"]["team_zero"]
            )
            sign = 1 if own_team_zero else -1
            for vector, native in [
                ("position_relative_ball", "location"),
                ("velocity_relative_ball", "linear_velocity"),
            ]:
                if car["rigid_body"][native] is not None and ball[native] is not None:
                    delta = {
                        axis: car["rigid_body"][native][axis] - ball[native][axis]
                        for axis in ("x", "y", "z")
                    }
                    np.testing.assert_allclose(
                        [
                            row["features"][f"closest_opponent_{vector}_{axis}"]
                            for axis in ("toward", "side", "up")
                        ],
                        [sign * delta["y"], -sign * delta["x"], delta["z"]],
                        rtol=1e-6,
                        atol=1e-5,
                    )
            assert row["features"][
                "closest_opponent_boost_before_pct"
            ] == pytest.approx(
                next(
                    point["value"]
                    for track in document["outputs"]["events"]["accumulation_tracks"]
                    if track["player_id"] == opponent
                    and track["quantity"] == "boost_amount"
                    for point in reversed(track["points"])
                    if point["frame"] <= before
                )
                * 100
                / 255,
                rel=1e-6,
                abs=1e-5,
            )
        for stage in ("first_touch", "second_touch"):
            sample = samples[stage]
            if sample is None:
                assert row["features"][f"{stage}_delay_s"] is None
                assert row["features"][f"subject_ball_distance_at_{stage}"] is None
            else:
                assert row["frame"] < sample["frame"]
                assert row["raw_time"] <= sample["raw_time"] <= samples["play_end"]
                assert sample["source_event_ids"]
        if samples["second_touch"]:
            assert samples["first_touch"]["frame"] < samples["second_touch"]["frame"]
    subject = batch["rows"][0]["player_id"]
    selected = extract(engine, combined_analysis, "--player", json.dumps(subject))
    assert selected["rows"] == [
        row for row in batch["rows"] if row["player_id"] == subject
    ]
    contract = json.loads(
        subprocess.run(
            [
                str(engine),
                "contract",
                "bang_with_time",
                "--schema",
                "bang_with_time.context.v1",
            ],
            capture_output=True,
            check=True,
        ).stdout
    )
    assert contract["feature_definitions"] == batch["feature_definitions"]


def test_parquet_reviews_and_training_arrays_preserve_types_order_and_nulls(
    engine, combined_analysis, tmp_path
):
    output = tmp_path / "features.parquet"
    alias = tmp_path / "alias.json"
    shutil.copyfile(combined_analysis, alias)
    assert compute([combined_analysis, alias], engine, SAMPLING, output) == {
        "analyses": 2,
        "replays": 1,
        "rows": 57,
        "features": 163,
    }
    table, contract = bang_dataset.load(output)
    assert contract["replays"][0]["replay_ids"] == ["sample", "alias"]
    assert contract["sampling"] == {"method": "native_touch_stages"}
    for name, data_type in [
        ("touch_id", pa.uint64()),
        ("candidate_dodging", pa.bool_()),
        ("first_touch_delay_s", pa.float32()),
        ("touches_before_next_goal", pa.uint32()),
        ("approaching_teammate_count", pa.uint32()),
        ("approaching_opponent_count", pa.uint32()),
        ("subject_consecutive_touches_before_candidate", pa.uint32()),
        ("closest_opponent_boost_before_pct", pa.float32()),
    ]:
        assert table[name].type == data_type
    field = table.schema.field("second_touch_delay_s")
    delays = table[field.name].to_pylist()
    delays[0] = None
    table = table.set_column(
        table.schema.get_field_index(field.name),
        field,
        pa.array(delays, type=field.type),
    )
    pq.write_table(table, output)
    reviews = []
    for label, row in enumerate(table.slice(0, 2).to_pylist()):
        reviews.append(
            {
                "kind": "bang_with_time",
                "replay_sha256": row["replay_sha256"],
                "frame": row["frame"],
                "touch_id": row["touch_id"],
                "player_id": json.loads(row["player_id"]),
                "raw_time": row["raw_time"],
                "source_event_ids": [row["source_event_id"]],
                "engine_revision": row["engine_revision"],
                "label": label,
            }
        )
    annotations = tmp_path / "reviews.json"
    annotations.write_text(json.dumps({"schema_version": 1, "annotations": reviews}))
    labeled = tmp_path / "labeled.parquet"
    assert prepare(output, annotations, labeled) == {
        "reviewed_rows": 2,
        "unreviewed_rows": 55,
    }
    prepared, training_contract = dataset.load(labeled)
    features, labels, groups = dataset.arrays(prepared, training_contract)
    assert features.shape == (2, 163)
    assert features.dtype == np.float32
    assert np.isnan(
        features[0, training_contract["feature_names"].index("second_touch_delay_s")]
    )
    assert labels.tolist() == [0, 1]
    assert len(set(groups)) == 1
    assert prepared.schema.metadata == table.schema.metadata
    assert training_contract["feature_names"] == [
        feature["name"] for feature in contract["feature_definitions"]
    ]


def test_failed_or_conflicting_extraction_keeps_saved_output(
    engine, combined_analysis, tmp_path
):
    output = tmp_path / "features.parquet"
    output.write_bytes(b"saved table")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}")
    with pytest.raises(ValueError, match="invalid.json"):
        compute([combined_analysis, invalid], engine, SAMPLING, output)
    assert output.read_bytes() == b"saved table"
    document = json.loads(combined_analysis.read_text())
    document["revision"] = "different supplier revision"
    conflicting = tmp_path / "conflicting.json"
    conflicting.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Conflicting analyses"):
        compute([combined_analysis, conflicting], engine, SAMPLING, output)
    assert output.read_bytes() == b"saved table"


def test_reader_rejects_duplicate_identities_and_wrong_feature_types(
    engine, combined_analysis, tmp_path
):
    output = tmp_path / "features.parquet"
    compute([combined_analysis], engine, SAMPLING, output)
    table = pq.read_table(output)
    pq.write_table(pa.concat_tables([table, table]), output)
    with pytest.raises(ValueError, match="Duplicate touch identity"):
        bang_dataset.load(output)
    table = table.set_column(
        table.schema.get_field_index("candidate_dodging"),
        "candidate_dodging",
        pa.array([0] * table.num_rows, type=pa.uint32()),
    )
    pq.write_table(table, output)
    with pytest.raises(ValueError, match="Feature type differs"):
        bang_dataset.load(output)
