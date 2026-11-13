"""Verify participant identities across public annotations, training, and inference."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from training.bump_dataset import prepare
from training.bump_features import compute
from training.dataset import arrays, load, select_features
from training.export import model_entry, write_manifest

ROOT = Path(__file__).resolve().parents[1]
SAMPLING = ROOT / "examples/bump_sampling.json"


def analysis():
    players = [{"Steam": "1"}, {"Steam": "2"}]

    def body(x):
        return {
            "location": {"x": x, "y": 0, "z": 17},
            "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "linear_velocity": {"x": 0, "y": 0, "z": 0},
        }

    bump = {
        "frame": 2,
        "time": 0.25,
        "initiator": players[0],
        "victim": players[1],
        "is_team_bump": True,
        "initiator_is_team_0": True,
        "victim_is_team_0": True,
        "contact_distance": 5,
        "confidence": 0.9,
        "closing_speed": 400,
        "victim_impulse": 200,
        "strength": 0.2,
        "initiator_position": [0, 0, 17],
        "victim_position": [100, 0, 17],
    }
    return {
        "name": "subtr-actor",
        "schema_version": 1,
        "status": "ok",
        "revision": "participant-fixture",
        "replay_sha256": "a" * 64,
        "outputs": {
            "events": {
                "events": {
                    "events": [
                        {
                            "meta": {"id": "bump:2", "stream": "bump"},
                            "payload": {"kind": "bump", "payload": bump},
                        },
                        {
                            "meta": {"id": "possession:0", "stream": "possession"},
                            "payload": {
                                "kind": "possession",
                                "payload": {
                                    "time": 0.0,
                                    "end_time": 0.875,
                                    "possession_state": "neutral",
                                },
                            },
                        },
                    ]
                },
                "accumulation_tracks": [
                    {
                        "player_id": player,
                        "quantity": "boost_amount",
                        "points": [{"frame": 0, "value": 100}],
                    }
                    for player in players
                ],
            },
            "replay_data": {
                "frame_data": {
                    "metadata_frames": [{"time": frame / 8} for frame in range(8)],
                    "ball_data": {
                        "frames": [
                            {"Data": {"rigid_body": body(1000)}} for _ in range(8)
                        ]
                    },
                    "players": [
                        [
                            player,
                            {
                                "frames": [
                                    {
                                        "Data": {
                                            "rigid_body": body(index * 100),
                                            "dodge_active": False,
                                        }
                                    }
                                    for _ in range(8)
                                ]
                            },
                        ]
                        for index, player in enumerate(players)
                    ],
                },
                "boost_pad_events": [],
                "boost_pads": [],
            },
        },
    }


def run(engine, *args, document=None):
    return json.loads(
        subprocess.run(
            [str(engine), *map(str, args)],
            input=json.dumps(document) if document is not None else None,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )


def model(path, names, flag_name, missing_name):
    inputs = [
        helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, len(names)])
    ]
    outputs = [
        helper.make_tensor_value_info("probability", TensorProto.FLOAT, [None, 1])
    ]
    initializers = [
        helper.make_tensor(
            "role_index", TensorProto.INT64, [1], [names.index(flag_name)]
        ),
        helper.make_tensor(
            "missing_index",
            TensorProto.INT64,
            [1],
            [names.index(missing_name)],
        ),
        helper.make_tensor("one", TensorProto.FLOAT, [1], [1]),
    ]
    nodes = [
        helper.make_node("Gather", ["features", "role_index"], ["role"], axis=1),
        helper.make_node("Sub", ["one", "role"], ["victim"]),
        helper.make_node(
            "Gather", ["features", "missing_index"], ["measurement"], axis=1
        ),
        helper.make_node("IsNaN", ["measurement"], ["missing"]),
        helper.make_node("Cast", ["missing"], ["missing_float"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["victim", "missing_float"], ["probability"]),
    ]
    graph = helper.make_graph(nodes, "participant-test", inputs, outputs, initializers)
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
        ),
        path,
    )


@pytest.mark.parametrize("reverse_repeat", [False, True])
@pytest.mark.parametrize(
    "subset_schema",
    [None, "bumping_teammate.context.v1", "bumping_teammate.context.v1.expanded90.v1"],
)
def test_same_rows_are_reviewed_prepared_and_scored(
    engine, tmp_path, reverse_repeat, subset_schema
):
    document = analysis()
    if reverse_repeat:
        events = document["outputs"]["events"]["events"]["events"]
        repeated = deepcopy(events[0])
        repeated["meta"]["id"] = "bump:3"
        bump = repeated["payload"]["payload"]
        bump["frame"], bump["time"] = 3, 0.375
        for left, right in (
            ("initiator", "victim"),
            ("initiator_position", "victim_position"),
        ):
            bump[left], bump[right] = bump[right], bump[left]
        events.insert(0, repeated)
    events = {
        key: document[key]
        for key in ("name", "schema_version", "status", "revision", "replay_sha256")
    }
    events["events"] = document["outputs"]["events"]["events"]["events"]
    batch = run(engine, "candidates", "bumping_teammate", document=events)[0]
    assert len(batch["candidates"]) == 2
    assert len(batch["feature_names"]) == 8
    assert "sampling" not in batch and "feature_definitions" not in batch
    assert batch["candidates"][0]["features"] == batch["candidates"][1]["features"]
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(document))
    features = tmp_path / "features.parquet"
    compute([source], engine, SAMPLING, features)
    reviews = tmp_path / "reviews.json"
    reviews.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "replays": [{"sha256": batch["replay_sha256"]}],
                "annotations": [
                    {
                        "replay_sha256": batch["replay_sha256"],
                        "kind": candidate["kind"],
                        "frame": candidate["frame"],
                        "initiator": candidate["evidence"]["initiator"],
                        "victim": candidate["evidence"]["victim"],
                        "player_id": candidate["player_id"],
                        "raw_time": candidate["raw_time"],
                        "evidence": candidate["evidence"],
                        "label": int(candidate["player_id"] == {"Steam": "2"}),
                        "engine_revision": batch["engine_revision"],
                        "source_event_ids": candidate["source_event_ids"],
                    }
                    for candidate in reversed(batch["candidates"])
                ],
            }
        )
    )
    assert run(engine, "validate-annotations", reviews) == {
        "replays": 1,
        "annotations": 2,
    }
    training = tmp_path / "training.parquet"
    assert prepare(features, reviews, training)["reviewed_rows"] == 2
    table, contract = load(training)
    X, y, groups = arrays(table, contract)
    by_player = {
        json.loads(player)["Steam"]: index
        for index, player in enumerate(table["player_id"].to_pylist())
    }
    measurements = run(engine, "bump-features", SAMPLING, document=document)
    assert len(measurements["feature_definitions"]) == 105
    for candidate in batch["candidates"]:
        index = by_player[candidate["player_id"]["Steam"]]
        row = next(
            row
            for row in measurements["rows"]
            if row["player_id"] == candidate["player_id"]
        )
        np.testing.assert_equal(
            X[index],
            np.array(
                [row["features"][name] for name in contract["feature_names"]],
                dtype=np.float32,
            ),
        )
        assert y[index] == int(candidate["player_id"] == {"Steam": "2"})
    assert groups.tolist() == ["a" * 64] * 2
    selection = {}
    if subset_schema is not None:
        names = ["p_velocity_delta_up", "p_is_initiator"]
        contract = select_features(contract, names, subset_schema)
        selection = {"feature_names": names, "input_schema": subset_schema}
    model(
        tmp_path / "model.onnx",
        contract["feature_names"],
        "p_is_initiator",
        "p_velocity_delta_up",
    )
    (tmp_path / "evaluation.json").write_text("{}")
    entry = model_entry(
        tmp_path,
        training,
        version="test",
        model_file="model.onnx",
        evaluation_file="evaluation.json",
        keep_threshold=0.5,
        source_revision="participant-test",
        status="experimental",
        **selection,
    )
    assert entry["sampling"] == json.loads(SAMPLING.read_text())
    write_manifest(tmp_path, "test", [entry])
    predictions = run(
        engine, "score", tmp_path, "bumping_teammate", "test", document=document
    )
    assert len(predictions["predictions"]) == 2
    assert predictions["input_schema"] == batch["input_schema"]
    assert predictions["feature_names"] == batch["feature_names"]
    assert predictions["model_input_schema"] == contract["input_schema"]
    assert predictions["model_feature_names"] == contract["feature_names"]
    for prediction in predictions["predictions"]:
        candidate = prediction["candidate"]
        original = next(
            row
            for row in batch["candidates"]
            if row["player_id"] == candidate["player_id"]
        )
        assert candidate["features"] == original["features"]
        assert len(candidate["features"]) == 8
        expected = int(candidate["player_id"] == {"Steam": "2"})
        assert prediction["score"] == pytest.approx(expected)
        assert prediction["keep"] == bool(expected)
    del entry["sampling"]
    write_manifest(tmp_path, "test", [entry])
    with pytest.raises(subprocess.CalledProcessError):
        run(engine, "score", tmp_path, "bumping_teammate", "test", document=document)


def test_bang_context_models_use_the_shared_join_export_and_scorer(
    engine, combined_analysis, tmp_path
):
    from training.bang_dataset import prepare as prepare_bang
    from training.bang_features import compute as compute_bang

    extracted = tmp_path / "features.parquet"
    compute_bang(
        [combined_analysis], engine, ROOT / "examples/bang_sampling.json", extracted
    )
    import pyarrow.parquet as pq

    table = pq.read_table(extracted)
    reviews = []
    for index, row in enumerate(table.to_pylist()):
        reviews.append(
            {
                "kind": "bang_with_time",
                "replay_sha256": row["replay_sha256"],
                "frame": row["frame"],
                "touch_id": row["touch_id"],
                "player_id": json.loads(row["player_id"]),
                "raw_time": row["raw_time"],
                "engine_revision": row["engine_revision"],
                "label": index % 2,
            }
        )
    annotations = tmp_path / "reviews.json"
    annotations.write_text(json.dumps({"schema_version": 1, "annotations": reviews}))
    training_table = tmp_path / "labeled.parquet"
    prepare_bang(extracted, annotations, training_table)
    prepared, source_contract = load(training_table)
    names = ["second_touch_outgoing_ball_speed", "candidate_dodging"]
    contract = select_features(
        source_contract, names, "bang_with_time.context.v1.outcome.v1"
    )
    features, _, _ = arrays(prepared, contract)
    model(
        tmp_path / "model.onnx",
        names,
        "candidate_dodging",
        "second_touch_outgoing_ball_speed",
    )
    (tmp_path / "evaluation.json").write_text("{}")
    entry = model_entry(
        tmp_path,
        training_table,
        version="test",
        model_file="model.onnx",
        evaluation_file="evaluation.json",
        keep_threshold=0.5,
        source_revision="fixture",
        status="experimental",
        feature_names=names,
        input_schema=contract["input_schema"],
    )
    write_manifest(tmp_path, "test", [entry])
    document = json.loads(combined_analysis.read_text())
    result = run(engine, "score", tmp_path, "bang_with_time", "test", document=document)
    expected = (1 - features[:, 1]) * np.isnan(features[:, 0])
    np.testing.assert_array_equal(
        [row["score"] for row in result["predictions"]], expected
    )
    assert result["model_feature_names"] == names
    assert result["sampling"] == {"method": "native_touch_stages"}
    assert [
        row["candidate"]["evidence"]["touch_id"] for row in result["predictions"]
    ] == table["touch_id"].to_pylist()
    entry["sampling"] = {"method": "first_acceleration_peak_then_minimum"}
    write_manifest(tmp_path, "test", [entry])
    rejected = subprocess.run(
        [str(engine), "score", str(tmp_path), "bang_with_time", "test"],
        input=combined_analysis.read_bytes(),
        capture_output=True,
    )
    assert rejected.returncode == 1
    assert b"sampling method" in rejected.stderr


def test_overcommitting_export_retains_sampling_for_selected_inputs(
    engine, tmp_path, monkeypatch
):
    import pyarrow as pa

    from training import export

    contract = {
        "schema_version": 1,
        "kind": "overcommitting",
        "input_schema": "overcommitting.context.v1",
        "sampling": {"method": "native_interval_boundaries"},
        "feature_names": ["approach_duration_s", "goal_side_teammate_count"],
        "feature_definitions": [
            {
                "name": "approach_duration_s",
                "data_type": "f32",
                "nullable": True,
                "source": "native fixture",
            },
            {
                "name": "goal_side_teammate_count",
                "data_type": "u32",
                "nullable": True,
                "source": "native fixture",
            },
        ],
    }
    table = pa.table({"engine_revision": ["fixture"]})
    monkeypatch.setattr(export, "load", lambda _path: (table, contract))
    graph = helper.make_model(
        helper.make_graph(
            [helper.make_node("Sigmoid", ["features"], ["score"])],
            "duration_probability",
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 1])],
            [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.save(graph, tmp_path / "model.onnx")
    (tmp_path / "evaluation.json").write_text("{}")
    entry = export.model_entry(
        tmp_path,
        tmp_path / "training.parquet",
        version="test",
        model_file="model.onnx",
        evaluation_file="evaluation.json",
        keep_threshold=0.5,
        source_revision="fixture",
        status="experimental",
        feature_names=["approach_duration_s"],
        input_schema="overcommitting.context.v1.release_test.v1",
        candidate_schema="overcommitting.candidates.v1",
    )
    assert entry["sampling"] == {"method": "native_interval_boundaries"}
    assert entry["feature_names"] == ["approach_duration_s"]
    write_manifest(tmp_path, "test", [entry])
    assert run(engine, "validate-manifest", tmp_path)["models"] == 1
    del entry["sampling"]
    write_manifest(tmp_path, "test", [entry])
    rejected = subprocess.run(
        [str(engine), "validate-manifest", str(tmp_path)], capture_output=True
    )
    assert rejected.returncode == 1
    assert b"feature sampling" in rejected.stderr
