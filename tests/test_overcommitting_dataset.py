import copy
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training import context_features, dataset, overcommitting_dataset
from training.overcommitting_features import compute


def inputs(tmp_path, monkeypatch):
    analysis = tmp_path / "replay.json"
    analysis.write_text("{}")
    sampling = tmp_path / "sampling.json"
    sampling.write_text('{"method":"native_interval_boundaries"}')
    candidates = tmp_path / "candidates.json"
    candidates.write_text("[{}]")
    rows = []
    reviews = []
    for index, (frame, end_frame, passage_frame) in enumerate(
        ((10, 20, 25), (30, 40, 45))
    ):
        samples = {
            "start_frame": frame,
            "start_time": frame / 10,
            "end_frame": end_frame,
            "end_time": end_frame / 10,
            "passage_frame": passage_frame,
            "passage_time": passage_frame / 10,
        }
        player = {"Epic": "player"}
        source_ids = ["touch:1:0", "touch:50:0"]
        rows.append(
            {
                "frame": frame,
                "end_frame": end_frame,
                "passage_frame": passage_frame,
                "raw_time": frame / 10,
                "player_id": player,
                "source_event_ids": source_ids,
                "samples": samples,
                "features": {
                    "opponent_contact": None if index == 0 else False,
                    "distance": None if index == 0 else 0.0,
                    "touch_count": index,
                },
            }
        )
        reviews.append(
            {
                "kind": "overcommitting",
                "replay_sha256": "a" * 64,
                "frame": frame,
                "end_frame": end_frame,
                "player_id": player,
                "raw_time": frame / 10,
                "source_event_ids": source_ids,
                "evidence": {**samples, "player": player},
                "engine_revision": "review-supplier",
                "label": index,
            }
        )
    batch = {
        "input_schema": "overcommitting.context.v1",
        "sampling": {"method": "native_interval_boundaries"},
        "replay_sha256": "a" * 64,
        "engine_revision": "feature-supplier",
        "feature_definitions": [
            {"name": "opponent_contact", "data_type": "boolean", "nullable": True},
            {"name": "distance", "data_type": "f32", "nullable": True},
            {"name": "touch_count", "data_type": "u32", "nullable": False},
        ],
        "rows": rows,
    }
    calls = []

    def run(arguments, **options):
        calls.append(arguments)
        assert options["stdin"].read() == b"{}"
        return SimpleNamespace(returncode=0, stdout=json.dumps(batch).encode())

    monkeypatch.setattr(context_features.subprocess, "run", run)
    features = tmp_path / "features.parquet"
    assert compute(
        [analysis],
        "engine",
        sampling,
        features,
        candidate_batches={analysis: candidates},
    ) == {"analyses": 1, "replays": 1, "rows": 2, "features": 3}
    assert calls == [
        [
            "engine",
            "overcommitting-features",
            str(sampling),
            "--candidates",
            str(candidates),
        ]
    ]
    annotations = tmp_path / "reviews.json"
    annotations.write_text(json.dumps({"schema_version": 1, "annotations": reviews}))
    return features, annotations, batch


def test_distinct_intervals_share_sources_and_keep_nulls_separate_from_false(
    tmp_path, monkeypatch
):
    features, annotations, _ = inputs(tmp_path, monkeypatch)
    table, contract = overcommitting_dataset.load(features)
    assert table["end_frame"].type == pa.uint64()
    assert table["passage_frame"].type == pa.uint64()
    assert "source_event_id" not in table.column_names
    assert (
        table["source_event_ids"].to_pylist()
        == [json.dumps(["touch:1:0", "touch:50:0"])] * 2
    )
    output = tmp_path / "labeled.parquet"
    assert overcommitting_dataset.prepare(features, annotations, output) == {
        "reviewed_rows": 2,
        "unreviewed_rows": 0,
    }
    prepared, prepared_contract = overcommitting_dataset.load(output, labeled=True)
    assert prepared_contract == contract
    assert prepared["opponent_contact"].to_pylist() == [None, False]
    assert prepared["distance"].to_pylist() == [None, 0.0]
    assert prepared["annotation_engine_revision"].to_pylist() == ["review-supplier"] * 2
    values, labels, groups = overcommitting_dataset.arrays(prepared, contract)
    np.testing.assert_equal(values, [[np.nan, np.nan, 0], [0, 0, 1]])
    assert labels.tolist() == [0, 1]
    assert groups.tolist() == ["a" * 64] * 2
    generic, generic_contract = dataset.load(output)
    generic_values, _, _ = dataset.arrays(generic, generic_contract)
    np.testing.assert_equal(generic_values, values)
    selected = dataset.select_features(
        generic_contract,
        ["distance", "opponent_contact"],
        "overcommitting.context.v1.selected.v1",
    )
    assert selected["feature_names"] == ["distance", "opponent_contact"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame", 11),
        ("end_frame", 21),
        ("raw_time", 1.01),
        ("player_id", {"Epic": "other"}),
        ("source_event_ids", ["touch:1:0"]),
        ("source_event_ids", ["touch:50:0", "touch:1:0"]),
    ],
)
def test_different_incident_or_provenance_never_joins(
    tmp_path, monkeypatch, field, value
):
    features, annotations, _ = inputs(tmp_path, monkeypatch)
    document = json.loads(annotations.read_text())
    document["annotations"][0][field] = value
    annotations.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        overcommitting_dataset.prepare(
            features, annotations, tmp_path / "labeled.parquet"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_frame", 11),
        ("end_frame", 21),
        ("passage_frame", 26),
        ("start_time", 1.01),
        ("end_time", 2.01),
        ("passage_time", 2.51),
        ("player", {"Epic": "other"}),
    ],
)
def test_boundary_evidence_must_match_native_samples(
    tmp_path, monkeypatch, field, value
):
    features, annotations, _ = inputs(tmp_path, monkeypatch)
    document = json.loads(annotations.read_text())
    document["annotations"][0]["evidence"][field] = value
    annotations.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="interval evidence differ"):
        overcommitting_dataset.prepare(
            features, annotations, tmp_path / "labeled.parquet"
        )


@pytest.mark.parametrize("field", ["source_event_ids", "evidence"])
def test_missing_interval_provenance_is_rejected(tmp_path, monkeypatch, field):
    features, annotations, _ = inputs(tmp_path, monkeypatch)
    document = json.loads(annotations.read_text())
    del document["annotations"][0][field]
    annotations.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        overcommitting_dataset.prepare(
            features, annotations, tmp_path / "labeled.parquet"
        )


def test_duplicate_interval_is_rejected_even_with_different_source_ids(
    tmp_path, monkeypatch
):
    features, _, _ = inputs(tmp_path, monkeypatch)
    table = pq.read_table(features)
    duplicate = copy.deepcopy(table.slice(0, 1).to_pylist()[0])
    duplicate["source_event_ids"] = json.dumps(["touch:9:0"])
    pq.write_table(
        pa.Table.from_pylist([*table.to_pylist(), duplicate], schema=table.schema),
        features,
    )
    with pytest.raises(ValueError, match="Duplicate player-specific interval"):
        overcommitting_dataset.load(features)


def test_explicit_candidate_batches_must_match_all_selected_analyses(tmp_path):
    with pytest.raises(ValueError, match="cover exactly"):
        compute(
            [tmp_path / "replay.json"],
            "engine",
            tmp_path / "sampling.json",
            tmp_path / "features.parquet",
            candidate_batches={},
        )
