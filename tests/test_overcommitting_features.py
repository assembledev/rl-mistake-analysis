import copy
import json
import subprocess
from pathlib import Path

import pytest

from training import overcommitting_dataset
from training.overcommitting_features import compute

SAMPLING = Path(__file__).resolve().parents[1] / "examples/overcommitting_sampling.json"


def run(engine, analysis, *arguments):
    with analysis.open("rb") as source:
        return subprocess.run(
            [str(engine), *arguments],
            stdin=source,
            capture_output=True,
            check=False,
        )


def extract(engine, analysis, *options):
    result = run(engine, analysis, "overcommitting-features", str(SAMPLING), *options)
    assert result.returncode == 0, result.stderr.decode()
    return json.loads(result.stdout)


def save_candidates(engine, analysis, path):
    result = run(engine, analysis, "candidates", "overcommitting")
    assert result.returncode == 0, result.stderr.decode()
    path.write_bytes(result.stdout)
    (selection,) = json.loads(result.stdout)
    assert selection["candidates"], (
        "The real replay must exercise nonempty overcommitting intervals"
    )
    return selection


def test_real_replay_preserves_full_interval_identity_and_native_stages(
    engine, combined_analysis, tmp_path
):
    supplied = tmp_path / "candidates.json"
    selection = save_candidates(engine, combined_analysis, supplied)
    batch = extract(engine, combined_analysis)
    assert len(batch["rows"]) == len(selection["candidates"])
    assert batch["sampling"] == {"method": "native_interval_boundaries"}
    assert batch["input_schema"] == "overcommitting.context.v1"
    definitions = batch["feature_definitions"]
    names = {feature["name"] for feature in definitions}
    assert names and len(names) == len(definitions)
    frames = json.loads(combined_analysis.read_text())["outputs"]["replay_data"][
        "frame_data"
    ]["metadata_frames"]
    assert any(len(row["source_event_ids"]) > 1 for row in batch["rows"])
    for candidate, row in zip(selection["candidates"], batch["rows"], strict=True):
        evidence = candidate["evidence"]
        assert set(row["features"]) == names
        for field in ("frame", "raw_time", "player_id", "source_event_ids"):
            assert row[field] == candidate[field]
        assert "source_event_id" not in row
        for field in ("end_frame", "passage_frame"):
            assert row[field] == evidence[field]
        for stage in ("start", "end", "passage"):
            frame = evidence[f"{stage}_frame"]
            assert row["samples"][f"{stage}_frame"] == frame
            assert (
                row["samples"][f"{stage}_time"]
                == evidence[f"{stage}_time"]
                == frames[frame]["time"]
            )
        assert row["frame"] < row["end_frame"] <= row["passage_frame"]
        assert row["features"]["approach_duration_s"] == pytest.approx(
            evidence["end_time"] - evidence["start_time"]
        )
    assert extract(engine, combined_analysis, "--candidates", str(supplied)) == batch
    player = batch["rows"][0]["player_id"]
    expected = [row for row in batch["rows"] if row["player_id"] == player]
    assert (
        extract(engine, combined_analysis, "--player", json.dumps(player))["rows"]
        == expected
    )
    assert (
        extract(
            engine,
            combined_analysis,
            "--candidates",
            str(supplied),
            "--player",
            json.dumps(player),
        )["rows"]
        == expected
    )
    contract = json.loads(
        subprocess.run(
            [str(engine), "overcommitting-feature-contract"],
            capture_output=True,
            check=True,
        ).stdout
    )
    assert contract == {
        "input_schema": batch["input_schema"],
        "feature_definitions": definitions,
    }
    generic_contract = json.loads(
        subprocess.run(
            [
                str(engine),
                "contract",
                "overcommitting",
                "--schema",
                batch["input_schema"],
            ],
            capture_output=True,
            check=True,
        ).stdout
    )
    assert generic_contract["feature_definitions"] == definitions
    features = tmp_path / "features.parquet"
    computed = compute(
        [combined_analysis],
        engine,
        SAMPLING,
        features,
        candidate_batches={combined_analysis: supplied},
    )
    table, prepared_contract = overcommitting_dataset.load(features)
    assert computed["rows"] == table.num_rows == len(batch["rows"])
    assert prepared_contract["feature_definitions"] == definitions


def test_supplied_candidates_select_exact_incidents_without_redetection(
    engine, combined_analysis, tmp_path
):
    selection = save_candidates(engine, combined_analysis, tmp_path / "candidates.json")
    batch = extract(engine, combined_analysis)
    assert len(batch["rows"]) > 1
    reduced = copy.deepcopy(selection)
    reduced["candidates"] = reduced["candidates"][1:2]
    path = tmp_path / "selected.json"
    path.write_text(json.dumps([reduced]))
    exact = extract(engine, combined_analysis, "--candidates", str(path))
    assert exact["rows"] == batch["rows"][1:2]
    reduced["candidates"] = []
    path.write_text(json.dumps([reduced]))
    assert extract(engine, combined_analysis, "--candidates", str(path))["rows"] == []


@pytest.mark.parametrize("shape", ["object", "empty", "multiple"])
def test_supplied_candidates_require_one_batch_array(
    engine, combined_analysis, tmp_path, shape
):
    path = tmp_path / "candidates.json"
    selection = save_candidates(engine, combined_analysis, path)
    invalid = {
        "object": selection,
        "empty": [],
        "multiple": [selection, selection],
    }[shape]
    path.write_text(json.dumps(invalid))
    result = run(
        engine,
        combined_analysis,
        "overcommitting-features",
        str(SAMPLING),
        "--candidates",
        str(path),
    )
    assert result.returncode == 1
    assert not result.stdout


def test_incompatible_sampling_fails_without_producing_rows(
    engine, combined_analysis, tmp_path
):
    sampling = tmp_path / "wrong-sampling.json"
    sampling.write_text('{"method":"native_touch_stages"}')
    result = run(engine, combined_analysis, "overcommitting-features", str(sampling))
    assert result.returncode == 1
    assert not result.stdout
    assert (
        "sampling method disagrees with the feature contract" in result.stderr.decode()
    )
