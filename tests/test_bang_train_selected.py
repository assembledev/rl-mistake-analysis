"""Verify selected bang recipes, replay isolation, and native/ONNX scoring."""

import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from catboost import CatBoostClassifier
from test_participant_pipeline import run

from training.bang_features import compute
from training.dataset import arrays, load, select_features
from training.train_selected import CONFIG, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "version, width", [("0.1.0", 103), ("0.2.0", 142), ("0.3.0", 141)]
)
def test_selected_bang_release_matches_native_and_cli_probabilities(
    engine, combined_analysis, tmp_path, version, width
):
    config = CONFIG.with_name(f"bang_with_time_{version}.json")
    settings = json.loads(config.read_text())
    feature_path = tmp_path / "features.parquet"
    compute(
        [combined_analysis], engine, ROOT / "examples/bang_sampling.json", feature_path
    )
    extracted = pq.read_table(feature_path)
    assert len(extracted) >= 4
    tables = []
    for replay in range(12):
        table = extracted.slice(0, 4).set_column(
            extracted.schema.get_field_index("replay_sha256"),
            "replay_sha256",
            pa.array([f"{replay:064x}"] * 4),
        )
        labels = [0, 0, 0, 1] if replay < 10 else [0, 1, 1, 1]
        tables.append(table.append_column("label", pa.array(labels, type=pa.uint8())))
    training_path = tmp_path / "training.parquet"
    pq.write_table(pa.concat_tables(tables), training_path)
    test_replays = tmp_path / "test_replays.json"
    test_replays.write_text(json.dumps([f"{replay:064x}" for replay in (10, 11)]))
    output = tmp_path / "release"
    report = train(
        training_path,
        test_replays,
        output,
        source_revision="training-test",
        config=config,
    )
    table, contract = load(training_path)
    contract = select_features(
        contract, settings["feature_names"], settings["input_schema"]
    )
    values, labels, groups = arrays(table, contract)
    native = (
        CatBoostClassifier().load_model(output / "catboost.cbm")
        if settings["model"] == "catboost"
        else joblib.load(
            output
            / (
                "extra_trees.joblib"
                if settings["model"] == "extra_trees"
                else "ebm.joblib"
            )
        )
    )
    saved = np.load(output / "predictions.npz")
    np.testing.assert_array_equal(saved["labels"], labels)
    np.testing.assert_allclose(
        saved["probabilities"], native.predict_proba(values)[:, 1]
    )
    splits = json.loads((output / "splits.json").read_text())
    assert len(splits["train"]) == 40 and len(splits["test"]) == 8
    assert not set(groups[splits["train"]]) & set(groups[splits["test"]])
    weighting = report["class_weighting"]
    assert weighting["training_class_counts"] == [30, 10]
    expected_weights = np.sqrt(40 / (2 * np.array([30, 10])))
    expected_weights /= np.average(expected_weights, weights=[30, 10])
    np.testing.assert_allclose(weighting["training_class_weights"], expected_weights)
    assert report["feature_contract"]["feature_names"] == settings["feature_names"]
    assert report["onnx_validation"]["rows"] == len(labels)
    assert report["onnx_validation"]["features"] == width
    assert json.loads((output / "recipe.json").read_text()) == settings
    if "candidate_schema" in settings:
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["models"][0]["candidate_schema"] == settings["candidate_schema"]
        assert report["candidate_schema"] == settings["candidate_schema"]
    if settings["model"] == "additive_ebm":
        fitted = report["fitted_parameters"]
        np.testing.assert_array_equal(fitted["intercept"], native.intercept_)
        for term, (feature,) in zip(
            fitted["terms"], native.term_features_, strict=True
        ):
            assert term["feature"] == settings["feature_names"][feature]
            np.testing.assert_array_equal(term["cuts"], native.bins_[feature][0])
    document = json.loads(combined_analysis.read_text())
    predictions = run(
        engine, "score", output, "bang_with_time", version, document=document
    )
    unlabeled = extracted.append_column(
        "label", pa.array([0] * len(extracted), type=pa.uint8())
    )
    all_values, _, _ = arrays(unlabeled, contract)
    expected = native.predict_proba(all_values)[:, 1]
    assert len(predictions["predictions"]) == len(extracted)
    assert predictions["model_feature_names"] == settings["feature_names"]
    for prediction, probability in zip(
        predictions["predictions"], expected, strict=True
    ):
        assert prediction["score"] == pytest.approx(probability, abs=1e-6)
        assert prediction["keep"] == bool(probability >= settings["keep_threshold"])


def test_selected_recipe_rejects_an_unsupported_kind_classifier_pair(tmp_path):
    settings = json.loads(CONFIG.read_text())
    settings["kind"] = "bang_with_time"
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="Unsupported mistake kind and classifier"):
        train(
            tmp_path / "training.parquet",
            tmp_path / "test.json",
            tmp_path / "release",
            source_revision="training-test",
            config=config,
        )
    assert not (tmp_path / "release").exists()
