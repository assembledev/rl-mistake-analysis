"""Train a selected classifier and export a verified model release."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import onnx
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from training.dataset import arrays, load, select_features
from training.export import model_entry, write_manifest
from training.onnx_export import additive_ebm, catboost, extra_trees, verify, xgboost

CONFIG = Path(__file__).parent / "configs/bumping_teammate_0.1.0.json"


def extra_trees_pipeline(**parameters):
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("classifier", ExtraTreesClassifier(**parameters)),
        ]
    )


MODELS = {
    ("bumping_teammate", "xgboost"): (XGBClassifier, xgboost),
    ("bang_with_time", "catboost"): (CatBoostClassifier, catboost),
    ("bang_with_time", "additive_ebm"): (ExplainableBoostingClassifier, additive_ebm),
    ("bang_with_time", "extra_trees"): (extra_trees_pipeline, extra_trees),
}


def partitions(labels, groups, test_replays):
    """Reserve the explicitly supplied replays without moving rows between groups."""
    if (
        not isinstance(test_replays, list)
        or not test_replays
        or any(not isinstance(replay, str) for replay in test_replays)
        or len(set(test_replays)) != len(test_replays)
        or set(test_replays) - set(groups)
    ):
        raise ValueError("Test replays must be unique, known replay checksums")
    mask = np.isin(groups, test_replays)
    train, test = np.flatnonzero(~mask), np.flatnonzero(mask)
    for rows in (train, test):
        if not np.array_equal(np.unique(labels[rows]), [0, 1]):
            raise ValueError(
                "Training and test partitions must each contain both classes"
            )
    return train, test


def metrics(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    values = {
        "precision": precision_score(labels, predictions, zero_division=np.nan),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "roc_auc": roc_auc_score(labels, probabilities),
        "average_precision": average_precision_score(labels, probabilities),
        "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(labels, probabilities),
    }
    return {
        **{
            name: float(value) if np.isfinite(value) else None
            for name, value in values.items()
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def train(training_table, test_replays, output, *, source_revision, config=CONFIG):
    """Fit one fixed configuration, evaluate complete held-out replays, and verify ONNX."""
    settings = json.loads(Path(config).read_text())
    identity = (settings["kind"], settings["model"])
    if identity not in MODELS:
        raise ValueError("Unsupported mistake kind and classifier in selected recipe")
    classifier_type, exporter = MODELS[identity]
    release_version = settings["version"]
    candidate_schema = settings.get("candidate_schema")
    if (
        not isinstance(release_version, str)
        or not release_version.strip()
        or release_version in (".", "..")
        or "/" in release_version
        or "\\" in release_version
        or settings["status"] not in ("experimental", "approved")
    ):
        raise ValueError("Provide a valid model release version and status")
    threshold = settings["keep_threshold"]
    if (
        type(threshold) not in (int, float)
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
        or not source_revision.strip()
    ):
        raise ValueError("Provide a valid threshold and training source revision")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory for each training run")
    table, source_contract = load(training_table)
    if source_contract["kind"] != settings["kind"]:
        raise ValueError("Training data and recipe mistake kinds differ")
    contract = select_features(
        source_contract, settings["feature_names"], settings["input_schema"]
    )
    if contract["sampling"] != settings["sampling"]:
        raise ValueError("Training data and recipe sampling methods differ")
    features, labels, groups = arrays(table, contract)
    train_rows, test_rows = partitions(
        labels, groups, json.loads(Path(test_replays).read_text())
    )
    weighting = settings["class_weighting"]
    fit_parameters = {}
    if weighting == "sqrt_balanced":
        training_labels = labels[train_rows]
        counts = np.bincount(training_labels, minlength=2)
        weights = np.sqrt(len(training_labels) / (2 * counts))
        weights /= np.mean(weights[training_labels])
        weight_parameter = (
            "classifier__sample_weight"
            if settings["model"] == "extra_trees"
            else "sample_weight"
        )
        fit_parameters[weight_parameter] = weights[training_labels]
        weighting = {
            "method": weighting,
            "power": 0.5,
            "normalization": "mean training-row weight equals one",
            "training_class_counts": counts.tolist(),
            "training_class_weights": weights.tolist(),
            "fit_scope": "each fit computes weights from its own training labels",
        }
    elif weighting != "none":
        raise ValueError("Unsupported selected recipe class weighting")
    classifier = classifier_type(**settings["hyperparameters"])
    classifier.fit(features[train_rows], labels[train_rows], **fit_parameters)
    probabilities = classifier.predict_proba(features)[:, 1]
    graph = exporter(classifier, features)
    validation = verify(graph, features, probabilities)
    with Path(training_table).open("rb") as stream:
        dataset_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "schema_version": 1,
        "kind": settings["kind"],
        "version": settings["version"],
        "status": settings["status"],
        "model": settings["model"],
        "hyperparameters": settings["hyperparameters"],
        "class_weighting": weighting,
        "decision_threshold": threshold,
        "decision_rule": "positive probability >= decision_threshold",
        "feature_contract": {
            key: contract[key]
            for key in (
                "schema_version",
                "kind",
                "input_schema",
                "sampling",
                "feature_names",
                "feature_definitions",
            )
        },
        "dataset": {
            "training_parquet_sha256": dataset_hash,
            "rows": len(labels),
            "replays": len(set(groups)),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_replays": len(set(groups[train_rows])),
            "test_replays": len(set(groups[test_rows])),
        },
        "partition_metrics": {
            "train": metrics(labels[train_rows], probabilities[train_rows], threshold),
            "test": metrics(labels[test_rows], probabilities[test_rows], threshold),
        },
        "onnx_validation": validation,
        "source_revision": source_revision,
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                {
                    "xgboost": "xgboost",
                    "catboost": "catboost",
                    "additive_ebm": "interpret-core",
                    "extra_trees": "scikit-learn",
                }[settings["model"]],
                "joblib",
                "scikit-learn",
                "onnx",
                "onnxmltools",
                "onnxruntime",
            )
        },
    }
    if settings["model"] == "additive_ebm":
        report["fitted_parameters"] = {
            "intercept": classifier.intercept_.tolist(),
            "link": classifier.link_,
            "terms": [
                {
                    "feature": contract["feature_names"][feature],
                    "cuts": classifier.bins_[feature][0].tolist(),
                    "scores": classifier.term_scores_[term].tolist(),
                }
                for term, (feature,) in enumerate(classifier.term_features_)
            ],
        }
    elif settings["model"] == "catboost":
        report["resolved_parameters"] = classifier.get_all_params()
        report["tree_count"] = classifier.tree_count_
    elif settings["model"] == "extra_trees":
        imputer = classifier.named_steps["imputer"]
        report["fitted_parameters"] = {
            "imputation_medians": imputer.statistics_.tolist(),
            "missing_indicator_columns": imputer.indicator_.features_.tolist(),
            "tree_count": len(classifier.named_steps["classifier"].estimators_),
        }
    if candidate_schema is not None:
        report["candidate_schema"] = candidate_schema
    relative = Path("models") / settings["kind"] / settings["version"]
    model_directory = output / relative
    model_directory.mkdir(parents=True, exist_ok=False)
    onnx.save(graph, model_directory / "model.onnx")
    (model_directory / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if settings["model"] == "xgboost":
        classifier.save_model(output / "xgboost.json")
    elif settings["model"] == "catboost":
        classifier.save_model(output / "catboost.cbm")
    elif settings["model"] == "additive_ebm":
        joblib.dump(classifier, output / "ebm.joblib")
    else:
        joblib.dump(classifier, output / "extra_trees.joblib")
    (output / "recipe.json").write_text(
        json.dumps(settings, indent=2, allow_nan=False) + "\n"
    )
    (output / "splits.json").write_text(
        json.dumps({"train": train_rows.tolist(), "test": test_rows.tolist()}, indent=2)
        + "\n"
    )
    np.savez_compressed(
        output / "predictions.npz", labels=labels, probabilities=probabilities
    )
    entry = model_entry(
        output,
        training_table,
        version=settings["version"],
        model_file=(relative / "model.onnx").as_posix(),
        evaluation_file=(relative / "evaluation.json").as_posix(),
        keep_threshold=threshold,
        source_revision=source_revision,
        status=settings["status"],
        feature_names=settings["feature_names"],
        input_schema=settings["input_schema"],
        candidate_schema=candidate_schema,
    )
    write_manifest(output, settings["version"], [entry])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_table", type=Path)
    parser.add_argument("test_replays", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    report = train(
        args.training_table,
        args.test_replays,
        args.output,
        source_revision=args.source_revision,
        config=args.config,
    )
    print(json.dumps(report["partition_metrics"], allow_nan=False))


if __name__ == "__main__":
    main()
