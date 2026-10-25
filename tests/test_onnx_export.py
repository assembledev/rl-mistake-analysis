import joblib
import numpy as np
import pytest
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier

from training.onnx_export import additive_ebm, catboost, extra_trees, verify
from training.train_selected import extra_trees_pipeline


def test_extra_trees_export_preserves_imputation_and_missing_indicators():
    features = np.array(
        [[-2, np.nan, 0], [-1, np.nan, 1], [1, np.nan, 2], [np.nan, np.nan, 3]],
        dtype=np.float32,
    )
    model = extra_trees_pipeline(n_estimators=20, random_state=42, n_jobs=1).fit(
        features, [0, 0, 1, 1]
    )
    examples = np.vstack([features, [100, 10, np.nan], [np.nan, np.nan, np.nan]])
    graph = extra_trees(model, features)
    for rows in (1, len(examples)):
        verify(graph, examples[:rows], model.predict_proba(examples[:rows])[:, 1])
    np.testing.assert_array_equal(
        model.named_steps["imputer"].statistics_, [-1, 0, 1.5]
    )
    np.testing.assert_array_equal(
        model.named_steps["imputer"].indicator_.features_, [0, 1]
    )


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_numeric_export_matches_native_probabilities_with_missing_values(
    nan_mode, tmp_path
):
    values = np.tile([-2, -1, 1, 2, np.nan], 8)
    features = np.repeat(values[:, None], 105, axis=1).astype(np.float32)
    labels = ((values > 0) | np.isnan(values)).astype(int)
    model = CatBoostClassifier(
        iterations=20,
        depth=2,
        nan_mode=nan_mode,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    ).fit(features, labels)
    graph = catboost(model, features)
    probabilities = model.predict_proba(features)[:, 1]
    for rows in (1, len(features)):
        validation = verify(graph, features[:rows], probabilities[:rows])
        assert validation["features"] == 105
        assert validation["rows"] == rows
        assert validation["max_absolute_error"] < 1e-6
    with pytest.raises(AssertionError):
        verify(graph, features, 1 - probabilities)
    joblib.dump(model, tmp_path / "model.joblib")
    restored = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(restored.predict_proba(features)[:, 1], probabilities)
    verify(catboost(restored, features), features, probabilities)


def test_rejects_native_categorical_features():
    features = [["a"], ["b"], ["a"], ["b"]]
    model = CatBoostClassifier(
        iterations=2, thread_count=1, verbose=False, allow_writing_files=False
    ).fit(features, [0, 1, 0, 1], cat_features=[0])
    with pytest.raises(ValueError, match="numeric features only"):
        catboost(model, features)


def test_rejects_incompatible_feature_width_and_labels():
    features = np.arange(8, dtype=np.float32).reshape(-1, 1)
    model = CatBoostClassifier(
        iterations=2, thread_count=1, verbose=False, allow_writing_files=False
    ).fit(features, [0, 1] * 4)
    with pytest.raises(ValueError, match="feature width"):
        catboost(model, np.zeros((8, 2)))
    model.fit(features, [1, 2] * 4)
    with pytest.raises(ValueError, match="binary labels"):
        catboost(model, features)


def test_additive_ebm_export_preserves_missing_bins_and_cut_boundaries():
    values = np.tile([-2, -1, 0, 1, 2, np.nan], 12).astype(np.float32)
    features = np.column_stack([values, np.zeros_like(values)])
    labels = ((values > 0) | np.isnan(values)).astype(int)
    model = ExplainableBoostingClassifier(
        feature_types=["continuous", "continuous"],
        interactions=0,
        outer_bags=1,
        validation_size=0,
        early_stopping_rounds=0,
        max_bins=8,
        max_rounds=20,
        n_jobs=1,
        random_state=42,
    ).fit(features, labels)
    cuts = model.bins_[0][0].astype(np.float32)
    boundary_values = np.concatenate(
        [
            np.nextafter(cuts, np.float32(-np.inf)),
            cuts,
            np.nextafter(cuts, np.float32(np.inf)),
            [np.nan],
        ]
    ).astype(np.float32)
    examples = np.column_stack([boundary_values, np.zeros_like(boundary_values)])
    graph = additive_ebm(model, features)
    for rows in (1, len(examples)):
        verify(graph, examples[:rows], model.predict_proba(examples[:rows])[:, 1])
    with pytest.raises(ValueError, match="feature width"):
        additive_ebm(model, np.zeros((8, 3)))
    model.fit(features, labels + 1)
    with pytest.raises(ValueError, match="binary classes"):
        additive_ebm(model, features)
