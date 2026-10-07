use rl_mistake_analysis_candidates::{CandidateBatch, MistakeKind, ReplayFacts};
use rl_mistake_analysis_inference::{
    Model,
    manifest::{ArtifactFile, Manifest, ModelEntry, Provenance, ReleaseStatus},
};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use tempfile::TempDir;

fn batch() -> CandidateBatch {
    let facts: ReplayFacts =
        serde_json::from_str(include_str!("../../../examples/bump.json")).unwrap();
    MistakeKind::BumpingTeammate.candidates(&facts).unwrap()
}

fn artifact(path: &Path, name: &str) -> ArtifactFile {
    ArtifactFile {
        path: name.into(),
        sha256: format!("{:x}", Sha256::digest(fs::read(path.join(name)).unwrap())),
    }
}

fn save_manifest(directory: &Path, catalog: &Manifest) {
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec(catalog).unwrap(),
    )
    .unwrap();
}

fn manifest(fixture: &str) -> (TempDir, Manifest) {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    fs::copy(fixtures.join(fixture), directory.path().join("model.onnx")).unwrap();
    fs::write(directory.path().join("evaluation.json"), "{}").unwrap();
    let mut files = vec![
        artifact(directory.path(), "model.onnx"),
        artifact(directory.path(), "evaluation.json"),
    ];
    if fixture == "external.onnx" {
        fs::copy(
            fixtures.join("external.data"),
            directory.path().join("external.data"),
        )
        .unwrap();
        files.push(artifact(directory.path(), "external.data"));
    }
    let kind = MistakeKind::BumpingTeammate;
    let catalog = Manifest {
        schema_version: 1,
        version: "1.0.0".into(),
        models: vec![ModelEntry {
            kind: kind.name().into(),
            version: "1.0.0".into(),
            input_schema: kind.schema().into(),
            feature_names: kind
                .feature_names()
                .iter()
                .map(|name| (*name).into())
                .collect(),
            runtime: "onnx".into(),
            model_file: "model.onnx".into(),
            evaluation_file: "evaluation.json".into(),
            keep_threshold: 0.5,
            status: ReleaseStatus::Experimental,
            provenance: Provenance {
                source_revision: "operator-fixture".into(),
                engine_revisions: vec!["independent-supplier-revision".into()],
            },
            files,
        }],
    };
    save_manifest(directory.path(), &catalog);
    (directory, catalog)
}

fn load(directory: &TempDir) -> Result<Model, String> {
    Model::load(directory.path(), "bumping_teammate", "1.0.0")
}

#[test]
fn onnx_scores_a_batch_preserving_order_and_model_identity() {
    let (directory, _) = manifest("square.onnx");
    let mut model = load(&directory).unwrap();
    let mut input = batch();
    input.candidates[0].features[0] = 0.8;
    let mut second = input.candidates[0].clone();
    second.source_event_ids = vec!["bump:second".into()];
    second.frame += 1;
    second.evidence["frame"] = second.frame.into();
    second.features[0] = 0.3;
    input.candidates.push(second);
    let predictions = model.score(input.clone()).unwrap();
    assert_eq!(predictions.model, *model.identity());
    assert_eq!(predictions.engine_revision, input.engine_revision);
    assert_eq!(predictions.predictions.len(), 2);
    assert!((predictions.predictions[0].score - 0.64).abs() < 1e-6);
    assert!((predictions.predictions[1].score - 0.09).abs() < 1e-6);
    assert!(predictions.predictions[0].keep);
    assert!(!predictions.predictions[1].keep);
    assert_eq!(
        predictions.predictions[1].candidate.source_event_ids,
        vec!["bump:second"]
    );
    input.candidates.clear();
    assert!(model.score(input).unwrap().predictions.is_empty());
}

#[test]
fn load_rejects_wrong_tensor_signatures_and_feature_order() {
    for fixture in ["wrong_width.onnx", "wrong_dtype.onnx", "wrong_output.onnx"] {
        let (directory, _) = manifest(fixture);
        assert!(load(&directory).unwrap_err().contains("float32 tensor"));
    }
    let (directory, mut catalog) = manifest("square.onnx");
    catalog.models[0].feature_names.reverse();
    save_manifest(directory.path(), &catalog);
    assert!(load(&directory).unwrap_err().contains("feature contract"));
}

#[test]
fn score_rejects_overflow_invalid_probabilities_and_changed_contracts() {
    let (directory, _) = manifest("square.onnx");
    let mut model = load(&directory).unwrap();
    let mut input = batch();
    input.candidates[0].features[0] = f64::MAX;
    assert!(model.score(input).unwrap_err().contains("float32"));
    let mut input = batch();
    input.candidates[0].features[0] = 2.0;
    assert!(model.score(input).unwrap_err().contains("probability"));
    let mut input = batch();
    input.feature_names.reverse();
    assert!(model.score(input).is_err());
}

#[test]
fn artifacts_and_external_tensor_changes_are_verified_and_identified() {
    let (directory, mut catalog) = manifest("external.onnx");
    let model = &mut catalog.models[0];
    fs::create_dir_all(directory.path().join("bumping_teammate/1.0.0")).unwrap();
    for file in &mut model.files {
        let nested = format!("bumping_teammate/1.0.0/{}", file.path);
        fs::rename(
            directory.path().join(&file.path),
            directory.path().join(&nested),
        )
        .unwrap();
        file.path = nested;
    }
    model.model_file = model.files[0].path.clone();
    model.evaluation_file = model.files[1].path.clone();
    let undeclared = catalog.models[0].files.pop().unwrap();
    save_manifest(directory.path(), &catalog);
    assert!(
        load(&directory)
            .unwrap_err()
            .contains("cannot load ONNX model")
    );
    catalog.models[0].files.push(undeclared);
    save_manifest(directory.path(), &catalog);
    let mut input = batch();
    input.candidates[0].features[0] = 0.8;
    input.candidates[0].features[1] = 0.2;
    let mut original = load(&directory).unwrap();
    let identity = original.identity().clone();
    assert!((original.score(input.clone()).unwrap().predictions[0].score - 0.64).abs() < 1e-6);
    drop(original);
    let tensor_path = catalog.models[0].files[2].path.clone();
    fs::write(directory.path().join(&tensor_path), 1i64.to_le_bytes()).unwrap();
    assert!(load(&directory).unwrap_err().contains("hash mismatch"));
    catalog.models[0].files[2] = artifact(directory.path(), &tensor_path);
    save_manifest(directory.path(), &catalog);
    let mut changed = load(&directory).unwrap();
    assert_ne!(*changed.identity(), identity);
    assert!((changed.score(input).unwrap().predictions[0].score - 0.04).abs() < 1e-6);
    drop(changed);
    fs::write(
        directory.path().join(&catalog.models[0].model_file),
        b"corrupt",
    )
    .unwrap();
    assert!(load(&directory).unwrap_err().contains("hash mismatch"));
}

#[test]
fn model_selection_does_not_require_unselected_artifacts_or_known_kinds() {
    let (directory, mut catalog) = manifest("square.onnx");
    let mut future = catalog.models[0].clone();
    future.kind = "future_kind".into();
    future.model_file = "future.onnx".into();
    future.evaluation_file = "future.json".into();
    future.files[0].path = future.model_file.clone();
    future.files[1].path = future.evaluation_file.clone();
    catalog.models.push(future);
    save_manifest(directory.path(), &catalog);
    let model = load(&directory).unwrap();
    assert_eq!(model.metadata().version, "1.0.0");
    assert!(Model::load(directory.path(), "bumping_teammate", "9.0.0").is_err());
    assert!(catalog.verify_files(directory.path()).is_err());
}

#[test]
fn manifest_rejects_unsafe_paths_duplicate_models_and_conflicting_hashes() {
    let (_, catalog) = manifest("square.onnx");
    for path in [
        "../outside",
        "/absolute",
        "nested/../outside",
        "nested//file",
    ] {
        let mut invalid = catalog.clone();
        invalid.models[0].files[0].path = path.into();
        assert!(invalid.validate().is_err());
    }
    let mut duplicate = catalog.clone();
    duplicate.models.push(duplicate.models[0].clone());
    assert!(duplicate.validate().is_err());
    let mut conflict = catalog.clone();
    let mut other = conflict.models[0].clone();
    other.version = "2.0.0".into();
    other.files[0].sha256 = "1".repeat(64);
    conflict.models.push(other);
    assert!(conflict.validate().is_err());
}
