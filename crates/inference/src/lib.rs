//! ONNX inference for validated Rocket League mistake candidates.

pub mod manifest;

use manifest::{Manifest, ModelEntry};
use ort::{
    session::Session,
    value::{Tensor, TensorElementType, ValueType},
};
use rl_mistake_analysis_candidates::feature_contract::{
    self, ContextFeatures, FeatureBatch, FeatureSampling,
};
use rl_mistake_analysis_candidates::{Candidate, CandidateBatch, MistakeKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{borrow::Cow, collections::HashMap, path::Path};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ModelIdentity {
    pub kind: String,
    pub version: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub candidate: Candidate,
    pub score: f64,
    pub keep: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictionBatch {
    pub kind: String,
    pub input_schema: String,
    pub feature_names: Vec<String>,
    pub model_input_schema: String,
    pub model_feature_names: Vec<String>,
    pub engine_revision: String,
    pub replay_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<FeatureSampling>,
    pub model: ModelIdentity,
    pub predictions: Vec<Prediction>,
}

#[derive(Debug)]
pub struct Model {
    entry: ModelEntry,
    identity: ModelIdentity,
    session: Session,
}

impl Model {
    /// Load the explicitly selected model and verify its declared artifacts.
    pub fn load(directory: &Path, kind: &str, version: &str) -> Result<Self, String> {
        let manifest = Manifest::load(directory)?;
        let entry = manifest.model(kind, version)?.clone();
        MistakeKind::parse(kind)?.validate_contract(&entry.input_schema, &entry.feature_names)?;
        if entry.runtime != "onnx" {
            return Err(format!("unsupported model runtime: {}", entry.runtime));
        }
        let mut builder = Session::builder().map_err(|error| error.to_string())?;
        let model_parent = Path::new(&entry.model_file).parent().unwrap();
        for artifact in &entry.files {
            if artifact.path == entry.model_file {
                continue;
            }
            if artifact.path == entry.evaluation_file {
                artifact.verify(directory)?;
                continue;
            }
            let relative = Path::new(&artifact.path)
                .strip_prefix(model_parent)
                .map_err(|_| "tensor files must be inside the model directory")?;
            let bytes = artifact.read_verified(directory)?;
            builder = builder
                .with_external_initializer_file_in_memory(relative, Cow::Owned(bytes))
                .map_err(|error| format!("cannot load tensor artifact: {error}"))?;
        }
        let graph = entry.model_artifact()?.read_verified(directory)?;
        let session = builder
            .commit_from_memory(&graph)
            .map_err(|error| format!("cannot load ONNX model: {error}"))?;
        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            return Err("ONNX model must have one input and one output".into());
        }
        validate_tensor(
            session.inputs()[0].dtype(),
            entry.feature_names.len(),
            "input",
        )?;
        validate_tensor(session.outputs()[0].dtype(), 1, "output")?;
        let identity = ModelIdentity {
            kind: entry.kind.clone(),
            version: entry.version.clone(),
            sha256: format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&entry).map_err(|error| error.to_string())?)
            ),
        };
        Ok(Self {
            entry,
            identity,
            session,
        })
    }

    pub fn metadata(&self) -> &ModelEntry {
        &self.entry
    }

    pub fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    /// Return positive-class probabilities in the original candidate order.
    pub fn score(&mut self, batch: CandidateBatch) -> Result<PredictionBatch, String> {
        batch.validate()?;
        if batch.kind != self.entry.kind
            || batch.input_schema != self.entry.input_schema
            || batch.feature_names != self.entry.feature_names
            || batch.sampling != self.entry.sampling
        {
            return Err("model/input feature contract mismatch".into());
        }
        if self.entry.input_schema == "bumping_teammate.native.v1"
            && batch
                .candidates
                .iter()
                .any(|candidate| candidate.evidence.get("initiator") != Some(&candidate.player_id))
        {
            return Err("eight-feature models score only the initiator; use a subject-specific model for victims".into());
        }
        let inputs = batch
            .candidates
            .iter()
            .map(|candidate| {
                candidate
                    .features
                    .iter()
                    .map(|value| value.unwrap() as f32)
                    .collect()
            })
            .collect();
        self.score_features(batch, inputs)
    }

    /// Join context measurements by native incident identity and judged player.
    pub fn score_context<R: ContextFeatures>(
        &mut self,
        batch: CandidateBatch,
        features: FeatureBatch<R>,
    ) -> Result<PredictionBatch, String> {
        batch.validate()?;
        let definition = MistakeKind::parse(&batch.kind)?.definition();
        if batch.kind != self.entry.kind
            || !feature_contract::supports_schema(
                definition.context_schema,
                &self.entry.input_schema,
            )
            || features.input_schema != definition.context_schema
            || features.feature_definitions != (definition.context_definitions)()
            || Some(features.sampling) != self.entry.sampling
            || features.engine_revision != batch.engine_revision
            || features.replay_sha256 != batch.replay_sha256
        {
            return Err("model/input feature contract mismatch".into());
        }
        let selected = feature_contract::select_definitions(
            (definition.context_definitions)(),
            &self.entry.feature_names,
        )?;
        let mut rows = HashMap::new();
        for row in features.rows {
            let (event, player) = row.key();
            let key = (
                event.to_owned(),
                serde_json::to_string(player).map_err(|error| error.to_string())?,
            );
            if rows.insert(key, row).is_some() {
                return Err("duplicate subject feature row".into());
            }
        }
        let mut inputs = Vec::with_capacity(batch.candidates.len());
        for candidate in &batch.candidates {
            let key = (
                feature_contract::context_key(candidate)?,
                serde_json::to_string(&candidate.player_id).map_err(|error| error.to_string())?,
            );
            let row = rows
                .remove(&key)
                .ok_or("candidate has no subject feature row")?;
            row.validate_candidate(candidate)?;
            let vector = feature_contract::model_inputs(row.features(), &selected)?;
            inputs.push(vector);
        }
        if !rows.is_empty() {
            return Err("subject feature rows have no corresponding candidates".into());
        }
        self.score_features(batch, inputs)
    }

    fn score_features(
        &mut self,
        batch: CandidateBatch,
        inputs: Vec<Vec<f32>>,
    ) -> Result<PredictionBatch, String> {
        let count = batch.candidates.len();
        let width = self.entry.feature_names.len();
        if inputs.len() != count || inputs.iter().any(|row| row.len() != width) {
            return Err("model inputs must contain one ordered vector per candidate".into());
        }
        let mut predictions = Vec::with_capacity(count);
        if count != 0 {
            let features: Vec<f32> = inputs.into_iter().flatten().collect();
            if features.iter().any(|value| value.is_infinite()) {
                return Err("features cannot be represented as finite float32 values".into());
            }
            let input = Tensor::from_array(([count, width], features.into_boxed_slice()))
                .map_err(|error| format!("cannot construct input tensor: {error}"))?;
            let outputs = self
                .session
                .run(ort::inputs![input])
                .map_err(|error| format!("ONNX inference failed: {error}"))?;
            let (shape, probabilities) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|error| format!("invalid ONNX output: {error}"))?;
            if shape.as_ref() != [count as i64, 1] || probabilities.len() != count {
                return Err("ONNX output must contain one probability per candidate".into());
            }
            for (candidate, probability) in batch.candidates.into_iter().zip(probabilities) {
                if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                    return Err("ONNX output contains an invalid probability".into());
                }
                let score = f64::from(*probability);
                predictions.push(Prediction {
                    candidate,
                    score,
                    keep: score >= self.entry.keep_threshold,
                });
            }
        }
        Ok(PredictionBatch {
            kind: batch.kind,
            input_schema: batch.input_schema,
            feature_names: batch.feature_names,
            model_input_schema: self.entry.input_schema.clone(),
            model_feature_names: self.entry.feature_names.clone(),
            engine_revision: batch.engine_revision,
            replay_sha256: batch.replay_sha256,
            sampling: self.entry.sampling,
            model: self.identity.clone(),
            predictions,
        })
    }
}

fn validate_tensor(value: &ValueType, width: usize, name: &str) -> Result<(), String> {
    if let ValueType::Tensor { ty, shape, .. } = value
        && *ty == TensorElementType::Float32
        && shape.as_ref() == [-1, width as i64]
    {
        return Ok(());
    }
    Err(format!(
        "ONNX {name} must be a float32 tensor with shape [N, {width}]"
    ))
}
