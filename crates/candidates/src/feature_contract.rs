use crate::Candidate;
pub use crate::feature_input::ReplayAnalysis;
pub use crate::feature_values::{DataType, FeatureDefinition, Measurement};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureSampling {
    pub method: SamplingMethod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingMethod {
    FirstAccelerationPeakThenMinimum,
    NativeTouchStages,
    NativeIntervalBoundaries,
}

impl FeatureSampling {
    pub fn validate(self, method: SamplingMethod) -> Result<(), String> {
        if self.method != method {
            return Err("sampling method disagrees with the feature contract".into());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct FeatureBatch<R> {
    pub input_schema: &'static str,
    pub engine_revision: String,
    pub replay_sha256: String,
    pub sampling: FeatureSampling,
    pub feature_definitions: Vec<FeatureDefinition>,
    pub rows: Vec<R>,
}

pub trait ContextFeatures {
    fn key(&self) -> (&str, &Value);
    fn validate_candidate(&self, candidate: &Candidate) -> Result<(), String>;
    fn features(&self) -> &BTreeMap<String, Option<Measurement>>;
}

pub fn context_key(candidate: &Candidate) -> Result<String, String> {
    match candidate.anchor()? {
        crate::annotations::IncidentAnchor::Overcommitting { frame, end_frame } => {
            Ok(format!("overcommitting:{frame}:{end_frame}"))
        }
        _ => match candidate.source_event_ids.as_slice() {
            [event] => Ok(event.clone()),
            _ => Err("contact context requires exactly one source event".into()),
        },
    }
}

pub fn supports_schema(base: &str, schema: &str) -> bool {
    schema == base
        || schema
            .strip_prefix(base)
            .and_then(|suffix| suffix.strip_prefix('.'))
            .and_then(|suffix| suffix.strip_suffix(".v1"))
            .is_some_and(|name| {
                !name.is_empty()
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            })
}

pub fn select_definitions(
    definitions: Vec<FeatureDefinition>,
    names: &[String],
) -> Result<Vec<FeatureDefinition>, String> {
    if names.is_empty() {
        return Err("context models require at least one feature".into());
    }
    let mut available: HashMap<_, _> = definitions
        .into_iter()
        .map(|definition| (definition.name.clone(), definition))
        .collect();
    names
        .iter()
        .map(|name| {
            available
                .remove(name)
                .ok_or_else(|| format!("unknown or duplicate context feature: {name}"))
        })
        .collect()
}

pub fn model_inputs(
    features: &BTreeMap<String, Option<Measurement>>,
    definitions: &[FeatureDefinition],
) -> Result<Vec<f32>, String> {
    definitions
        .iter()
        .map(|definition| {
            match features
                .get(&definition.name)
                .ok_or("missing model measurement")?
            {
                None if definition.nullable => Ok(f32::NAN),
                Some(Measurement::Boolean(value)) if definition.data_type == DataType::Boolean => {
                    Ok(f32::from(u8::from(*value)))
                }
                Some(Measurement::Count(value)) if definition.data_type == DataType::U32 => {
                    Ok(*value as f32)
                }
                Some(Measurement::Float(value))
                    if definition.data_type == DataType::F32 && value.is_finite() =>
                {
                    Ok(*value)
                }
                _ => Err("invalid model measurement".into()),
            }
        })
        .collect()
}
