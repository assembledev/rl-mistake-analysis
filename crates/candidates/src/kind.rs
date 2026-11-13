use crate::feature_contract::{self, FeatureDefinition, FeatureSampling, SamplingMethod};
use crate::input::ReplayFacts;
use crate::{bang_features, bump_features, overcommitting_features};
use crate::{native_bump, native_touch, overcommitting};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

pub struct KindDefinition {
    pub name: &'static str,
    pub schema: &'static str,
    pub feature_names: &'static [&'static str],
    pub context_schema: &'static str,
    pub context_definitions: fn() -> Vec<FeatureDefinition>,
    pub sampling_method: SamplingMethod,
    candidates: fn(&ReplayFacts) -> Result<Vec<Candidate>, String>,
    anchor: fn(&Candidate) -> Result<crate::annotations::IncidentAnchor, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistakeKind {
    BumpingTeammate,
    BangWithTime,
    Overcommitting,
}

impl MistakeKind {
    pub const ALL: [Self; 3] = [
        Self::BumpingTeammate,
        Self::BangWithTime,
        Self::Overcommitting,
    ];

    pub fn parse(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.definition().name == name)
            .ok_or_else(|| format!("unknown mistake kind: {name}"))
    }

    pub fn definition(self) -> &'static KindDefinition {
        match self {
            Self::BumpingTeammate => &KindDefinition {
                name: "bumping_teammate",
                schema: native_bump::SCHEMA,
                feature_names: native_bump::FEATURES,
                context_schema: bump_features::SCHEMA,
                context_definitions: bump_features::definitions,
                sampling_method: SamplingMethod::FirstAccelerationPeakThenMinimum,
                candidates: native_bump::candidates,
                anchor: |candidate| {
                    native_bump::validate_evidence(&candidate.evidence)?;
                    native_bump::anchor(candidate)
                },
            },
            Self::BangWithTime => &KindDefinition {
                name: "bang_with_time",
                schema: native_touch::SCHEMA,
                feature_names: &[],
                context_schema: bang_features::SCHEMA,
                context_definitions: bang_features::definitions,
                sampling_method: SamplingMethod::NativeTouchStages,
                candidates: native_touch::select,
                anchor: native_touch::anchor,
            },
            Self::Overcommitting => &KindDefinition {
                name: "overcommitting",
                schema: overcommitting::SCHEMA,
                feature_names: &[],
                context_schema: overcommitting_features::SCHEMA,
                context_definitions: overcommitting_features::definitions,
                sampling_method: SamplingMethod::NativeIntervalBoundaries,
                candidates: |_| {
                    Err("overcommitting requires combined events and replay_data".into())
                },
                anchor: overcommitting::anchor,
            },
        }
    }

    pub fn name(self) -> &'static str {
        self.definition().name
    }

    pub fn schema(self) -> &'static str {
        self.definition().schema
    }

    pub fn feature_names(self) -> &'static [&'static str] {
        self.definition().feature_names
    }

    pub fn validate_contract(self, schema: &str, feature_names: &[String]) -> Result<(), String> {
        let definition = self.definition();
        let compatible = match schema {
            schema if schema == definition.schema => feature_names == definition.feature_names,
            schema if feature_contract::supports_schema(definition.context_schema, schema) => {
                feature_contract::select_definitions(
                    (definition.context_definitions)(),
                    feature_names,
                )?;
                true
            }
            _ => false,
        };
        if !compatible {
            return Err(format!(
                "incompatible feature contract for {}",
                definition.name
            ));
        }
        Ok(())
    }

    pub fn candidates(self, replay: &ReplayFacts) -> Result<CandidateBatch, String> {
        replay.validate()?;
        let definition = self.definition();
        let candidates = (definition.candidates)(replay)?;
        self.batch(replay, candidates)
    }

    pub fn candidates_from_analysis(
        self,
        analysis: &crate::feature_contract::ReplayAnalysis,
    ) -> Result<CandidateBatch, String> {
        if self == Self::Overcommitting {
            self.batch(&analysis.facts(), overcommitting::candidates(analysis)?)
        } else {
            self.candidates(&analysis.facts())
        }
    }

    fn batch(
        self,
        replay: &ReplayFacts,
        candidates: Vec<Candidate>,
    ) -> Result<CandidateBatch, String> {
        let definition = self.definition();
        let batch = CandidateBatch {
            kind: definition.name.into(),
            input_schema: definition.schema.into(),
            feature_names: definition
                .feature_names
                .iter()
                .map(|name| (*name).into())
                .collect(),
            engine_revision: replay.revision.clone(),
            replay_sha256: replay.replay_sha256.clone(),
            sampling: None,
            feature_definitions: Vec::new(),
            candidates,
        };
        batch.validate()?;
        Ok(batch)
    }

    /// Preserve the subject convention of the eight-feature training contract.
    pub fn initiator_candidates(self, replay: &ReplayFacts) -> Result<CandidateBatch, String> {
        if self != Self::BumpingTeammate {
            return Err("initiator candidates require bumping_teammate".into());
        }
        let mut batch = self.candidates(replay)?;
        batch
            .candidates
            .retain(|candidate| candidate.evidence.get("initiator") == Some(&candidate.player_id));
        Ok(batch)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub kind: String,
    pub source_event_ids: Vec<String>,
    pub frame: u64,
    pub raw_time: f64,
    pub player_id: Value,
    pub features: Vec<Option<f64>>,
    pub evidence: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateBatch {
    pub kind: String,
    pub input_schema: String,
    pub feature_names: Vec<String>,
    pub engine_revision: String,
    pub replay_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<FeatureSampling>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feature_definitions: Vec<FeatureDefinition>,
    pub candidates: Vec<Candidate>,
}

impl CandidateBatch {
    pub fn validate(&self) -> Result<(), String> {
        let definition = MistakeKind::parse(&self.kind)?.definition();
        if self.input_schema != definition.schema
            || self.feature_names != definition.feature_names
            || self.sampling.is_some()
            || !self.feature_definitions.is_empty()
        {
            return Err("candidate feature definitions disagree with the input schema".into());
        }
        if self.engine_revision.trim().is_empty() || !crate::input::is_sha256(&self.replay_sha256) {
            return Err("missing engine revision or invalid replay fingerprint".into());
        }
        let mut identities = HashSet::new();
        let mut incidents = HashSet::new();
        for candidate in &self.candidates {
            if candidate.kind != self.kind
                || candidate.player_id.is_null()
                || !candidate.raw_time.is_finite()
                || candidate.features.len() != self.feature_names.len()
                || candidate
                    .features
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
            {
                return Err("invalid candidate measurements or identity".into());
            }
            let anchor = candidate.anchor()?;
            for (index, value) in candidate.features.iter().enumerate() {
                if value.is_none() {
                    return Err(format!(
                        "invalid measurement or subject for {}",
                        self.feature_names[index]
                    ));
                }
            }
            let subject =
                serde_json::to_string(&candidate.player_id).map_err(|error| error.to_string())?;
            if !incidents.insert((anchor.key()?, subject.clone())) {
                return Err("duplicate incident anchor".into());
            }
            let mut ids: Vec<_> = candidate.source_event_ids.iter().collect();
            ids.sort();
            if ids.is_empty()
                || ids.iter().any(|id| id.trim().is_empty())
                || ids.windows(2).any(|pair| pair[0] == pair[1])
                || (self.kind != "overcommitting" && !identities.insert((ids, subject)))
            {
                return Err("candidate event identities must be nonempty and unique".into());
            }
        }
        Ok(())
    }
}

impl Candidate {
    pub fn anchor(&self) -> Result<crate::annotations::IncidentAnchor, String> {
        (MistakeKind::parse(&self.kind)?.definition().anchor)(self)
    }
}
