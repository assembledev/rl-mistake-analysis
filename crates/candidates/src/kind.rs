use crate::input::ReplayFacts;
use crate::native_bump;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistakeKind {
    BumpingTeammate,
}

impl MistakeKind {
    pub const ALL: [Self; 1] = [Self::BumpingTeammate];

    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "bumping_teammate" => Ok(Self::BumpingTeammate),
            _ => Err(format!("unknown mistake kind: {name}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::BumpingTeammate => "bumping_teammate",
        }
    }

    pub fn schema(self) -> &'static str {
        match self {
            Self::BumpingTeammate => native_bump::SCHEMA,
        }
    }

    pub fn feature_names(self) -> &'static [&'static str] {
        match self {
            Self::BumpingTeammate => native_bump::FEATURES,
        }
    }

    pub fn validate_contract(self, schema: &str, feature_names: &[String]) -> Result<(), String> {
        if schema != self.schema() || feature_names != self.feature_names() {
            return Err(format!("incompatible feature contract for {}", self.name()));
        }
        Ok(())
    }

    pub fn candidates(self, replay: &ReplayFacts) -> Result<CandidateBatch, String> {
        replay.validate()?;
        let candidates = match self {
            Self::BumpingTeammate => native_bump::candidates(replay)?,
        };
        let batch = CandidateBatch {
            kind: self.name().into(),
            input_schema: self.schema().into(),
            feature_names: self
                .feature_names()
                .iter()
                .map(|name| (*name).into())
                .collect(),
            engine_revision: replay.revision.clone(),
            replay_sha256: replay.replay_sha256.clone(),
            candidates,
        };
        batch.validate()?;
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
    pub features: Vec<f64>,
    pub evidence: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateBatch {
    pub kind: String,
    pub input_schema: String,
    pub feature_names: Vec<String>,
    pub engine_revision: String,
    pub replay_sha256: String,
    pub candidates: Vec<Candidate>,
}

impl CandidateBatch {
    pub fn validate(&self) -> Result<(), String> {
        let kind = MistakeKind::parse(&self.kind)?;
        kind.validate_contract(&self.input_schema, &self.feature_names)?;
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
                || candidate.features.iter().any(|value| !value.is_finite())
            {
                return Err("invalid candidate measurements or identity".into());
            }
            if !incidents.insert(candidate.anchor()?.key()?) {
                return Err("duplicate incident anchor".into());
            }
            let mut ids: Vec<_> = candidate.source_event_ids.iter().collect();
            ids.sort();
            if ids.is_empty()
                || ids.iter().any(|id| id.trim().is_empty())
                || ids.windows(2).any(|pair| pair[0] == pair[1])
                || !identities.insert(ids)
            {
                return Err("candidate event identities must be nonempty and unique".into());
            }
        }
        Ok(())
    }
}

impl Candidate {
    pub fn anchor(&self) -> Result<crate::annotations::IncidentAnchor, String> {
        match MistakeKind::parse(&self.kind)? {
            MistakeKind::BumpingTeammate => native_bump::anchor(self),
        }
    }
}
