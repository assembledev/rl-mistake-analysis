//! Portable annotations anchored to incidents in exact replay recordings.

use crate::{Candidate, MistakeKind, input::is_sha256};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IncidentAnchor {
    BumpingTeammate {
        frame: u64,
        initiator: Value,
        victim: Value,
    },
}

impl IncidentAnchor {
    pub fn kind(&self) -> MistakeKind {
        match self {
            Self::BumpingTeammate { .. } => MistakeKind::BumpingTeammate,
        }
    }

    pub fn key(&self) -> Result<String, String> {
        fn ordered(value: Value) -> Value {
            match value {
                Value::Object(values) => {
                    let mut entries: Vec<_> = values.into_iter().collect();
                    entries.sort_by(|left, right| left.0.cmp(&right.0));
                    Value::Object(entries.into_iter().map(|(k, v)| (k, ordered(v))).collect())
                }
                Value::Array(values) => Value::Array(values.into_iter().map(ordered).collect()),
                value => value,
            }
        }
        let value = serde_json::to_value(self).map_err(|error| error.to_string())?;
        serde_json::to_string(&ordered(value)).map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayReference {
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rocket_league_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download: Option<ReplayDownload>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReplayDownload {
    Ballchasing { id: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Annotation {
    pub replay_sha256: String,
    #[serde(flatten)]
    pub incident: IncidentAnchor,
    pub raw_time: f64,
    pub evidence: Value,
    pub label: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reject_reason: Option<String>,
    pub engine_revision: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_event_ids: Vec<String>,
}

impl Annotation {
    pub fn validate(&self) -> Result<(), String> {
        if !is_sha256(&self.replay_sha256)
            || self.label > 1
            || !self.raw_time.is_finite()
            || self.engine_revision.trim().is_empty()
            || self
                .reject_reason
                .as_ref()
                .is_some_and(|reason| reason.trim().is_empty())
        {
            return Err("invalid annotation fingerprint, label, time, or provenance".into());
        }
        let candidate = match &self.incident {
            IncidentAnchor::BumpingTeammate {
                frame, initiator, ..
            } => Candidate {
                kind: self.incident.kind().name().into(),
                frame: *frame,
                raw_time: self.raw_time,
                player_id: initiator.clone(),
                evidence: self.evidence.clone(),
                source_event_ids: self.source_event_ids.clone(),
                features: Vec::new(),
            },
        };
        if candidate.anchor()? != self.incident {
            return Err("annotation anchor does not agree with its event evidence".into());
        }
        let mut ids = HashSet::new();
        if self
            .source_event_ids
            .iter()
            .any(|id| id.trim().is_empty() || !ids.insert(id))
        {
            return Err("annotation source event IDs must be nonempty and unique".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationDataset {
    pub schema_version: u32,
    pub replays: Vec<ReplayReference>,
    pub annotations: Vec<Annotation>,
}

impl AnnotationDataset {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 || self.replays.is_empty() || self.annotations.is_empty() {
            return Err("invalid annotation schema or empty dataset".into());
        }
        let mut recordings = HashSet::new();
        for replay in &self.replays {
            if !is_sha256(&replay.sha256)
                || !recordings.insert(&replay.sha256)
                || replay
                    .rocket_league_id
                    .as_ref()
                    .is_some_and(|id| id.trim().is_empty())
            {
                return Err("invalid or duplicate replay reference".into());
            }
            if let Some(ReplayDownload::Ballchasing { id }) = &replay.download
                && (id.is_empty()
                    || id.len() > 64
                    || !id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
            {
                return Err("invalid Ballchasing replay ID".into());
            }
        }
        let mut incidents = HashSet::new();
        for annotation in &self.annotations {
            annotation.validate()?;
            if !recordings.contains(&annotation.replay_sha256) {
                return Err("annotation references an undeclared replay".into());
            }
            if !incidents.insert((&annotation.replay_sha256, annotation.incident.key()?)) {
                return Err("duplicate annotated incident".into());
            }
        }
        Ok(())
    }
}
