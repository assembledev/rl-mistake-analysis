use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReplayFacts {
    pub name: String,
    pub schema_version: u32,
    pub status: String,
    pub revision: String,
    pub replay_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rocket_league_id: Option<String>,
    pub events: Vec<MechanicEvent>,
}

impl ReplayFacts {
    pub fn validate(&self) -> Result<(), String> {
        if self.name != "subtr-actor" || self.schema_version != 1 || self.status != "ok" {
            return Err("native mechanic engine unavailable or incompatible".into());
        }
        if self.revision.trim().is_empty() {
            return Err("missing engine revision".into());
        }
        if !is_sha256(&self.replay_sha256) {
            return Err("invalid replay_sha256".into());
        }
        if self
            .rocket_league_id
            .as_ref()
            .is_some_and(|id| id.trim().is_empty())
        {
            return Err("empty intrinsic replay ID".into());
        }
        let mut ids = HashSet::new();
        for event in &self.events {
            if event.meta.id.trim().is_empty()
                || event.meta.stream.trim().is_empty()
                || !ids.insert(&event.meta.id)
            {
                return Err("event IDs must be nonempty and unique".into());
            }
        }
        Ok(())
    }
}

pub fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MechanicEvent {
    pub meta: EventMeta,
    pub payload: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EventMeta {
    pub stream: String,
    pub id: String,
}
