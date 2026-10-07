use crate::annotations::IncidentAnchor;
use crate::input::ReplayFacts;
use crate::kind::Candidate;
use serde::Deserialize;
use serde_json::Value;

pub const SCHEMA: &str = "bumping_teammate.native.v1";
pub const FEATURES: &[&str] = &[
    "contact_confidence",
    "closing_speed",
    "victim_impulse",
    "contact_distance",
    "strength",
    "initiator_height",
    "victim_height",
    "victim_distance_to_own_goal",
];

#[derive(Debug, Clone, Deserialize)]
struct BumpEvent {
    kind: String,
    payload: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct Bump {
    is_team_bump: bool,
    initiator_is_team_0: bool,
    victim_is_team_0: bool,
    initiator: Value,
    victim: Value,
    frame: u64,
    time: f64,
    initiator_position: [f64; 3],
    victim_position: [f64; 3],
    confidence: f64,
    closing_speed: f64,
    victim_impulse: f64,
    contact_distance: f64,
    strength: f64,
}

pub(crate) fn anchor(candidate: &Candidate) -> Result<IncidentAnchor, String> {
    let bump: Bump = serde_json::from_value(candidate.evidence.clone())
        .map_err(|error| format!("invalid bump evidence: {error}"))?;
    if candidate.frame != bump.frame
        || candidate.raw_time != bump.time
        || candidate.player_id != bump.initiator
        || !bump.is_team_bump
        || bump.initiator_is_team_0 != bump.victim_is_team_0
        || bump.initiator.is_null()
        || bump.victim.is_null()
        || bump.initiator == bump.victim
    {
        return Err("bump anchor does not agree with its event evidence".into());
    }
    Ok(IncidentAnchor::BumpingTeammate {
        frame: bump.frame,
        initiator: bump.initiator,
        victim: bump.victim,
    })
}

pub(crate) fn candidates(replay: &ReplayFacts) -> Result<Vec<Candidate>, String> {
    let mut candidates = Vec::new();
    for event in replay
        .events
        .iter()
        .filter(|event| event.meta.stream == "bump")
    {
        let parsed: BumpEvent = serde_json::from_value(event.payload.clone())
            .map_err(|error| format!("invalid bump event: {error}"))?;
        if parsed.kind != "bump" {
            return Err("invalid bump event kind".into());
        }
        let bump: Bump = serde_json::from_value(parsed.payload.clone())
            .map_err(|error| format!("invalid bump measurements: {error}"))?;
        if bump.is_team_bump != (bump.initiator_is_team_0 == bump.victim_is_team_0) {
            return Err("inconsistent native bump teams".into());
        }
        if !bump.is_team_bump {
            continue;
        }
        if bump.initiator.is_null() || bump.victim.is_null() || bump.initiator == bump.victim {
            return Err("missing or identical bump participants".into());
        }
        let goal_y = if bump.initiator_is_team_0 {
            -5120.0
        } else {
            5120.0
        };
        let features = vec![
            bump.confidence,
            bump.closing_speed,
            bump.victim_impulse,
            bump.contact_distance,
            bump.strength,
            bump.initiator_position[2],
            bump.victim_position[2],
            bump.victim_position[0].hypot(bump.victim_position[1] - goal_y),
        ];
        if !bump.time.is_finite()
            || bump
                .initiator_position
                .iter()
                .any(|number| !number.is_finite())
            || bump
                .victim_position
                .iter()
                .any(|number| !number.is_finite())
            || features.iter().any(|number| !number.is_finite())
        {
            return Err("nonfinite native bump measurement".into());
        }
        let evidence = parsed.payload;
        candidates.push(Candidate {
            kind: "bumping_teammate".into(),
            source_event_ids: vec![event.meta.id.clone()],
            frame: bump.frame,
            raw_time: bump.time,
            player_id: bump.initiator,
            features,
            evidence,
        });
    }
    candidates.sort_by(|left, right| {
        left.frame
            .cmp(&right.frame)
            .then(left.source_event_ids.cmp(&right.source_event_ids))
    });
    Ok(candidates)
}
