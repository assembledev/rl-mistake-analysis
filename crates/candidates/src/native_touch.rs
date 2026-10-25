//! Touches immediately before an opponent contact in the same play.

use crate::{Candidate, ReplayFacts, annotations::IncidentAnchor};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;

pub const SCHEMA: &str = "bang_with_time.candidates.v2";

#[derive(Deserialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
enum NativeEvent {
    Touch(Touch),
    FiftyFifty(Fifty),
    GoalContext { time: f64 },
    Kickoff { start_time: f64 },
}

#[derive(Deserialize)]
struct Tag {
    group: String,
    value: String,
}

#[derive(Deserialize)]
struct Touch {
    touch_id: u64,
    frame: u64,
    time: f64,
    player: Value,
    is_team_0: bool,
    tags: Vec<Tag>,
}

#[derive(Deserialize)]
struct Fifty {
    start_frame: u64,
    resolve_frame: u64,
}

fn validate_player(player: &Value) -> Result<(), String> {
    if player.as_object().is_none_or(|id| id.is_empty()) {
        return Err("touches require player identities".into());
    }
    Ok(())
}

pub(crate) fn anchor(candidate: &Candidate) -> Result<IncidentAnchor, String> {
    let touch: Touch = serde_json::from_value(candidate.evidence.clone())
        .map_err(|error| format!("invalid native touch evidence: {error}"))?;
    validate_player(&touch.player)?;
    if !touch.time.is_finite()
        || candidate.frame != touch.frame
        || candidate.raw_time != touch.time
        || candidate.player_id != touch.player
    {
        return Err("touch anchor does not agree with its event evidence".into());
    }
    Ok(IncidentAnchor::BangWithTime {
        frame: touch.frame,
        touch_id: touch.touch_id,
    })
}

pub(crate) fn select(replay: &ReplayFacts) -> Result<Vec<Candidate>, String> {
    let mut touches = Vec::new();
    let mut touch_ids = HashSet::new();
    let mut fifties = Vec::<Fifty>::new();
    let mut goals = Vec::new();
    let mut kickoffs = Vec::new();
    for event in &replay.events {
        let stream = event.meta.stream.as_str();
        if !matches!(stream, "touch" | "fifty_fifty" | "goal_context" | "kickoff") {
            continue;
        }
        if event.payload["kind"] != stream {
            return Err(format!("event payload disagrees with stream: {stream}"));
        }
        let native = serde_json::from_value(event.payload.clone())
            .map_err(|error| format!("invalid native {stream}: {error}"))?;
        match native {
            NativeEvent::Touch(touch) => {
                validate_player(&touch.player)?;
                if !touch.time.is_finite() || !touch_ids.insert(touch.touch_id) {
                    return Err("invalid touch time or duplicate touch ID".into());
                }
                touches.push((touch, event));
            }
            NativeEvent::FiftyFifty(fifty) => {
                if fifty.resolve_frame < fifty.start_frame {
                    return Err("fifty has reversed boundaries".into());
                }
                fifties.push(fifty);
            }
            NativeEvent::GoalContext { time } => goals.push(time),
            NativeEvent::Kickoff { start_time } => kickoffs.push(start_time),
        }
    }
    if goals.iter().chain(&kickoffs).any(|time| !time.is_finite()) {
        return Err("play boundaries require finite native times".into());
    }
    touches.sort_by_key(|(touch, _)| (touch.frame, touch.touch_id));
    if touches.windows(2).any(|pair| {
        pair[0].0.time > pair[1].0.time
            || (pair[0].0.frame == pair[1].0.frame && pair[0].0.time != pair[1].0.time)
    }) {
        return Err("touch times disagree with native contact-frame order".into());
    }
    let eligible = |touch: &Touch| {
        !fifties
            .iter()
            .any(|fifty| fifty.start_frame <= touch.frame && touch.frame <= fifty.resolve_frame)
            && !touch
                .tags
                .iter()
                .any(|tag| tag.group == "contested" && tag.value == "contested")
    };
    let mut stages = touches
        .chunk_by(|left, right| left.0.frame == right.0.frame)
        .peekable();
    let mut candidates = Vec::new();
    while let Some(current) = stages.next() {
        let Some(next) = stages.peek() else {
            break;
        };
        let time = current[0].0.time;
        let next_time = next[0].0.time;
        if !next.iter().all(|(touch, _)| eligible(touch))
            || goals.iter().any(|goal| time <= *goal && *goal < next_time)
            || kickoffs
                .iter()
                .any(|kickoff| time < *kickoff && *kickoff <= next_time)
        {
            continue;
        }
        for (touch, event) in current {
            if !eligible(touch)
                || !next
                    .iter()
                    .any(|(opponent, _)| opponent.is_team_0 != touch.is_team_0)
            {
                continue;
            }
            candidates.push(Candidate {
                kind: "bang_with_time".into(),
                source_event_ids: vec![event.meta.id.clone()],
                frame: touch.frame,
                raw_time: touch.time,
                player_id: touch.player.clone(),
                features: Vec::new(),
                evidence: event.payload["payload"].clone(),
            });
        }
    }
    Ok(candidates)
}
