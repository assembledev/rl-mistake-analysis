//! Defensive approaches with later ball-passage confirmation.

use crate::annotations::IncidentAnchor;
use crate::feature_context::{Context, tag};
use crate::feature_input::{Event, PlayerData, PlayerFrame};
use crate::{Candidate, feature_contract::ReplayAnalysis};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub const SCHEMA: &str = "overcommitting.candidates.v1";

struct Span<'a> {
    start: f64,
    end: f64,
    state: &'a str,
    active: bool,
}

struct Player<'a> {
    id: &'a Value,
    team: bool,
    frames: &'a PlayerData,
    activity: Vec<Span<'a>>,
}

impl Player<'_> {
    fn state(&self, time: f64) -> Option<&str> {
        let before = self.activity.partition_point(|span| span.start < time);
        before.checked_sub(1).and_then(|index| {
            let span = &self.activity[index];
            (time <= span.end).then_some(span.state)
        })
    }
}

struct Play {
    first: usize,
    end: usize,
}

struct Timeline<'a> {
    players: Vec<Player<'a>>,
    plays: Vec<Play>,
    possession: Vec<Span<'a>>,
    touches: BTreeMap<usize, Vec<&'a Event>>,
    shots: BTreeMap<usize, Vec<&'a Event>>,
    demolitions: HashMap<String, BTreeSet<usize>>,
    fifties: HashMap<String, Vec<(usize, usize)>>,
}

impl<'a> Timeline<'a> {
    fn new(context: &Context<'a>) -> Result<Self, String> {
        let analysis = context.analysis;
        let frames = &analysis.outputs.replay_data.frame_data;
        let roster = analysis
            .outputs
            .events
            .replay_meta
            .as_ref()
            .ok_or("missing native roster")?;
        let mut players = Vec::new();
        let mut spans = Vec::new();
        for (team, members) in [(true, &roster.team_zero), (false, &roster.team_one)] {
            for member in members {
                if players
                    .iter()
                    .any(|player: &Player<'_>| player.id == &member.remote_id)
                {
                    return Err("duplicate native roster player".into());
                }
                let id = &member.remote_id;
                let mut activity = Vec::new();
                for event in context.events("player_activity") {
                    if event.participant("player")? == id {
                        let start = event.number("time")?;
                        let end = event.number("end_time")?;
                        if end < start {
                            return Err("reversed player activity span".into());
                        }
                        spans.push((start, end));
                        activity.push(Span {
                            start,
                            end,
                            state: event.text("state")?,
                            active: true,
                        });
                    }
                }
                activity.sort_by(|left, right| left.start.total_cmp(&right.start));
                if activity.windows(2).any(|pair| pair[0].end > pair[1].start) {
                    return Err("overlapping native player activity spans".into());
                }
                players.push(Player {
                    id,
                    team,
                    frames: context.player(id)?,
                    activity,
                });
            }
        }
        let mut possession = Vec::new();
        for event in context.events("possession") {
            possession.push(Span {
                start: event.number("time")?,
                end: event.number("end_time")?,
                state: event.text("possession_state")?,
                active: event.boolean("active")?,
            });
        }
        possession.sort_by(|left, right| left.start.total_cmp(&right.start));
        if possession.iter().any(|span| span.end < span.start)
            || possession
                .windows(2)
                .any(|pair| pair[0].end > pair[1].start)
        {
            return Err("invalid native possession spans".into());
        }
        spans.sort_by(|left, right| left.0.total_cmp(&right.0));
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (start, end) in spans {
            if let Some(previous) = merged.last_mut().filter(|previous| start <= previous.1) {
                previous.1 = previous.1.max(end);
            } else {
                merged.push((start, end));
            }
        }
        let mut plays = Vec::new();
        for (index, &(start, end)) in merged.iter().enumerate() {
            let first = frames
                .metadata_frames
                .partition_point(|frame| frame.time <= start);
            let stop = frames
                .metadata_frames
                .partition_point(|frame| frame.time <= end);
            if first >= stop {
                continue;
            }
            let next_start = merged.get(index + 1).map(|span| span.0);
            let mut goal = None;
            for event in context.events("goal_context") {
                let time = event.number("time")?;
                if time > start && next_start.is_none_or(|next| time < next) {
                    if goal.is_some() {
                        return Err("multiple native goals in one live play".into());
                    }
                    goal = Some(event.frame("frame")?);
                }
            }
            let end = match goal {
                Some(frame) => frame,
                None => stop - 1,
            };
            if end < first || end >= frames.metadata_frames.len() {
                return Err("invalid native play boundary".into());
            }
            plays.push(Play { first, end });
        }
        let mut touches: BTreeMap<usize, Vec<&Event>> = BTreeMap::new();
        let mut shots: BTreeMap<usize, Vec<&Event>> = BTreeMap::new();
        for event in context.events("touch") {
            let frame = event.frame("frame")?;
            event.participant("player")?;
            event.boolean("is_team_0")?;
            touches.entry(frame).or_default().push(event);
            if tag(event, "action")? == Some("shot") {
                shots.entry(frame).or_default().push(event);
            }
        }
        for event in context.events("timeline") {
            if event.text("kind")? == "Shot" {
                event.boolean("is_team_0")?;
                shots.entry(event.frame("frame")?).or_default().push(event);
            }
        }
        let mut demolitions: HashMap<String, BTreeSet<usize>> = HashMap::new();
        for event in context.events("demolition") {
            demolitions
                .entry(event.participant("victim")?.to_string())
                .or_default()
                .insert(event.frame("frame")?);
        }
        let mut fifties: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        for event in context.events("fifty_fifty") {
            let first = event.frame("start_frame")?;
            let last = event.frame("resolve_frame")?;
            if last < first {
                return Err("reversed native fifty interval".into());
            }
            for field in ["team_zero_player", "team_one_player"] {
                fifties
                    .entry(event.participant(field)?.to_string())
                    .or_default()
                    .push((first, last));
            }
        }
        for (&frame, events) in touches.iter().chain(&shots) {
            let metadata = frames
                .metadata_frames
                .get(frame)
                .ok_or("native contact or shot frame is outside the replay")?;
            for event in events {
                if event.number("time")? != metadata.time {
                    return Err("native contact or shot time disagrees with its frame".into());
                }
            }
        }
        Ok(Self {
            players,
            plays,
            possession,
            touches,
            shots,
            demolitions,
            fifties,
        })
    }

    fn opponent_context(
        &self,
        player: &Player<'_>,
        play: &Play,
        frame: usize,
        time: f64,
    ) -> Result<bool, String> {
        let before = self.possession.partition_point(|span| span.start <= time);
        let Some(index) = before.checked_sub(1) else {
            return Ok(false);
        };
        let span = &self.possession[index];
        let opponent = if player.team { "team_one" } else { "team_zero" };
        if !span.active || time >= span.end || (span.state != "neutral" && span.state != opponent) {
            return Ok(false);
        }
        let Some((_, touches)) = self.touches.range(play.first..=frame).next_back() else {
            return Ok(false);
        };
        for touch in touches {
            if touch.boolean("is_team_0")? == player.team {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[derive(Default)]
struct Observation {
    front: Option<bool>,
    opening: bool,
    deepest: bool,
    farthest_goal_side: bool,
}

struct Attempt {
    start: usize,
    opening: Observation,
    opening_y: f64,
    end: Option<usize>,
}

fn observe(
    analysis: &ReplayAnalysis,
    timeline: &Timeline<'_>,
    player: &Player<'_>,
    play: &Play,
    frame: usize,
    demolished: bool,
) -> Result<Observation, String> {
    let frames = &analysis.outputs.replay_data.frame_data;
    let time = frames.metadata_frames[frame].time;
    let mut result = Observation::default();
    if demolished || player.state(time) == Some("demolished") {
        return Ok(result);
    }
    let (Some(car), Some(ball)) = (
        player.frames.frames[frame].body(),
        frames.ball_data.frames[frame].body(),
    ) else {
        return Ok(result);
    };
    let sign = if player.team { 1.0 } else { -1.0 };
    let delta = ball.location.subtract(car.location);
    let front = sign * delta.y >= 0.0;
    result.front = Some(front);
    if player.state(time) != Some("tracked") || !front {
        return Ok(result);
    }
    let (Some(velocity), Some(ball_velocity)) = (car.linear_velocity, ball.linear_velocity) else {
        return Ok(result);
    };
    if sign * velocity.y <= 0.0 || velocity.dot(delta) <= 0.0 || sign * ball_velocity.y >= 0.0 {
        return Ok(result);
    }
    if !timeline.opponent_context(player, play, frame, time)? {
        return Ok(result);
    }
    let mut deepest = true;
    let mut farthest = true;
    for teammate in &timeline.players {
        if teammate.team != player.team || teammate.id == player.id {
            continue;
        }
        let state = teammate.state(time);
        if state == Some("demolished") {
            continue;
        }
        let Some(other) = teammate.frames.frames[frame]
            .body()
            .filter(|_| state == Some("tracked"))
        else {
            return Ok(result);
        };
        deepest &= sign * (other.location.y - car.location.y) >= 0.0;
        let other_delta = ball.location.subtract(other.location);
        if sign * other_delta.y >= 0.0 {
            farthest &= other_delta.dot(other_delta) <= delta.dot(delta);
        }
    }
    result.deepest = deepest;
    result.farthest_goal_side = farthest;
    result.opening = deepest || farthest;
    Ok(result)
}

pub(crate) fn candidates(analysis: &ReplayAnalysis) -> Result<Vec<Candidate>, String> {
    let context = Context::new(
        analysis,
        &[
            "touch",
            "possession",
            "player_activity",
            "fifty_fifty",
            "demolition",
            "goal_context",
            "timeline",
        ],
    )?;
    let timeline = Timeline::new(&context)?;
    let frames = &analysis.outputs.replay_data.frame_data.metadata_frames;
    let mut result = Vec::new();
    for player in &timeline.players {
        for play in &timeline.plays {
            let mut own = BTreeSet::new();
            for (&frame, group) in timeline.touches.range(play.first..=play.end) {
                for event in group {
                    if event.boolean("is_team_0")? == player.team {
                        own.insert(frame);
                    }
                }
            }
            let identity = player.id.to_string();
            let sign = if player.team { 1.0 } else { -1.0 };
            let mut opened: Option<Attempt> = None;
            for frame in play.first..=play.end {
                let demo = timeline
                    .demolitions
                    .get(&identity)
                    .is_some_and(|values| values.contains(&frame));
                let fifty = timeline.fifties.get(&identity).is_some_and(|values| {
                    values
                        .iter()
                        .any(|&(start, end)| start <= frame && frame <= end)
                });
                let state = observe(analysis, &timeline, player, play, frame, demo)?;
                let active = player.state(frames[frame].time) == Some("tracked");
                let lifecycle = frame == play.end || demo;
                if opened.is_some()
                    && (!active
                        || state.front != Some(true)
                        || own.contains(&frame)
                        || fifty
                        || lifecycle)
                {
                    let attempt = opened.take().unwrap();
                    let start = attempt.start;
                    if active && state.front == Some(false) && !own.contains(&frame) && !fifty {
                        let next_touch = own
                            .range((std::ops::Bound::Excluded(start), std::ops::Bound::Unbounded))
                            .next()
                            .copied();
                        let horizon = match next_touch {
                            Some(next) => next - 1,
                            None => play.end,
                        };
                        let mut contacts = Vec::new();
                        let mut shots = Vec::new();
                        for (_, group) in timeline.touches.range(start..=frame) {
                            for event in group {
                                if event.boolean("is_team_0")? != player.team {
                                    contacts.push(*event);
                                }
                            }
                        }
                        for (_, group) in timeline.shots.range(start..=horizon) {
                            for event in group {
                                if event.boolean("is_team_0")? != player.team {
                                    shots.push(*event);
                                }
                            }
                        }
                        if !contacts.is_empty() || !shots.is_empty() {
                            let (end, finish) = match attempt.end {
                                Some(end) => (end, "retreat_to_opening_depth"),
                                None => (frame, "ball_passage"),
                            };
                            let opening = attempt.opening;
                            let latest = timeline
                                .touches
                                .range(play.first..=start)
                                .next_back()
                                .unwrap()
                                .1;
                            let mut source_event_ids: Vec<_> = latest
                                .iter()
                                .chain(contacts.iter())
                                .chain(shots.iter())
                                .map(|event| event.meta.id.clone())
                                .collect();
                            source_event_ids.sort();
                            source_event_ids.dedup();
                            result.push(Candidate {
                                kind: "overcommitting".into(),
                                source_event_ids,
                                frame: start as u64,
                                raw_time: frames[start].time,
                                player_id: player.id.clone(),
                                features: Vec::new(),
                                evidence: json!({
                                    "start_frame": start,
                                    "end_frame": end,
                                    "start_time": frames[start].time,
                                    "end_time": frames[end].time,
                                    "passage_frame": frame,
                                    "passage_time": frames[frame].time,
                                    "player": player.id,
                                    "is_team_0": player.team,
                                    "play_start_frame": play.first,
                                    "play_end_frame": play.end,
                                    "finish": finish,
                                    "opening": {
                                        "position_y": attempt.opening_y,
                                        "deepest_by_y": opening.deepest,
                                        "farthest_goal_side": opening.farthest_goal_side,
                                        "last_touch_ids": latest.iter().map(|event| &event.meta.id).collect::<Vec<_>>(),
                                    },
                                    "opponent_touch_ids": contacts.iter().map(|event| &event.meta.id).collect::<Vec<_>>(),
                                    "opponent_shot_ids": shots.iter().map(|event| &event.meta.id).collect::<Vec<_>>(),
                                    "next_own_touch_frame": next_touch,
                                }),
                            });
                        }
                    }
                }
                if opened.is_none() && active && state.opening && !lifecycle && !fifty {
                    opened = Some(Attempt {
                        start: frame,
                        opening: state,
                        opening_y: player.frames.frames[frame].body().unwrap().location.y,
                        end: None,
                    });
                }
                if let Some(attempt) = opened.as_mut()
                    && attempt.end.is_none()
                    && let PlayerFrame::Data {
                        rigid_body,
                        dodge_active: false,
                        ..
                    } = &player.frames.frames[frame]
                    && let Some(velocity) = rigid_body.linear_velocity
                    && sign * velocity.y < 0.0
                    && sign * (rigid_body.location.y - attempt.opening_y) <= 0.0
                {
                    attempt.end = Some(frame);
                }
            }
        }
    }
    result.sort_by(|left, right| {
        left.frame
            .cmp(&right.frame)
            .then_with(|| left.player_id.to_string().cmp(&right.player_id.to_string()))
    });
    Ok(result)
}

#[derive(Deserialize)]
struct Evidence {
    start_frame: u64,
    end_frame: u64,
    start_time: f64,
    end_time: f64,
    passage_frame: u64,
    passage_time: f64,
    player: Value,
    finish: Finish,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Finish {
    BallPassage,
    RetreatToOpeningDepth,
}

pub(crate) fn anchor(candidate: &Candidate) -> Result<IncidentAnchor, String> {
    let evidence: Evidence = serde_json::from_value(candidate.evidence.clone())
        .map_err(|error| format!("invalid overcommitting evidence: {error}"))?;
    if evidence.start_frame != candidate.frame
        || evidence.start_time != candidate.raw_time
        || evidence.player != candidate.player_id
        || evidence.player.is_null()
        || evidence.end_frame <= evidence.start_frame
        || !evidence.start_time.is_finite()
        || !evidence.end_time.is_finite()
        || evidence.end_time < evidence.start_time
        || evidence.passage_frame < evidence.end_frame
        || !evidence.passage_time.is_finite()
        || evidence.passage_time < evidence.end_time
        || match evidence.finish {
            Finish::BallPassage => {
                evidence.end_frame != evidence.passage_frame
                    || evidence.end_time != evidence.passage_time
            }
            Finish::RetreatToOpeningDepth => evidence.end_frame >= evidence.passage_frame,
        }
    {
        return Err("overcommitting interval does not agree with its candidate anchor".into());
    }
    Ok(IncidentAnchor::Overcommitting {
        frame: evidence.start_frame,
        end_frame: evidence.end_frame,
    })
}
