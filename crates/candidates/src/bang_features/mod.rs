//! Touch-stage context for native bang-with-time candidates.

mod exchange;
mod schema;
mod touch_context;

pub use crate::feature_input::ReplayAnalysis;
pub use crate::feature_values::Measurement;
pub use schema::{DataType, FeatureDefinition, definitions};

use crate::feature_context::{Context, possession_state, tag};
use crate::feature_contract::{self, ContextFeatures};
pub use crate::feature_contract::{FeatureSampling, SamplingMethod};
use crate::feature_input::{Body, Event, Vector};
use crate::feature_values::Values;
use crate::{Candidate, CandidateBatch, MistakeKind};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

pub const SCHEMA: &str = "bang_with_time.context.v1";

pub type FeatureBatch = feature_contract::FeatureBatch<FeatureRow>;

#[derive(Debug, Serialize)]
pub struct FeatureRow {
    pub source_event_id: String,
    pub frame: usize,
    pub touch_id: u64,
    pub raw_time: f64,
    pub player_id: Value,
    pub samples: Samples,
    pub features: BTreeMap<String, Option<Measurement>>,
}

#[derive(Debug, Serialize)]
pub struct Samples {
    pub play_start: f64,
    pub play_end: f64,
    pub before_frame: Option<usize>,
    pub outgoing_sample_frame: Option<usize>,
    pub closest_opponent_id: Option<Value>,
    pub closest_opponent_at_candidate_id: Option<Value>,
    pub first_touch_opponent_id: Option<Value>,
    pub previous_touch: Option<TouchStage>,
    pub first_touch: Option<TouchStage>,
    pub second_touch: Option<TouchStage>,
}

#[derive(Debug, Serialize)]
pub struct TouchStage {
    pub frame: usize,
    pub raw_time: f64,
    pub source_event_ids: Vec<String>,
    pub touch_ids: Vec<u64>,
}

fn touch_id(event: &Event) -> Result<u64, String> {
    event
        .field("touch_id")?
        .as_u64()
        .ok_or_else(|| format!("invalid touch ID: {}", event.meta.id))
}

fn stage(events: &[&Event]) -> Result<Option<TouchStage>, String> {
    let Some(first) = events.first() else {
        return Ok(None);
    };
    Ok(Some(TouchStage {
        frame: first.frame("frame")?,
        raw_time: first.number("time")?,
        source_event_ids: events.iter().map(|event| event.meta.id.clone()).collect(),
        touch_ids: events
            .iter()
            .map(|event| touch_id(event))
            .collect::<Result<_, _>>()?,
    }))
}

fn team_basis(team_zero: bool) -> Vector {
    Vector {
        x: 0.0,
        y: if team_zero { 1.0 } else { -1.0 },
        z: 0.0,
    }
}

fn stage_features(
    context: &Context<'_>,
    prefix: &str,
    events: &[&Event],
    candidate: &Event,
    start: f64,
    values: &mut Values,
) -> Result<(), String> {
    let subject = candidate.participant("player")?;
    let team = candidate.boolean("is_team_0")?;
    let time = candidate.number("time")?;
    let frame = events
        .first()
        .map(|event| event.frame("frame"))
        .transpose()?;
    if prefix != "candidate" {
        for relation in schema::RELATIONS {
            let mut found = false;
            for event in events {
                let actor = if event.participant("player")? == subject {
                    "self"
                } else if event.boolean("is_team_0")? == team {
                    "teammate"
                } else {
                    "opponent"
                };
                found |= actor == *relation;
            }
            values.boolean(&format!("{prefix}_{relation}"), frame.map(|_| found));
        }
        values.float(
            &format!("{prefix}_delay_s"),
            events
                .first()
                .map(|event| {
                    event.number("time").map(|next| {
                        if prefix == "previous_touch" {
                            time - next
                        } else {
                            next - time
                        }
                    })
                })
                .transpose()?,
        )?;
    }
    if prefix == "previous_touch" {
        return Ok(());
    }
    values.vector(
        &format!("{prefix}_ball_position"),
        context.ball(frame).map(|body| body.location),
        Some(team_basis(team)),
    )?;
    let incoming = frame.and_then(|frame| context.previous_frame(frame, start));
    values.float(
        &format!("{prefix}_incoming_ball_speed"),
        context
            .ball(incoming)
            .and_then(|body| body.linear_velocity)
            .map(Vector::length),
    )?;
    for event in events {
        context.validate_touch_sample(event)?;
    }
    values.float(
        &format!("{prefix}_outgoing_ball_speed"),
        context
            .ball(frame)
            .and_then(|body| body.linear_velocity)
            .map(Vector::length),
    )?;
    values.category(
        &format!("subject_ball_depth_at_{prefix}"),
        events
            .first()
            .map(|event| {
                event
                    .number("time")
                    .and_then(|time| context.state("ball_depth", Some(subject), time))
            })
            .transpose()?
            .flatten(),
        &["level_with_ball", "ahead_of_ball"],
        "behind_ball",
    )?;
    let actions = events
        .iter()
        .map(|event| tag(event, "action").map(|action| action.unwrap_or("unclassified")))
        .collect::<Result<Vec<_>, _>>()?;
    if actions
        .iter()
        .any(|action| !schema::ACTIONS.contains(action))
    {
        return Err("unsupported native touch action".into());
    }
    for action in schema::ACTIONS {
        values.boolean(
            &format!("{prefix}_action_{action}"),
            frame.map(|_| actions.contains(action)),
        );
    }
    Ok(())
}

struct OpponentSample<'a> {
    player_id: &'a Value,
    player_key: String,
    body: &'a Body,
    distance: f64,
}

struct Surroundings<'a> {
    approaching: [Option<u32>; 2],
    closest_opponent: Option<OpponentSample<'a>>,
}

fn surrounding_players<'a>(
    context: &Context<'a>,
    roster: &HashMap<String, bool>,
    subject: &Value,
    team: bool,
    frame: Option<usize>,
) -> Result<Surroundings<'a>, String> {
    let frames = &context.analysis.outputs.replay_data.frame_data;
    let observation = frame.zip(context.ball(frame));
    let mut counts = [observation.map(|_| 0u32); 2];
    let mut nearest: Option<OpponentSample<'_>> = None;
    if let Some((frame, ball)) = observation {
        let time = frames.metadata_frames[frame].time;
        let mut opponent_missing = false;
        for (id, data) in &frames.players {
            if id == subject {
                continue;
            }
            let player_key = id.to_string();
            let opponent = roster[&player_key] != team;
            let role = usize::from(opponent);
            match context.state("player_activity", Some(id), time)? {
                Some("demolished") => continue,
                Some("tracked") => (),
                None => {
                    counts[role] = None;
                    opponent_missing |= opponent;
                    continue;
                }
                Some(state) => return Err(format!("invalid player activity state: {state}")),
            }
            let native_frame = &data.frames[frame];
            let Some(body) = native_frame.body() else {
                counts[role] = None;
                opponent_missing |= opponent;
                continue;
            };
            let to_ball = ball.location.subtract(body.location);
            let distance = to_ball.length();
            if !distance.is_finite() {
                return Err("nonfinite car-ball distance".into());
            }
            if let Some(velocity) = body.linear_velocity {
                let approach = velocity.dot(to_ball);
                if !approach.is_finite() {
                    return Err("nonfinite car approach measurement".into());
                }
                counts[role] = counts[role].map(|count| count + u32::from(approach > 0.0));
            } else {
                counts[role] = None;
            }
            if opponent
                && nearest.as_ref().is_none_or(|previous| {
                    distance < previous.distance
                        || (distance == previous.distance && player_key < previous.player_key)
                })
            {
                nearest = Some(OpponentSample {
                    player_id: id,
                    player_key,
                    body,
                    distance,
                });
            }
        }
        if opponent_missing {
            nearest = None;
        }
    }
    Ok(Surroundings {
        approaching: counts,
        closest_opponent: nearest,
    })
}

fn surrounding_features(
    context: &Context<'_>,
    roster: &HashMap<String, bool>,
    subject: &Value,
    team: bool,
    frame: Option<usize>,
    values: &mut Values,
) -> Result<Option<Value>, String> {
    let Surroundings {
        approaching,
        closest_opponent: nearest,
    } = surrounding_players(context, roster, subject, team, frame)?;
    let observation = frame.zip(context.ball(frame));
    for (role, count) in ["teammate", "opponent"].into_iter().zip(approaching) {
        values.0.insert(
            format!("approaching_{role}_count"),
            count.map(Measurement::Count),
        );
    }
    let basis = Some(team_basis(team));
    values.vector(
        "closest_opponent_position_relative_ball",
        nearest
            .as_ref()
            .zip(observation)
            .map(|(opponent, (_, ball))| opponent.body.location.subtract(ball.location)),
        basis,
    )?;
    values.vector(
        "closest_opponent_heading",
        nearest
            .as_ref()
            .map(|opponent| opponent.body.rotation.forward())
            .transpose()?,
        basis,
    )?;
    values.vector(
        "closest_opponent_velocity_relative_ball",
        nearest
            .as_ref()
            .zip(observation)
            .and_then(|(opponent, (_, ball))| {
                opponent
                    .body
                    .linear_velocity
                    .zip(ball.linear_velocity)
                    .map(|(car, ball)| car.subtract(ball))
            }),
        basis,
    )?;
    let boost = nearest
        .as_ref()
        .map(|opponent| context.boost_before(opponent.player_id, frame))
        .transpose()?
        .flatten();
    values.float("closest_opponent_boost_before_pct", boost)?;
    Ok(nearest.map(|opponent| opponent.player_id.clone()))
}

fn displacement(
    context: &Context<'_>,
    subject: &Value,
    before: Option<usize>,
    end: Option<usize>,
) -> Result<Option<Vector>, String> {
    let Some((before, end)) = before.zip(end) else {
        return Ok(None);
    };
    for event in context.events("demolition") {
        if event.participant("victim")? == subject
            && event.frame("frame")? >= before
            && event.frame("frame")? <= end
        {
            return Ok(None);
        }
    }
    let frames = &context.player(subject)?.frames;
    if frames[before..=end]
        .iter()
        .any(|frame| frame.body().is_none())
    {
        return Ok(None);
    }
    Ok(Some(
        frames[end]
            .body()
            .unwrap()
            .location
            .subtract(frames[before].body().unwrap().location),
    ))
}

struct ReplayContext<'a> {
    context: Context<'a>,
    roster: HashMap<String, bool>,
    touches: BTreeMap<usize, Vec<&'a Event>>,
    team_distances: Vec<[exchange::TeamDistance; 2]>,
}

impl<'a> ReplayContext<'a> {
    fn new(context: Context<'a>) -> Result<Self, String> {
        let analysis = context.analysis;
        let frames = &analysis.outputs.replay_data.frame_data;
        let native_roster = analysis
            .outputs
            .events
            .replay_meta
            .as_ref()
            .ok_or("bang features require events.replay_meta")?;
        let mut roster = HashMap::new();
        for (team, players) in [
            (true, &native_roster.team_zero),
            (false, &native_roster.team_one),
        ] {
            for player in players {
                if player.remote_id.is_null()
                    || roster.insert(player.remote_id.to_string(), team).is_some()
                {
                    return Err("invalid or duplicate native roster player".into());
                }
                context.player(&player.remote_id)?;
            }
        }
        if frames
            .players
            .iter()
            .any(|(id, _)| !roster.contains_key(&id.to_string()))
        {
            return Err("player frames are absent from native roster".into());
        }
        let mut groups: BTreeMap<usize, Vec<&Event>> = BTreeMap::new();
        for touch in context.events("touch") {
            let frame = touch.frame("frame")?;
            let time = touch.number("time")?;
            if frames
                .metadata_frames
                .get(frame)
                .is_none_or(|metadata| metadata.time != time)
            {
                return Err(format!(
                    "touch {} disagrees with native frame time",
                    touch.meta.id
                ));
            }
            if roster.get(&touch.participant("player")?.to_string())
                != Some(&touch.boolean("is_team_0")?)
            {
                return Err("touch team disagrees with native roster".into());
            }
            groups.entry(frame).or_default().push(touch);
        }
        for group in groups.values_mut() {
            group.sort_by_key(|event| event.meta.id.as_str());
        }
        let team_distances = exchange::team_distances(&context, &roster)?;
        Ok(Self {
            context,
            roster,
            touches: groups,
            team_distances,
        })
    }

    fn row(
        &self,
        candidate: &Candidate,
        definitions: &[FeatureDefinition],
    ) -> Result<FeatureRow, String> {
        let context = &self.context;
        let frames = &context.analysis.outputs.replay_data.frame_data;
        let frame = usize::try_from(candidate.frame).map_err(|error| error.to_string())?;
        let touch = self.touches[&frame]
            .iter()
            .find(|event| candidate.source_event_ids == [event.meta.id.clone()])
            .ok_or("candidate source touch is missing")?;
        let time = touch.number("time")?;
        let player = touch.participant("player")?;
        let team = touch.boolean("is_team_0")?;
        let (play_start, play_end) = context.bounds(time)?;
        let previous = self
            .touches
            .range(..frame)
            .next_back()
            .filter(|(frame, _)| frames.metadata_frames[**frame].time >= play_start)
            .map(|(_, events)| events.as_slice())
            .unwrap_or(&[]);
        let later = self
            .touches
            .range((std::ops::Bound::Excluded(frame), std::ops::Bound::Unbounded))
            .take_while(|(frame, _)| frames.metadata_frames[**frame].time <= play_end)
            .map(|(_, events)| events.as_slice())
            .take(2)
            .collect::<Vec<_>>();
        let first = later.first().copied().unwrap_or(&[]);
        let second = later.get(1).copied().unwrap_or(&[]);
        let before = context.previous_frame(frame, play_start);
        let mut samples = Samples {
            play_start,
            play_end,
            before_frame: before,
            outgoing_sample_frame: Some(frame),
            closest_opponent_id: None,
            closest_opponent_at_candidate_id: None,
            first_touch_opponent_id: None,
            previous_touch: stage(previous)?,
            first_touch: stage(first)?,
            second_touch: stage(second)?,
        };
        let mut values = Values::default();
        for (prefix, events) in [
            ("candidate", std::slice::from_ref(touch)),
            ("previous_touch", previous),
            ("first_touch", first),
            ("second_touch", second),
        ] {
            stage_features(context, prefix, events, touch, play_start, &mut values)?;
        }
        subject_state_features(context, touch, &samples, &mut values)?;
        samples.closest_opponent_id =
            surrounding_features(context, &self.roster, player, team, before, &mut values)?;
        exchange_features(context, touch, &samples, first, &mut values)?;
        possession_features(context, touch, play_end, &mut values)?;
        context.goal_features(time, play_end, team, &mut values)?;
        subject_followup_features(context, touch, &samples, &mut values)?;
        exchange::features(self, touch, &samples, &mut values)?;
        touch_context::features(self, touch, &mut samples, first, &mut values)?;
        values.validate(definitions)?;
        Ok(FeatureRow {
            source_event_id: touch.meta.id.clone(),
            frame,
            touch_id: touch_id(touch)?,
            raw_time: time,
            player_id: player.clone(),
            samples,
            features: values.0,
        })
    }
}

fn subject_state_features(
    context: &Context<'_>,
    touch: &Event,
    samples: &Samples,
    values: &mut Values,
) -> Result<(), String> {
    let player = touch.participant("player")?;
    let time = touch.number("time")?;
    let frame = touch.frame("frame")?;
    let before = samples.before_frame;
    let play_start = samples.play_start;
    values.float(
        "subject_boost_before_pct",
        context.boost_before(player, before)?,
    )?;
    for (stream, name, indicators, reference) in [
        ("field_half", "field_half", &["offensive"][..], "defensive"),
        (
            "depth_role",
            "depth_role",
            &["most_back", "most_forward", "mid", "no_teammates"][..],
            "other",
        ),
    ] {
        values.category(
            &format!("subject_{name}_at_candidate"),
            context.state(stream, Some(player), time)?,
            indicators,
            reference,
        )?;
    }
    values.category(
        "subject_height_band_at_candidate",
        context.height_band(player, Some(frame))?,
        &["low_air", "high_air"],
        "ground",
    )?;
    let dodge = tag(touch, "dodge_state")?.ok_or("touch is missing native dodge state")?;
    if !["dodge", "no_dodge"].contains(&dodge) {
        return Err("invalid native dodge state".into());
    }
    values.boolean("candidate_dodging", Some(dodge == "dodge"));
    for (feature, stream, participant) in [
        ("ground_carry_before", "ball_carry", "player_id"),
        ("air_dribble_before", "air_dribble", "player_id"),
        ("receiving_pass_before", "pass", "receiver"),
    ] {
        values.boolean(
            feature,
            Some(context.active_interval(stream, participant, player, time, Some(play_start))?),
        );
    }
    Ok(())
}

fn exchange_features(
    context: &Context<'_>,
    touch: &Event,
    samples: &Samples,
    first: &[&Event],
    values: &mut Values,
) -> Result<(), String> {
    let player = touch.participant("player")?;
    let team = touch.boolean("is_team_0")?;
    let frame = touch.frame("frame")?;
    let mut challenged = false;
    for event in context.events("beaten_to_ball") {
        challenged |= event.frame("frame")? == frame
            && event.participant("winner")? == player
            && event.boolean("is_team_0")? != team;
    }
    values.boolean("opponent_beaten_at_candidate", Some(challenged));
    let end_frame = samples.second_touch.as_ref().map(|stage| stage.frame);
    let end_time = samples.second_touch.as_ref().map(|stage| stage.raw_time);
    for (stream, feature, participant) in [
        ("whiff", "opponent_whiff_during_exchange", "player"),
        ("beaten_to_ball", "subject_beaten_during_exchange", "player"),
        ("demolition", "subject_demo_during_exchange", "victim"),
    ] {
        let mut found = false;
        if let Some(end) = end_frame {
            for event in context.events(stream) {
                let event_frame = event.frame("frame")?;
                if event_frame < frame
                    || (event_frame == frame && stream != "demolition")
                    || event_frame > end
                {
                    continue;
                }
                let qualifies = if stream == "whiff" {
                    event.text("kind")? == "whiff"
                        && event.boolean("is_team_0")? != team
                        && event.number("resolved_time")? <= end_time.unwrap()
                } else {
                    event.participant(participant)? == player
                };
                found |= qualifies;
            }
        }
        values.boolean(feature, end_frame.map(|_| found));
    }
    let mut opponent_control = false;
    for event in first {
        opponent_control |= event.boolean("is_team_0")? != team
            && matches!(tag(event, "possession")?, Some("control" | "advance"));
    }
    values.boolean(
        "first_touch_opponent_control",
        samples.first_touch.as_ref().map(|_| opponent_control),
    );
    Ok(())
}

fn possession_features(
    context: &Context<'_>,
    touch: &Event,
    play_end: f64,
    values: &mut Values,
) -> Result<(), String> {
    let time = touch.number("time")?;
    let team = touch.boolean("is_team_0")?;
    let before_possession = context
        .state("possession", None, time)?
        .map(|state| possession_state(state, team))
        .transpose()?;
    values.category(
        "possession_before",
        before_possession,
        &["own_team", "opponent", "neutral"],
        "unavailable",
    )?;
    let change = context.possession_change(time, play_end, team, before_possession)?;
    values.category(
        "next_possession",
        before_possession.map(|_| change.map_or("no_change", |(_, state)| state)),
        &["own_team", "opponent", "neutral", "no_change"],
        "unavailable",
    )?;
    values.float(
        "possession_change_delay_s",
        change.map(|(start, _)| start - time),
    )?;
    Ok(())
}

fn subject_followup_features(
    context: &Context<'_>,
    touch: &Event,
    samples: &Samples,
    values: &mut Values,
) -> Result<(), String> {
    let player = touch.participant("player")?;
    let team = touch.boolean("is_team_0")?;
    let time = touch.number("time")?;
    let frame = touch.frame("frame")?;
    let play_end = samples.play_end;
    let mut next = None::<f64>;
    for event in context.events("touch") {
        let touch_time = event.number("time")?;
        if event.frame("frame")? > frame
            && touch_time <= play_end
            && event.participant("player")? == player
            && next.is_none_or(|previous| touch_time < previous)
        {
            next = Some(touch_time);
        }
    }
    values.float("subject_next_touch_delay_s", next.map(|next| next - time))?;
    let player_data = context.player(player)?;
    for (name, stage) in [
        ("first_touch", &samples.first_touch),
        ("second_touch", &samples.second_touch),
    ] {
        let stage_frame = stage.as_ref().map(|stage| stage.frame);
        let player_body = stage_frame.and_then(|frame| player_data.frames[frame].body());
        values.float(
            &format!("subject_ball_distance_at_{name}"),
            player_body
                .zip(context.ball(stage_frame))
                .map(|(car, ball)| ball.location.subtract(car.location).length()),
        )?;
        values.vector(
            &format!("subject_heading_at_{name}"),
            player_body
                .map(|body| body.rotation.forward())
                .transpose()?,
            Some(team_basis(team)),
        )?;
        values.vector(
            &format!("subject_displacement_to_{name}"),
            displacement(context, player, Some(frame), stage_frame)?,
            Some(team_basis(team)),
        )?;
    }
    Ok(())
}

/// Extract native touch candidates and their context measurements in matching order.
pub fn extract(
    analysis: &ReplayAnalysis,
    sampling: FeatureSampling,
    subject: Option<&Value>,
) -> Result<(CandidateBatch, FeatureBatch), String> {
    sampling.validate(SamplingMethod::NativeTouchStages)?;
    let context = Context::new(
        analysis,
        &[
            "touch",
            "possession",
            "goal_context",
            "kickoff",
            "whiff",
            "bump",
            "beaten_to_ball",
            "demolition",
            "ball_carry",
            "air_dribble",
            "pass",
            "player_activity",
            "movement",
            "field_half",
            "field_third",
            "shadow_defense",
            "ball_depth",
            "depth_role",
        ],
    )?;
    let mut candidates = MistakeKind::BangWithTime.candidates(&analysis.facts())?;
    let replay = ReplayContext::new(context)?;
    if let Some(subject) = subject {
        replay.context.player(subject)?;
        candidates
            .candidates
            .retain(|candidate| candidate.player_id == *subject);
    }
    let feature_definitions = definitions();
    let rows = candidates
        .candidates
        .iter()
        .map(|candidate| replay.row(candidate, &feature_definitions))
        .collect::<Result<_, _>>()?;
    let features = FeatureBatch {
        input_schema: SCHEMA,
        engine_revision: analysis.revision.clone(),
        replay_sha256: analysis.replay_sha256.clone(),
        sampling,
        feature_definitions,
        rows,
    };
    Ok((candidates, features))
}

impl ContextFeatures for FeatureRow {
    fn key(&self) -> (&str, &Value) {
        (&self.source_event_id, &self.player_id)
    }

    fn validate_candidate(&self, candidate: &Candidate) -> Result<(), String> {
        if self.frame as u64 != candidate.frame
            || self.raw_time != candidate.raw_time
            || candidate.evidence.get("touch_id").and_then(Value::as_u64) != Some(self.touch_id)
        {
            return Err("subject feature row disagrees with the candidate".into());
        }
        Ok(())
    }

    fn features(&self) -> &BTreeMap<String, Option<Measurement>> {
        &self.features
    }
}
