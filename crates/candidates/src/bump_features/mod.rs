//! Subject-specific teammate-bump measurements from combined replay analysis.

mod schema;

pub use crate::feature_contract::{FeatureSampling, SamplingMethod};
pub use crate::feature_input::ReplayAnalysis;
pub use crate::feature_values::Measurement;
use crate::impact_sampling as sampling;
pub use crate::impact_sampling::{ParticipantSamples, SamplingStatus};
pub use schema::{DataType, FeatureDefinition, definitions};

use crate::Candidate;
use crate::bump_contacts::{Contact, unique_contact_ids};
use crate::feature_context::{Context, boost_percent, possession_state, tag};
use crate::feature_contract::{self, ContextFeatures};
use crate::feature_input::{Body, Event, PadEventKind, PlayerData, PlayerFrame, Vector};
use crate::feature_values::Values;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub const SCHEMA: &str = "bumping_teammate.context.v1";

/// Named v1 subsets use the same measurements as the base context contract.
pub fn supports_schema(schema: &str) -> bool {
    feature_contract::supports_schema(SCHEMA, schema)
}

/// Resolve unique model inputs in their declared order.
pub fn select_definitions(names: &[String]) -> Result<Vec<FeatureDefinition>, String> {
    feature_contract::select_definitions(definitions(), names)
}

pub type FeatureBatch = feature_contract::FeatureBatch<FeatureRow>;

#[derive(Debug, Serialize)]
pub struct FeatureRow {
    pub source_event_id: String,
    pub frame: usize,
    pub raw_time: f64,
    pub player_id: Value,
    pub teammate_id: Value,
    pub initiator: Value,
    pub victim: Value,
    pub samples: SampleFrames,
    pub features: BTreeMap<String, Option<Measurement>>,
}

#[derive(Debug, Serialize)]
pub struct SampleFrames {
    pub play_start: f64,
    pub play_end: f64,
    pub p: ParticipantSamples,
    pub q: ParticipantSamples,
}

impl Context<'_> {
    fn samples(
        &self,
        bump: &Event,
        player: &Value,
        teammate: &Value,
    ) -> Result<SampleFrames, String> {
        let time = bump.number("time")?;
        let frame = bump.frame("frame")?;
        let (play_start, play_end) = self.bounds(time)?;
        let frames = &self.analysis.outputs.replay_data.frame_data.metadata_frames;
        Ok(SampleFrames {
            play_start,
            play_end,
            p: sampling::measure(self.player(player)?, frames, frame, play_start, play_end),
            q: sampling::measure(self.player(teammate)?, frames, frame, play_start, play_end),
        })
    }
}

/// Compute both participant perspectives, or only the explicitly selected player.
pub fn extract(
    analysis: &ReplayAnalysis,
    sampling: FeatureSampling,
    subject: Option<&Value>,
) -> Result<FeatureBatch, String> {
    sampling.validate(SamplingMethod::FirstAccelerationPeakThenMinimum)?;
    let mut context = Context::new(
        analysis,
        &[
            "bump",
            "touch",
            "whiff",
            "beaten_to_ball",
            "possession",
            "goal_context",
            "kickoff",
            "movement",
            "ball_carry",
            "air_dribble",
            "pass",
            "shadow_defense",
            "field_half",
            "ball_depth",
            "depth_role",
            "boost_pickups",
            "demolition",
        ],
    )?;
    let mut contacts = Vec::new();
    for bump in context.events("bump") {
        let initiator = bump.participant("initiator")?;
        let victim = bump.participant("victim")?;
        let team = bump.boolean("initiator_is_team_0")?;
        let team_bump = bump.boolean("is_team_bump")?;
        if team_bump != (team == bump.boolean("victim_is_team_0")?) || initiator == victim {
            return Err(format!(
                "event {} has inconsistent bump participants or teams",
                bump.meta.id
            ));
        }
        if !team_bump {
            continue;
        }
        let frame = bump.frame("frame")?;
        let time = bump.number("time")?;
        if analysis
            .outputs
            .replay_data
            .frame_data
            .metadata_frames
            .get(frame)
            .is_none_or(|metadata| metadata.time != time)
        {
            return Err(format!(
                "event {} disagrees with native frame time",
                bump.meta.id
            ));
        }
        contacts.push(Contact {
            id: &bump.meta.id,
            frame: frame as u64,
            initiator,
            victim,
        });
    }
    let selected = unique_contact_ids(contacts.into_iter())?;
    if let Some(bumps) = context.streams.get_mut("bump") {
        bumps.retain(|bump| {
            bump.boolean("is_team_bump") == Ok(false) || selected.contains(&bump.meta.id)
        });
    }
    if let Some(subject) = subject {
        context.player(subject)?;
    }
    let feature_definitions = definitions();
    let mut rows = Vec::new();
    for bump in context.events("bump") {
        if !bump.boolean("is_team_bump")? {
            continue;
        }
        let initiator = bump.participant("initiator")?;
        let victim = bump.participant("victim")?;
        let team = bump.boolean("initiator_is_team_0")?;
        let frame = bump.frame("frame")?;
        let time = bump.number("time")?;
        for (player, teammate) in [(initiator, victim), (victim, initiator)] {
            if subject.is_some_and(|subject| subject != player) {
                continue;
            }
            let samples = context.samples(bump, player, teammate)?;
            let mut values = Values::default();
            global_features(
                &context,
                bump,
                player,
                teammate,
                team,
                &samples,
                &mut values,
            )?;
            participant_features(&context, "p", player, teammate, bump, &samples, &mut values)?;
            participant_features(&context, "q", teammate, player, bump, &samples, &mut values)?;
            values.validate(&feature_definitions)?;
            rows.push(FeatureRow {
                source_event_id: bump.meta.id.clone(),
                frame,
                raw_time: time,
                player_id: player.clone(),
                teammate_id: teammate.clone(),
                initiator: initiator.clone(),
                victim: victim.clone(),
                samples,
                features: values.0,
            });
        }
    }
    rows.sort_by(|left, right| {
        left.frame
            .cmp(&right.frame)
            .then_with(|| left.source_event_id.cmp(&right.source_event_id))
            .then_with(|| left.player_id.to_string().cmp(&right.player_id.to_string()))
    });
    Ok(FeatureBatch {
        input_schema: SCHEMA,
        engine_revision: analysis.revision.clone(),
        replay_sha256: analysis.replay_sha256.clone(),
        sampling,
        feature_definitions,
        rows,
    })
}

fn actor<'a>(
    event: &Event,
    player: &Value,
    teammate: &Value,
    team: bool,
) -> Result<&'a str, String> {
    let toucher = event.participant("player")?;
    Ok(if toucher == player {
        "p"
    } else if toucher == teammate {
        "q"
    } else if event.boolean("is_team_0")? == team {
        "other_teammate"
    } else {
        "opponent"
    })
}

fn global_features(
    context: &Context<'_>,
    bump: &Event,
    player: &Value,
    teammate: &Value,
    team: bool,
    samples: &SampleFrames,
    values: &mut Values,
) -> Result<(), String> {
    let time = bump.number("time")?;
    values.boolean(
        "p_is_initiator",
        Some(bump.participant("initiator")? == player),
    );
    values.float("contact_gap", Some(bump.number("contact_distance")?))?;
    let next_touch = context.next("touch", "time", time, samples.play_end, None)?;
    for group in ["p", "q", "other_teammate", "opponent"] {
        let mut touched = false;
        if let Some(next_touch) = next_touch {
            let frame = next_touch.frame("frame")?;
            for touch in context.events("touch") {
                if touch.frame("frame")? == frame && actor(touch, player, teammate, team)? == group
                {
                    touched = true;
                }
            }
        }
        values.boolean(&format!("first_touch_{group}"), Some(touched));
    }
    values.float(
        "first_touch_delay_s",
        next_touch
            .map(|event| event.number("time").map(|next| next - time))
            .transpose()?,
    )?;
    let before = context
        .state("possession", None, time)?
        .ok_or("missing pre-bump possession state")?;
    let before = possession_state(before, team)?;
    values.category(
        "possession_before",
        Some(before),
        &["own_team", "opponent"],
        "neutral",
    )?;
    let change = context.possession_change(time, samples.play_end, team, Some(before))?;
    values.category(
        "next_possession",
        Some(match change {
            Some((_, state)) => state,
            None => "no_change",
        }),
        &["own_team", "opponent", "neutral"],
        "no_change",
    )?;
    values.float(
        "possession_change_delay_s",
        change.map(|(start, _)| start - time),
    )?;
    context.goal_features(time, samples.play_end, team, values)?;
    Ok(())
}

fn body(player: &PlayerData, frame: Option<usize>) -> Option<&Body> {
    frame.and_then(|frame| player.frames[frame].body())
}

fn physical_features(
    context: &Context<'_>,
    prefix: &str,
    player: &Value,
    other: &Value,
    samples: &ParticipantSamples,
    values: &mut Values,
) -> Result<(), String> {
    let player_data = context.player(player)?;
    let before = body(player_data, samples.before_frame);
    let after = body(player_data, samples.after_frame);
    let other_before = body(context.player(other)?, samples.before_frame);
    let toward = before
        .zip(other_before)
        .and_then(|(left, right)| right.location.subtract(left.location).horizontal_unit());
    let velocity_before = before.and_then(|body| body.linear_velocity);
    let velocity_after = after.and_then(|body| body.linear_velocity);
    values.vector(
        &format!("{prefix}_velocity_before"),
        velocity_before,
        toward,
    )?;
    values.vector(
        &format!("{prefix}_velocity_delta"),
        velocity_after
            .zip(velocity_before)
            .map(|(after, before)| after.subtract(before)),
        toward,
    )?;
    values.vector(
        &format!("{prefix}_position_delta"),
        after
            .zip(before)
            .map(|(after, before)| after.location.subtract(before.location)),
        toward,
    )?;
    let balls = &context
        .analysis
        .outputs
        .replay_data
        .frame_data
        .ball_data
        .frames;
    let ball_before = samples.before_frame.and_then(|frame| balls[frame].body());
    let ball_after = samples.after_frame.and_then(|frame| balls[frame].body());
    let distance_before = before
        .zip(ball_before)
        .map(|(car, ball)| car.location.horizontal_distance(ball.location));
    let distance_after = after
        .zip(ball_after)
        .map(|(car, ball)| car.location.horizontal_distance(ball.location));
    values.float(&format!("{prefix}_ball_distance_before"), distance_before)?;
    values.float(
        &format!("{prefix}_ball_distance_delta"),
        distance_after
            .zip(distance_before)
            .map(|(after, before)| after - before),
    )?;
    let heading = before
        .map(|body| body.rotation.forward())
        .transpose()?
        .and_then(Vector::horizontal_unit);
    values.float(
        &format!("{prefix}_heading_alignment_before"),
        heading
            .zip(toward)
            .map(|(heading, basis)| heading.project(basis)[0]),
    )?;
    values.float(
        &format!("{prefix}_upright_after"),
        after.map(|body| body.rotation.upright()).transpose()?,
    )?;
    Ok(())
}

fn participant_features(
    context: &Context<'_>,
    prefix: &str,
    player: &Value,
    other: &Value,
    bump: &Event,
    samples: &SampleFrames,
    values: &mut Values,
) -> Result<(), String> {
    let time = bump.number("time")?;
    let frame = bump.frame("frame")?;
    let physical_samples = match prefix {
        "p" => &samples.p,
        "q" => &samples.q,
        _ => return Err("unsupported participant feature prefix".into()),
    };
    physical_features(context, prefix, player, other, physical_samples, values)?;
    let next_touch = context.next(
        "touch",
        "time",
        time,
        samples.play_end,
        Some(("player", player)),
    )?;
    values.float(
        &format!("{prefix}_next_touch_delay_s"),
        next_touch
            .map(|event| event.number("time").map(|next| next - time))
            .transpose()?,
    )?;
    let action = match next_touch.map(|event| tag(event, "action")).transpose()? {
        None => "no_touch",
        Some(None) => "unclassified",
        Some(Some(action)) => action,
    };
    values.category(
        &format!("{prefix}_next_touch_action"),
        Some(action),
        &["unclassified", "shot", "save", "pass", "clear", "boom"],
        "no_touch",
    )?;
    let cutoff = match next_touch {
        Some(event) => event.number("time")?,
        None => samples.play_end,
    };
    let mut whiff = false;
    for event in context.events("whiff") {
        if event.participant("player")? == player
            && event.text("kind")? == "whiff"
            && event.number("time")? > time
            && event.number("resolved_time")? < cutoff
        {
            whiff = true;
        }
    }
    let mut beaten = false;
    for event in context.events("beaten_to_ball") {
        if event.participant("player")? == player
            && event.number("time")? > time
            && event.number("time")? < cutoff
        {
            beaten = true;
        }
    }
    values.boolean(&format!("{prefix}_whiff_before_next_touch"), Some(whiff));
    values.boolean(&format!("{prefix}_beaten_before_next_touch"), Some(beaten));
    for (name, field) in [
        ("advance", "advance_distance"),
        ("retreat", "retreat_distance"),
    ] {
        let distance = match next_touch {
            Some(event) => match event.payload.payload.get("ball_movement") {
                None | Some(Value::Null) => None,
                Some(movement) => Some(
                    movement
                        .get(field)
                        .and_then(Value::as_f64)
                        .ok_or("invalid touch ball-movement measurement")?,
                ),
            },
            None => None,
        };
        values.float(&format!("{prefix}_next_touch_ball_{name}"), distance)?;
    }
    for (stream, feature, participant) in [
        ("ball_carry", "ground_carry_before", "player_id"),
        ("air_dribble", "air_dribble_before", "player_id"),
        ("pass", "receiving_pass_before", "receiver"),
        ("shadow_defense", "shadowing_before", "player"),
    ] {
        values.boolean(
            &format!("{prefix}_{feature}"),
            Some(context.active_interval(stream, participant, player, time, None)?),
        );
    }

    let player_data = context.player(player)?;
    let previous_frame = frame.checked_sub(1);
    let dodging = previous_frame.and_then(|frame| match &player_data.frames[frame] {
        PlayerFrame::Data { dodge_active, .. } => Some(*dodge_active),
        PlayerFrame::Empty => None,
    });
    values.boolean(&format!("{prefix}_dodging_before"), dodging);
    values.category(
        &format!("{prefix}_field_half_before"),
        context.state("field_half", Some(player), time)?,
        &["offensive"],
        "defensive",
    )?;
    values.category(
        &format!("{prefix}_ball_depth_before"),
        context.state("ball_depth", Some(player), time)?,
        &["level_with_ball", "ahead_of_ball"],
        "behind_ball",
    )?;
    values.category(
        &format!("{prefix}_depth_role_before"),
        context.state("depth_role", Some(player), time)?,
        &["most_back", "most_forward", "mid", "no_teammates"],
        "other",
    )?;
    values.category(
        &format!("{prefix}_height_band_before"),
        context.height_band(player, previous_frame)?,
        &["low_air", "high_air"],
        "ground",
    )?;
    values.float(
        &format!("{prefix}_boost_before_pct"),
        context.boost_before(player, previous_frame)?,
    )?;
    let pickup = context.next(
        "boost_pickups",
        "time",
        time,
        samples.play_end,
        Some(("player_id", player)),
    )?;
    values.float(
        &format!("{prefix}_next_boost_pickup_delay_s"),
        pickup
            .map(|event| event.number("time").map(|next| next - time))
            .transpose()?,
    )?;
    values.float(
        &format!("{prefix}_next_boost_pickup_gain_pct"),
        pickup
            .map(|event| event.number("collected_amount").map(boost_percent))
            .transpose()?,
    )?;
    pad_distance(context, prefix, player, pickup, physical_samples, values)?;
    for (stream, feature) in [
        ("bump", "previous_bump_age_s"),
        ("demolition", "previous_demolition_age_s"),
    ] {
        let mut previous: Option<f64> = None;
        for event in context.events(stream) {
            let involved = event.participant("victim")? == player
                || (stream == "bump" && event.participant("initiator")? == player);
            let event_time = event.number("time")?;
            if involved
                && event_time >= samples.play_start
                && event_time < time
                && previous.is_none_or(|previous| event_time > previous)
            {
                previous = Some(event_time);
            }
        }
        values.float(
            &format!("{prefix}_{feature}"),
            previous.map(|previous| time - previous),
        )?;
    }
    Ok(())
}

fn pad_distance(
    context: &Context<'_>,
    prefix: &str,
    player: &Value,
    pickup: Option<&Event>,
    samples: &ParticipantSamples,
    values: &mut Values,
) -> Result<(), String> {
    let replay = &context.analysis.outputs.replay_data;
    let mut matches = Vec::new();
    if let Some(pickup) = pickup {
        for event in &replay.boost_pad_events {
            if event.player.as_ref() == Some(player)
                && event.frame == pickup.frame("frame")?
                && event.time == pickup.number("time")?
                && matches!(event.kind, PadEventKind::PickedUp { .. })
            {
                matches.push(event);
            }
        }
    }
    let distance = if matches.len() == 1 {
        let id = &matches[0].pad_id;
        let pads: Vec<_> = replay
            .boost_pads
            .iter()
            .filter(|pad| pad.pad_id.as_ref() == Some(id))
            .collect();
        if pads.len() > 1 {
            return Err(format!("duplicate resolved boost pad {id}"));
        }
        let player = context.player(player)?;
        pads.first()
            .zip(body(player, samples.before_frame).zip(body(player, samples.after_frame)))
            .map(|(pad, (before, after))| {
                after.location.horizontal_distance(pad.position)
                    - before.location.horizontal_distance(pad.position)
            })
    } else {
        None
    };
    values.float(&format!("{prefix}_next_pad_distance_delta"), distance)
}

impl ContextFeatures for FeatureRow {
    fn key(&self) -> (&str, &Value) {
        (&self.source_event_id, &self.player_id)
    }

    fn validate_candidate(&self, candidate: &Candidate) -> Result<(), String> {
        if self.frame as u64 != candidate.frame
            || self.raw_time != candidate.raw_time
            || candidate.evidence.get("initiator") != Some(&self.initiator)
            || candidate.evidence.get("victim") != Some(&self.victim)
            || self.features.get("p_is_initiator")
                != Some(&Some(Measurement::Boolean(
                    self.player_id == self.initiator,
                )))
        {
            return Err("subject feature row disagrees with the candidate".into());
        }
        Ok(())
    }

    fn features(&self) -> &BTreeMap<String, Option<Measurement>> {
        &self.features
    }
}
