use super::{ReplayContext, Samples, surrounding_players, team_basis, touch_id};
use crate::feature_context::{Context, possession_state, tag};
use crate::feature_input::Event;
use crate::feature_values::{Measurement, Values};
use serde_json::Value;
use std::collections::HashSet;

fn opponent_at_first_touch(first: &[&Event], team: bool) -> Result<Option<Value>, String> {
    let mut opponent = None;
    for event in first {
        if event.boolean("is_team_0")? == team {
            continue;
        }
        let player = event.participant("player")?;
        if opponent.is_some_and(|current| current != player) {
            return Ok(None);
        }
        opponent = Some(player);
    }
    Ok(opponent.cloned())
}

fn possession_duration(
    context: &Context<'_>,
    time: Option<f64>,
    samples: &Samples,
    team: bool,
) -> Result<Option<f64>, String> {
    let Some(time) = time else {
        return Ok(None);
    };
    let mut duration = None;
    for event in context.events("possession") {
        let start = event.number("time")?;
        let end = event.number("end_time")?;
        if end < start {
            return Err("reversed native possession span".into());
        }
        let terminal = time == samples.play_end && time == end;
        if start >= end || !event.boolean("active")? || start > time || (time >= end && !terminal) {
            continue;
        }
        if duration.is_some() {
            return Err("overlapping possession spans at the following touch".into());
        }
        duration = Some(
            if possession_state(event.text("possession_state")?, team)? == "opponent" {
                end.min(samples.play_end) - start.max(samples.play_start)
            } else {
                0.0
            },
        );
    }
    Ok(duration)
}

fn shadowing(
    context: &Context<'_>,
    player: &Value,
    frame: usize,
    time: f64,
) -> Result<Option<bool>, String> {
    match context.state("player_activity", Some(player), time)? {
        Some("demolished") => Ok(Some(false)),
        Some("tracked") if context.player(player)?.frames[frame].body().is_some() => {
            match context.state("shadow_defense", Some(player), time)? {
                None => Ok(Some(false)),
                Some("shadowing") => Ok(Some(true)),
                Some(state) => Err(format!("unsupported shadow-defense state: {state}")),
            }
        }
        None | Some("tracked") => Ok(None),
        Some(state) => Err(format!("invalid player activity state: {state}")),
    }
}

pub(super) fn features(
    replay: &ReplayContext<'_>,
    touch: &Event,
    samples: &mut Samples,
    first: &[&Event],
    values: &mut Values,
) -> Result<(), String> {
    let context = &replay.context;
    let player = touch.participant("player")?;
    let team = touch.boolean("is_team_0")?;
    let frame = touch.frame("frame")?;
    let time = touch.number("time")?;
    let mut consecutive = true;
    let mut preceding = HashSet::new();
    let mut previous_opponent = None;
    for (_, contacts) in replay.touches.range(..frame).rev() {
        let contact_time = contacts[0].number("time")?;
        if contact_time < samples.play_start {
            break;
        }
        let mut subject_only = true;
        let mut opponent = false;
        for contact in contacts {
            subject_only &= contact.participant("player")? == player;
            opponent |= contact.boolean("is_team_0")? != team;
        }
        consecutive &= subject_only;
        if consecutive {
            for contact in contacts {
                preceding.insert(touch_id(contact)?);
            }
        }
        if opponent {
            previous_opponent = Some(contact_time);
            break;
        }
    }
    values.0.insert(
        "subject_consecutive_touches_before_candidate".into(),
        Some(Measurement::Count(
            u32::try_from(preceding.len()).map_err(|error| error.to_string())?,
        )),
    );
    values.float(
        "previous_opponent_touch_delay_s",
        previous_opponent.map(|previous| time - previous),
    )?;
    values.float(
        "next_opponent_possession_duration_s",
        possession_duration(
            context,
            samples.first_touch.as_ref().map(|stage| stage.raw_time),
            samples,
            team,
        )?,
    )?;
    values.float(
        "subject_height_at_candidate",
        context.player(player)?.frames[frame]
            .body()
            .map(|body| body.location.z),
    )?;
    values.category(
        "subject_field_third_at_candidate",
        context.state("field_third", Some(player), time)?,
        &["defensive", "offensive"],
        "neutral",
    )?;
    values.category(
        "candidate_surface",
        tag(touch, "surface")?,
        &["air", "wall"],
        "ground",
    )?;
    let subject_shadowing = shadowing(context, player, frame, time)?;
    values.boolean("subject_shadow_defense_at_candidate", subject_shadowing);
    let mut team_shadowing = Some(false);
    for (id, _) in &context.analysis.outputs.replay_data.frame_data.players {
        if replay.roster[&id.to_string()] == team {
            team_shadowing = team_shadowing
                .zip(shadowing(context, id, frame, time)?)
                .map(|(observed, shadowing)| observed || shadowing);
        }
    }
    values.boolean("own_team_shadow_defense_at_candidate", team_shadowing);
    let closest =
        surrounding_players(context, &replay.roster, player, team, Some(frame))?.closest_opponent;
    values.float(
        "closest_opponent_speed_at_candidate",
        closest
            .as_ref()
            .and_then(|sample| sample.body.linear_velocity)
            .map(|velocity| velocity.length()),
    )?;
    samples.closest_opponent_at_candidate_id = closest.map(|sample| sample.player_id.clone());
    samples.first_touch_opponent_id = opponent_at_first_touch(first, team)?;
    let opponent_body = samples
        .first_touch_opponent_id
        .as_ref()
        .zip(samples.first_touch.as_ref())
        .map(|(id, stage)| {
            context
                .player(id)
                .map(|data| data.frames[stage.frame].body())
        })
        .transpose()?
        .flatten();
    values.vector(
        "first_touch_opponent_heading",
        opponent_body
            .map(|body| body.rotation.forward())
            .transpose()?,
        Some(team_basis(team)),
    )?;
    values.vector(
        "first_touch_opponent_up",
        opponent_body.map(|body| body.rotation.up()).transpose()?,
        Some(team_basis(team)),
    )?;
    Ok(())
}
