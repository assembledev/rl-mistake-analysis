use crate::feature_context::{Context, possession_state};
use crate::feature_geometry::goal_opening_distance;
use crate::feature_input::{Body, Metadata, Vector};
use serde_json::Value;
use std::collections::HashMap;

pub(crate) fn team_basis(team_zero: bool) -> Vector {
    Vector {
        x: 0.0,
        y: if team_zero { 1.0 } else { -1.0 },
        z: 0.0,
    }
}

pub(crate) struct OpponentSample<'a> {
    pub(crate) player_id: &'a Value,
    player_key: String,
    pub(crate) body: &'a Body,
    pub(crate) distance: f64,
}

pub(crate) struct Surroundings<'a> {
    pub(crate) approaching: [Option<u32>; 2],
    pub(crate) closest_opponent: Option<OpponentSample<'a>>,
}

pub(crate) fn surrounding_players<'a>(
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

pub(crate) fn displacement(
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

#[derive(Clone, Copy)]
pub(crate) struct TeamDistance {
    pub(crate) observed: bool,
    pub(crate) nearest: Option<f64>,
}

pub(crate) fn team_distances(
    context: &Context<'_>,
    roster: &HashMap<String, bool>,
) -> Result<Vec<[TeamDistance; 2]>, String> {
    let frames = &context.analysis.outputs.replay_data.frame_data;
    let players = frames
        .players
        .iter()
        .map(|(id, data)| {
            Ok((
                usize::from(roster[&id.to_string()]),
                data,
                context.states_by_frame("player_activity", id)?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    (0..frames.metadata_frames.len())
        .map(|frame| {
            let Some(ball) = context.ball(Some(frame)) else {
                return Ok([TeamDistance {
                    observed: false,
                    nearest: None,
                }; 2]);
            };
            let mut distances = [TeamDistance {
                observed: true,
                nearest: None,
            }; 2];
            for (team, data, activity) in &players {
                let team = *team;
                match activity[frame] {
                    Some("demolished") => continue,
                    Some("tracked") => (),
                    None => {
                        distances[team].observed = false;
                        continue;
                    }
                    Some(state) => return Err(format!("invalid player activity state: {state}")),
                }
                let Some(car) = data.frames[frame].body() else {
                    distances[team].observed = false;
                    continue;
                };
                let distance = ball.location.subtract(car.location).length();
                if !distance.is_finite() {
                    return Err("nonfinite car-ball distance".into());
                }
                distances[team].nearest = Some(
                    distances[team]
                        .nearest
                        .map_or(distance, |old| old.min(distance)),
                );
            }
            Ok(distances)
        })
        .collect()
}

pub(crate) fn minimum_team_distance(values: impl Iterator<Item = TeamDistance>) -> Option<f64> {
    let mut nearest: Option<f64> = None;
    for value in values {
        if !value.observed {
            return None;
        }
        if let Some(distance) = value.nearest {
            nearest = Some(nearest.map_or(distance, |old| old.min(distance)));
        }
    }
    nearest
}

pub(crate) fn mean_team_distance(
    context: &Context<'_>,
    distances: &[[TeamDistance; 2]],
    range: Option<(usize, usize)>,
    team: bool,
) -> Option<f64> {
    time_weighted_mean(
        &context
            .analysis
            .outputs
            .replay_data
            .frame_data
            .metadata_frames,
        range,
        |frame| {
            let value = distances[frame][usize::from(team)];
            value.nearest.filter(|_| value.observed)
        },
    )
}

pub(crate) fn time_weighted_mean(
    frames: &[Metadata],
    range: Option<(usize, usize)>,
    mut value: impl FnMut(usize) -> Option<f64>,
) -> Option<f64> {
    let (start, end) = range?;
    let duration = frames[end].time - frames[start].time;
    if duration == 0.0 {
        return None;
    }
    let mut previous = value(start)?;
    let mut integral = 0.0;
    for frame in start + 1..=end {
        let current = value(frame)?;
        let elapsed = frames[frame].time - frames[frame - 1].time;
        integral += (previous + current) * elapsed / 2.0;
        previous = current;
    }
    Some(integral / duration)
}

pub(crate) fn minimum_goal_distance(
    context: &Context<'_>,
    range: Option<(usize, usize)>,
    team: bool,
) -> Result<Option<f64>, String> {
    let Some((start, end)) = range else {
        return Ok(None);
    };
    let mut nearest: Option<f64> = None;
    for frame in start..=end {
        let Some(ball) = context.ball(Some(frame)) else {
            return Ok(None);
        };
        let distance = goal_opening_distance(ball.location, team)?;
        nearest = Some(nearest.map_or(distance, |old| old.min(distance)));
    }
    Ok(nearest)
}

pub(crate) fn possession_durations(
    context: &Context<'_>,
    start: f64,
    end: f64,
    team: bool,
) -> Result<Option<[f64; 3]>, String> {
    let mut spans = context
        .events("possession")
        .iter()
        .map(|event| Ok((event.number("time")?, event.number("end_time")?, *event)))
        .collect::<Result<Vec<_>, String>>()?;
    spans.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut duration = [0.0; 3];
    let mut cursor = start;
    for (span_start, span_end, event) in spans {
        let left = start.max(span_start);
        let right = end.min(span_end);
        if left >= right {
            continue;
        }
        if left < cursor {
            return Err("overlapping possession spans during exchange".into());
        }
        if left > cursor {
            return Ok(None);
        }
        let index = match possession_state(event.text("possession_state")?, team)? {
            "own_team" => 0,
            "opponent" => 1,
            "neutral" => 2,
            _ => unreachable!(),
        };
        duration[index] += right - left;
        cursor = right;
    }
    Ok((cursor == end).then_some(duration))
}
