use super::{ReplayContext, Samples, team_basis};
use crate::feature_context::Context;
use crate::feature_geometry::directed_at_goal;
use crate::feature_input::Event;
use crate::feature_measurements::{
    mean_team_distance, minimum_goal_distance, minimum_team_distance, possession_durations,
};
use crate::feature_values::Values;

pub(super) fn features(
    replay: &ReplayContext<'_>,
    touch: &Event,
    samples: &Samples,
    values: &mut Values,
) -> Result<(), String> {
    let context = &replay.context;
    let team = touch.boolean("is_team_0")?;
    let frame = touch.frame("frame")?;
    let first = samples.first_touch.as_ref().map(|stage| stage.frame);
    let second = samples.second_touch.as_ref().map(|stage| stage.frame);
    values.float(
        "play_elapsed_s",
        Some(touch.number("time")? - samples.play_start),
    )?;
    values.vector(
        "candidate_outgoing_ball_velocity",
        context
            .ball(Some(frame))
            .and_then(|ball| ball.linear_velocity),
        Some(team_basis(team)),
    )?;
    for (name, attacking_team) in [("own", !team), ("opponent", team)] {
        values.boolean(
            &format!("candidate_directed_at_{name}_goal"),
            context
                .ball(Some(frame))
                .map(|ball| directed_at_goal(ball, attacking_team))
                .transpose()?
                .flatten(),
        );
    }
    for (name, range) in [
        ("candidate_to_first_touch", first.map(|end| (frame, end))),
        ("at_first_touch", first.map(|frame| (frame, frame))),
        ("first_to_second_touch", first.zip(second)),
        ("at_second_touch", second.map(|frame| (frame, frame))),
    ] {
        for (relation, other_team) in [("own_team", team), ("opponent", !team)] {
            values.float(
                &format!("{relation}_ball_distance_{name}"),
                range.and_then(|(start, end)| {
                    minimum_team_distance(
                        (start..=end)
                            .map(|frame| replay.team_distances[frame][usize::from(other_team)]),
                    )
                }),
            )?;
            if name == "candidate_to_first_touch" || name == "first_to_second_touch" {
                values.float(
                    &format!("{relation}_ball_distance_mean_{name}"),
                    mean_team_distance(context, &replay.team_distances, range, other_team),
                )?;
            }
        }
        for (goal, attacking_team) in [("own", !team), ("opponent", team)] {
            values.float(
                &format!("ball_{goal}_goal_distance_{name}"),
                minimum_goal_distance(context, range, attacking_team)?,
            )?;
        }
    }
    let extrema = second.and_then(|end| {
        let position = context
            .ball(Some(frame))?
            .location
            .project(team_basis(team));
        let mut minimum = position;
        let mut maximum = position;
        for frame in frame + 1..=end {
            let position = context
                .ball(Some(frame))?
                .location
                .project(team_basis(team));
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(position[axis]);
                maximum[axis] = maximum[axis].max(position[axis]);
            }
        }
        Some((minimum, maximum))
    });
    for (axis, name) in ["toward", "side", "up"].iter().enumerate() {
        values.float(
            &format!("exchange_ball_position_min_{name}"),
            extrema.map(|(min, _)| min[axis]),
        )?;
        values.float(
            &format!("exchange_ball_position_max_{name}"),
            extrema.map(|(_, max)| max[axis]),
        )?;
    }
    event_features(context, touch, samples, values)?;
    let duration = samples
        .second_touch
        .as_ref()
        .map(|stage| possession_durations(context, touch.number("time")?, stage.raw_time, team))
        .transpose()?
        .flatten();
    for (index, state) in ["own_team", "opponent", "neutral"].iter().enumerate() {
        values.float(
            &format!("exchange_possession_{state}_s"),
            duration.map(|duration| duration[index]),
        )?;
    }
    Ok(())
}

fn event_features(
    context: &Context<'_>,
    touch: &Event,
    samples: &Samples,
    values: &mut Values,
) -> Result<(), String> {
    let subject = touch.participant("player")?;
    let team = touch.boolean("is_team_0")?;
    let frame = touch.frame("frame")?;
    let end = samples.second_touch.as_ref();
    let mut subject_bumped = false;
    let mut any_bump = false;
    let mut subject_whiff = false;
    let mut teammate_whiff = false;
    if let Some(end) = end {
        for event in context.events("bump") {
            if (frame..=end.frame).contains(&event.frame("frame")?) {
                any_bump = true;
                subject_bumped |= event.participant("victim")? == subject;
            }
        }
        for event in context.events("whiff") {
            let event_frame = event.frame("frame")?;
            if event_frame > frame
                && event_frame <= end.frame
                && event.text("kind")? == "whiff"
                && event.number("resolved_time")? <= end.raw_time
            {
                let is_subject = event.participant("player")? == subject;
                subject_whiff |= is_subject;
                teammate_whiff |= !is_subject && event.boolean("is_team_0")? == team;
            }
        }
    }
    for (name, found) in [
        ("subject_bumped_during_exchange", subject_bumped),
        ("any_bump_during_exchange", any_bump),
        ("subject_whiff_during_exchange", subject_whiff),
        ("teammate_whiff_during_exchange", teammate_whiff),
    ] {
        values.boolean(name, end.map(|_| found));
    }
    Ok(())
}
