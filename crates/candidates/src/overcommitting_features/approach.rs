use super::{Evidence, Extractor, Samples, closing_speed, count};
use crate::feature_geometry::{directed_at_goal, goal_opening_distance};
use crate::feature_input::Body;
use crate::feature_measurements::{possession_durations, surrounding_players, team_basis};
use crate::feature_values::Values;
use serde_json::Value;

struct GoalSideTeammate<'a> {
    player_id: &'a Value,
    player_key: String,
    body: &'a Body,
    distance: f64,
}

impl Extractor<'_> {
    pub(super) fn challenge_features(
        &self,
        e: &Evidence,
        minimum: Option<(usize, f64)>,
        samples: &mut Samples,
        values: &mut Values,
    ) -> Result<(), String> {
        let c = &self.context;
        let frame = minimum.map(|(frame, _)| frame);
        let ball = c.ball(frame);
        let subject_data = c.player(&e.player)?;
        let subject = frame.and_then(|frame| subject_data.frames[frame].body());
        values.float(
            "approach_min_subject_ball_distance",
            minimum.map(|(_, distance)| distance),
        )?;
        let alignment = subject
            .zip(ball)
            .and_then(|(car, ball)| {
                ball.location
                    .subtract(car.location)
                    .unit()
                    .map(|to_ball| (car, to_ball))
            })
            .map(|(car, to_ball)| car.rotation.forward().map(|forward| forward.dot(to_ball)))
            .transpose()?;
        values.float("approach_subject_heading_alignment", alignment)?;
        values.float(
            "approach_subject_ball_closing_speed",
            subject
                .zip(ball)
                .and_then(|(car, ball)| closing_speed(car, ball)),
        )?;
        let surroundings = surrounding_players(c, &self.roster, &e.player, e.is_team_0, frame)?;
        let opponent = surroundings.closest_opponent;
        samples.approach_closest_opponent_id = opponent.as_ref().map(|car| car.player_id.clone());
        values.float(
            "approach_closest_opponent_ball_distance",
            opponent.as_ref().map(|car| car.distance),
        )?;
        values.float(
            "approach_closest_opponent_ball_closing_speed",
            opponent
                .as_ref()
                .zip(ball)
                .and_then(|(car, ball)| closing_speed(car.body, ball)),
        )?;
        for (name, frame, selected) in [
            (
                "approach",
                frame,
                &mut samples.approach_closest_goal_side_teammate_id,
            ),
            (
                "passage",
                Some(e.passage_frame),
                &mut samples.passage_closest_goal_side_teammate_id,
            ),
        ] {
            let (number, teammate) = self.goal_side_teammates(e, frame)?;
            count(values, &format!("{name}_goal_side_teammate_count"), number);
            *selected = teammate.as_ref().map(|car| car.player_id.clone());
            values.float(
                &format!("{name}_closest_goal_side_teammate_ball_distance"),
                teammate.as_ref().map(|car| car.distance),
            )?;
            values.float(
                &format!("{name}_closest_goal_side_teammate_ball_closing_speed"),
                teammate
                    .as_ref()
                    .zip(c.ball(frame))
                    .and_then(|(car, ball)| closing_speed(car.body, ball)),
            )?;
        }
        values.float(
            "last_opponent_touch_delay_from_approach_minimum_s",
            samples
                .last_opponent_touch
                .as_ref()
                .zip(frame)
                .map(|(touch, frame)| {
                    touch.raw_time
                        - c.analysis.outputs.replay_data.frame_data.metadata_frames[frame].time
                }),
        )?;
        values.boolean(
            "approach_ball_directed_at_own_goal",
            ball.map(|ball| directed_at_goal(ball, !e.is_team_0))
                .transpose()?
                .flatten(),
        );
        values.float(
            "approach_ball_own_goal_distance",
            ball.map(|ball| goal_opening_distance(ball.location, !e.is_team_0))
                .transpose()?,
        )?;
        values.float(
            "approach_opponent_possession_fraction",
            possession_durations(c, e.start_time, e.end_time, e.is_team_0)?
                .map(|duration| duration[1] / (e.end_time - e.start_time)),
        )?;
        Ok(())
    }

    fn goal_side_teammates(
        &self,
        e: &Evidence,
        frame: Option<usize>,
    ) -> Result<(Option<u32>, Option<GoalSideTeammate<'_>>), String> {
        let Some((frame, ball)) = frame.zip(self.context.ball(frame)) else {
            return Ok((None, None));
        };
        let mut count = 0;
        let mut nearest: Option<GoalSideTeammate<'_>> = None;
        let mut observed = true;
        for (id, data) in &self.context.analysis.outputs.replay_data.frame_data.players {
            let player_key = id.to_string();
            if id == &e.player || self.roster[&player_key] != e.is_team_0 {
                continue;
            }
            match self.activity[&player_key][frame] {
                Some("demolished") => continue,
                Some("tracked") => (),
                None => {
                    observed = false;
                    continue;
                }
                Some(state) => return Err(format!("invalid player activity state: {state}")),
            }
            let Some(car) = data.frames[frame].body() else {
                observed = false;
                continue;
            };
            let to_ball = ball.location.subtract(car.location);
            let distance = to_ball.length();
            if !distance.is_finite() {
                return Err("nonfinite goal-side teammate-ball distance".into());
            }
            if to_ball.dot(team_basis(e.is_team_0)) < 0.0 {
                continue;
            }
            count += 1;
            if nearest.as_ref().is_none_or(|previous| {
                distance < previous.distance
                    || (distance == previous.distance && player_key < previous.player_key)
            }) {
                nearest = Some(GoalSideTeammate {
                    player_id: id,
                    player_key,
                    body: car,
                    distance,
                });
            }
        }
        if observed {
            Ok((Some(count), nearest))
        } else {
            Ok((None, None))
        }
    }
}
