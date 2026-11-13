//! Native interval context for overcommitting candidates.

mod approach;
mod schema;

pub use crate::feature_contract::{FeatureSampling, SamplingMethod};
pub use crate::feature_input::ReplayAnalysis;
pub use crate::feature_values::Measurement;
pub use schema::{DataType, FeatureDefinition, definitions};

use crate::feature_context::{Context, tag};
use crate::feature_contract::{self, ContextFeatures};
use crate::feature_geometry::{goal_opening_distance, point_goal_opening_distance};
use crate::feature_input::{Body, Event, PlayerFrame, Vector};
use crate::feature_measurements::{
    self, TeamDistance, minimum_team_distance, surrounding_players, team_basis, time_weighted_mean,
};
use crate::feature_values::Values;
use crate::impact_sampling::{self, ParticipantSamples};
use crate::{Candidate, CandidateBatch, MistakeKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};

pub const SCHEMA: &str = "overcommitting.context.v1";
pub type FeatureBatch = feature_contract::FeatureBatch<FeatureRow>;

#[derive(Debug, Serialize)]
pub struct FeatureRow {
    #[serde(skip)]
    identity: String,
    pub frame: usize,
    pub end_frame: usize,
    pub passage_frame: usize,
    pub raw_time: f64,
    pub player_id: Value,
    pub source_event_ids: Vec<String>,
    pub samples: Samples,
    pub features: BTreeMap<String, Option<Measurement>>,
}

#[derive(Debug, Serialize)]
pub struct Samples {
    pub start_frame: usize,
    pub start_time: f64,
    pub end_frame: usize,
    pub end_time: f64,
    pub passage_frame: usize,
    pub passage_time: f64,
    pub play_start_frame: usize,
    pub play_end_frame: usize,
    pub closest_approach_frame: Option<usize>,
    pub closest_approach_time: Option<f64>,
    pub approach_closest_approach_frame: Option<usize>,
    pub approach_closest_approach_time: Option<f64>,
    pub approach_closest_opponent_id: Option<Value>,
    pub approach_closest_goal_side_teammate_id: Option<Value>,
    pub passage_closest_goal_side_teammate_id: Option<Value>,
    pub closest_opponent_id: Option<Value>,
    pub closest_teammate_id: Option<Value>,
    pub nearest_big_pad_index: Option<usize>,
    pub nearest_big_pad_id: Option<String>,
    pub last_opponent_touch: Option<TouchStage>,
    pub previous_subject_touch: Option<TouchStage>,
    pub previous_subject_touch_following_contact: Option<TouchStage>,
    pub previous_subject_touch_possession: Option<PossessionStage>,
    pub first_touch: Option<TouchStage>,
    pub second_touch: Option<TouchStage>,
    pub strongest_bump: Option<BumpStage>,
}

#[derive(Debug, Serialize)]
pub struct TouchStage {
    pub frame: usize,
    pub raw_time: f64,
    pub source_event_ids: Vec<String>,
    pub touch_ids: Vec<u64>,
}

#[derive(Debug, Serialize)]
pub struct PossessionStage {
    pub source_event_id: String,
    pub start_time: f64,
    pub end_time: f64,
    pub state: String,
    pub player_id: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct BumpStage {
    pub source_event_id: String,
    pub frame: usize,
    pub raw_time: f64,
    pub impact: ParticipantSamples,
}

#[derive(Deserialize)]
struct Evidence {
    start_frame: usize,
    start_time: f64,
    end_frame: usize,
    end_time: f64,
    passage_frame: usize,
    passage_time: f64,
    play_start_frame: usize,
    play_end_frame: usize,
    player: Value,
    is_team_0: bool,
}

impl ContextFeatures for FeatureRow {
    fn key(&self) -> (&str, &Value) {
        (&self.identity, &self.player_id)
    }
    fn features(&self) -> &BTreeMap<String, Option<Measurement>> {
        &self.features
    }
    fn validate_candidate(&self, candidate: &Candidate) -> Result<(), String> {
        let evidence: Evidence = serde_json::from_value(candidate.evidence.clone())
            .map_err(|error| error.to_string())?;
        if self.identity != feature_contract::context_key(candidate)?
            || self.frame as u64 != candidate.frame
            || self.raw_time != candidate.raw_time
            || self.frame != evidence.start_frame
            || self.end_frame != evidence.end_frame
            || self.passage_frame != evidence.passage_frame
            || self.player_id != candidate.player_id
            || self.source_event_ids != candidate.source_event_ids
            || self.samples.start_time != evidence.start_time
            || self.samples.end_time != evidence.end_time
            || self.samples.passage_time != evidence.passage_time
            || self.samples.play_start_frame != evidence.play_start_frame
            || self.samples.play_end_frame != evidence.play_end_frame
        {
            return Err(
                "overcommitting context does not match its candidate interval and sources".into(),
            );
        }
        Ok(())
    }
}

pub fn extract(
    analysis: &ReplayAnalysis,
    sampling: FeatureSampling,
    subject: Option<&Value>,
) -> Result<(CandidateBatch, FeatureBatch), String> {
    let mut candidates = MistakeKind::Overcommitting.candidates_from_analysis(analysis)?;
    if let Some(subject) = subject {
        candidates
            .candidates
            .retain(|candidate| &candidate.player_id == subject);
    }
    let features = extract_for_candidates(analysis, sampling, &candidates)?;
    Ok((candidates, features))
}

pub fn extract_for_candidates(
    analysis: &ReplayAnalysis,
    sampling: FeatureSampling,
    candidates: &CandidateBatch,
) -> Result<FeatureBatch, String> {
    sampling.validate(SamplingMethod::NativeIntervalBoundaries)?;
    candidates.validate()?;
    if candidates.kind != MistakeKind::Overcommitting.name()
        || candidates.replay_sha256 != analysis.replay_sha256
    {
        return Err("overcommitting context requires matching native interval candidates".into());
    }
    let extractor = Extractor::new(analysis)?;
    let feature_definitions = definitions();
    let rows = candidates
        .candidates
        .iter()
        .map(|candidate| extractor.row(candidate, &feature_definitions))
        .collect::<Result<_, _>>()?;
    Ok(FeatureBatch {
        input_schema: SCHEMA,
        engine_revision: analysis.revision.clone(),
        replay_sha256: analysis.replay_sha256.clone(),
        sampling,
        feature_definitions,
        rows,
    })
}

struct Extractor<'a> {
    context: Context<'a>,
    roster: HashMap<String, bool>,
    activity: HashMap<String, Vec<Option<&'a str>>>,
    touches: BTreeMap<usize, Vec<&'a Event>>,
    team_distances: Vec<[TeamDistance; 2]>,
    event_ids: HashSet<&'a str>,
}

impl<'a> Extractor<'a> {
    fn new(analysis: &'a ReplayAnalysis) -> Result<Self, String> {
        let context = Context::new(
            analysis,
            &[
                "touch",
                "possession",
                "boost_pickups",
                "timeline",
                "bump",
                "demolition",
                "whiff",
                "beaten_to_ball",
                "goal_context",
                "player_activity",
            ],
        )?;
        let native_roster = analysis
            .outputs
            .events
            .replay_meta
            .as_ref()
            .ok_or("missing native roster")?;
        let mut roster = HashMap::new();
        for (players, team) in [
            (&native_roster.team_zero, true),
            (&native_roster.team_one, false),
        ] {
            for player in players {
                context.player(&player.remote_id)?;
                if roster.insert(player.remote_id.to_string(), team).is_some() {
                    return Err("duplicate native roster player".into());
                }
            }
        }
        let mut activity = HashMap::new();
        for (id, _) in &analysis.outputs.replay_data.frame_data.players {
            if !roster.contains_key(&id.to_string()) {
                return Err("native player is absent from roster".into());
            }
            activity.insert(
                id.to_string(),
                context.states_by_frame("player_activity", id)?,
            );
        }
        let frames = &analysis.outputs.replay_data.frame_data.metadata_frames;
        let mut touches: BTreeMap<usize, Vec<&Event>> = BTreeMap::new();
        for event in context.events("touch") {
            let frame = event.frame("frame")?;
            let time = event.number("time")?;
            if frames
                .get(frame)
                .is_none_or(|metadata| metadata.time != time)
            {
                return Err("touch disagrees with native frame time".into());
            }
            if roster.get(&event.participant("player")?.to_string())
                != Some(&event.boolean("is_team_0")?)
            {
                return Err("touch team disagrees with native roster".into());
            }
            touches.entry(frame).or_default().push(event);
        }
        for group in touches.values_mut() {
            group.sort_by_key(|event| event.meta.id.as_str());
        }
        let team_distances = feature_measurements::team_distances(&context, &roster)?;
        let event_ids = analysis
            .outputs
            .events
            .events
            .events
            .iter()
            .map(|event| event.meta.id.as_str())
            .collect();
        Ok(Self {
            context,
            roster,
            activity,
            touches,
            team_distances,
            event_ids,
        })
    }

    fn row(
        &self,
        candidate: &Candidate,
        definitions: &[FeatureDefinition],
    ) -> Result<FeatureRow, String> {
        candidate.anchor()?;
        let e: Evidence = serde_json::from_value(candidate.evidence.clone())
            .map_err(|error| format!("invalid overcommitting context evidence: {error}"))?;
        let c = &self.context;
        let native = &c.analysis.outputs.replay_data.frame_data;
        let metadata = &native.metadata_frames;
        if e.start_frame as u64 != candidate.frame
            || e.start_time != candidate.raw_time
            || e.player != candidate.player_id
            || self.roster.get(&e.player.to_string()) != Some(&e.is_team_0)
            || e.play_start_frame > e.start_frame
            || e.start_frame >= e.end_frame
            || e.end_frame > e.passage_frame
            || e.passage_frame > e.play_end_frame
            || e.play_end_frame >= metadata.len()
        {
            return Err("invalid native overcommitting interval boundaries".into());
        }
        for (frame, time) in [
            (e.start_frame, e.start_time),
            (e.end_frame, e.end_time),
            (e.passage_frame, e.passage_time),
        ] {
            if metadata[frame].time != time {
                return Err("overcommitting boundary disagrees with native frame time".into());
            }
        }
        if candidate
            .source_event_ids
            .iter()
            .any(|id| !self.event_ids.contains(id.as_str()))
        {
            return Err("overcommitting source event is absent from analysis".into());
        }
        for state in &self.activity[&e.player.to_string()][e.start_frame..=e.passage_frame] {
            if *state != Some("tracked") {
                return Err(
                    "overcommitting interval crosses demolished or unknown subject activity".into(),
                );
            }
        }
        for event in c.events("demolition") {
            if event.participant("victim")? == &e.player && in_range(event.frame("frame")?, &e) {
                return Err("overcommitting interval crosses subject demolition".into());
            }
        }
        let opponent_groups = self
            .touches
            .range(e.start_frame..=e.passage_frame)
            .map(|(_, events)| {
                events
                    .iter()
                    .copied()
                    .filter_map(|event| match event.boolean("is_team_0") {
                        Ok(team) if team != e.is_team_0 => Some(Ok(event)),
                        Ok(_) => None,
                        Err(error) => Some(Err(error)),
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let last_opponent = opponent_groups
            .iter()
            .rev()
            .find(|group| !group.is_empty())
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let previous = self
            .touches
            .range(e.play_start_frame..e.start_frame)
            .rev()
            .find_map(|(_, events)| {
                let subject = events
                    .iter()
                    .copied()
                    .filter(|event| event.participant("player") == Ok(&e.player))
                    .collect::<Vec<_>>();
                (!subject.is_empty()).then_some(subject)
            })
            .unwrap_or_default();
        let later = self
            .touches
            .range((
                std::ops::Bound::Excluded(e.passage_frame),
                std::ops::Bound::Included(e.play_end_frame),
            ))
            .map(|(_, events)| events.as_slice())
            .take(2)
            .collect::<Vec<_>>();
        let first = later.first().copied().unwrap_or(&[]);
        let second = later.get(1).copied().unwrap_or(&[]);
        let minimum = self.closest_approach(&e, e.passage_frame)?;
        let approach_minimum = self.closest_approach(&e, e.end_frame)?;
        let mut samples = Samples {
            start_frame: e.start_frame,
            start_time: e.start_time,
            end_frame: e.end_frame,
            end_time: e.end_time,
            passage_frame: e.passage_frame,
            passage_time: e.passage_time,
            play_start_frame: e.play_start_frame,
            play_end_frame: e.play_end_frame,
            closest_approach_frame: minimum.map(|(frame, _)| frame),
            closest_approach_time: minimum.map(|(frame, _)| metadata[frame].time),
            approach_closest_approach_frame: approach_minimum.map(|(frame, _)| frame),
            approach_closest_approach_time: approach_minimum.map(|(frame, _)| metadata[frame].time),
            approach_closest_opponent_id: None,
            approach_closest_goal_side_teammate_id: None,
            passage_closest_goal_side_teammate_id: None,
            closest_opponent_id: None,
            closest_teammate_id: None,
            nearest_big_pad_index: None,
            nearest_big_pad_id: None,
            last_opponent_touch: stage(last_opponent)?,
            previous_subject_touch: stage(&previous)?,
            previous_subject_touch_following_contact: None,
            previous_subject_touch_possession: None,
            first_touch: stage(first)?,
            second_touch: stage(second)?,
            strongest_bump: None,
        };
        let mut values = Values::default();
        self.approach_features(&e, &mut values)?;
        self.boost_target_features(&e, &mut samples, &mut values)?;
        self.opening_features(&e, &mut samples, &mut values)?;
        self.previous_touch_features(&e, &previous, &mut samples, &mut values)?;
        self.failure_features(&e, last_opponent, &opponent_groups, minimum, &mut values)?;
        self.bump_features(&e, minimum, &mut samples, &mut values)?;
        followup_features(c, &e, first, second, &previous, &mut values)?;
        self.shot_features(&e, &mut values)?;
        self.challenge_features(&e, approach_minimum, &mut samples, &mut values)?;
        values.validate(definitions)?;
        let row = FeatureRow {
            identity: feature_contract::context_key(candidate)?,
            frame: e.start_frame,
            end_frame: e.end_frame,
            passage_frame: e.passage_frame,
            raw_time: e.start_time,
            player_id: e.player,
            source_event_ids: candidate.source_event_ids.clone(),
            samples,
            features: values.0,
        };
        row.validate_candidate(candidate)?;
        Ok(row)
    }

    fn previous_touch_features(
        &self,
        e: &Evidence,
        previous: &[&Event],
        samples: &mut Samples,
        values: &mut Values,
    ) -> Result<(), String> {
        let Some(touch) = previous.first() else {
            values.boolean("previous_subject_touch_possession_before_self", None);
            values.boolean("previous_subject_touch_followed_by_opponent", None);
            return Ok(());
        };
        let time = touch.number("time")?;
        let frame = touch.frame("frame")?;
        for event in self.context.events("possession") {
            let start = event.number("time")?;
            let end = event.number("end_time")?;
            if end < start {
                return Err("reversed native possession span".into());
            }
        }
        let possession = self.context.state_event("possession", None, time)?;
        let owned = if let Some(event) = possession {
            let state = event.text("possession_state")?;
            let owner = event
                .payload
                .payload
                .get("player_id")
                .filter(|owner| !owner.is_null());
            let team = match state {
                "neutral" => None,
                "team_zero" => Some(true),
                "team_one" => Some(false),
                _ => return Err(format!("unsupported native possession state: {state}")),
            };
            if let Some(owner) = owner {
                let owner_team = self
                    .roster
                    .get(&owner.to_string())
                    .ok_or("native possession owner is absent from roster")?;
                if team != Some(*owner_team) {
                    return Err("native possession state disagrees with owner team".into());
                }
            }
            samples.previous_subject_touch_possession = Some(PossessionStage {
                source_event_id: event.meta.id.clone(),
                start_time: event.number("time")?,
                end_time: event.number("end_time")?,
                state: state.into(),
                player_id: owner.cloned(),
            });
            if team == Some(e.is_team_0) {
                owner.map(|owner| owner == &e.player)
            } else {
                Some(false)
            }
        } else {
            None
        };
        values.boolean("previous_subject_touch_possession_before_self", owned);
        let following = self
            .touches
            .range((
                std::ops::Bound::Excluded(frame),
                std::ops::Bound::Included(e.play_end_frame),
            ))
            .next()
            .map(|(_, events)| events.as_slice());
        samples.previous_subject_touch_following_contact =
            following.map(stage).transpose()?.flatten();
        let opponent = following
            .map(|events| {
                let mut found = false;
                for event in events {
                    found |= event.boolean("is_team_0")? != e.is_team_0;
                }
                Ok::<_, String>(found)
            })
            .transpose()?;
        values.boolean("previous_subject_touch_followed_by_opponent", opponent);
        Ok(())
    }

    fn ball_withdrawal_distance(
        &self,
        e: &Evidence,
        start: usize,
        end: usize,
    ) -> Result<Option<f64>, String> {
        let player = self.context.player(&e.player)?;
        let metadata = &self
            .context
            .analysis
            .outputs
            .replay_data
            .frame_data
            .metadata_frames;
        let mut distance = 0.0;
        for frame in start..end {
            let Some(car) = player.frames[frame].body() else {
                return Ok(None);
            };
            let Some(ball) = self.context.ball(Some(frame)) else {
                return Ok(None);
            };
            let Some(velocity) = car.linear_velocity else {
                return Ok(None);
            };
            let Some(toward) = ball.location.subtract(car.location).horizontal_unit() else {
                return Ok(None);
            };
            let closing_speed = velocity.dot(toward);
            if !closing_speed.is_finite() {
                return Err("nonfinite horizontal ball-withdrawal speed".into());
            }
            distance +=
                (-closing_speed).max(0.0) * (metadata[frame + 1].time - metadata[frame].time);
        }
        Ok(Some(distance))
    }

    fn closest_approach(
        &self,
        e: &Evidence,
        end_frame: usize,
    ) -> Result<Option<(usize, f64)>, String> {
        let subject = self.context.player(&e.player)?;
        let mut closest: Option<(usize, f64)> = None;
        for frame in e.start_frame..=end_frame {
            let Some((car, ball)) = subject.frames[frame]
                .body()
                .zip(self.context.ball(Some(frame)))
            else {
                return Ok(None);
            };
            let distance = ball.location.subtract(car.location).length();
            if !distance.is_finite() {
                return Err("nonfinite closest approach distance".into());
            }
            if closest.is_none_or(|(_, minimum)| distance < minimum) {
                closest = Some((frame, distance));
            }
        }
        Ok(closest)
    }

    fn approach_features(&self, e: &Evidence, values: &mut Values) -> Result<(), String> {
        let c = &self.context;
        let player = c.player(&e.player)?;
        let metadata = &c.analysis.outputs.replay_data.frame_data.metadata_frames;
        let basis = team_basis(e.is_team_0);
        values.float("approach_duration_s", Some(e.end_time - e.start_time))?;
        values.float(
            "passage_delay_after_approach_s",
            Some(e.passage_time - e.end_time),
        )?;
        let bodies = (e.start_frame..=e.end_frame)
            .map(|frame| player.frames[frame].body())
            .collect::<Option<Vec<_>>>();
        let mut distance = [None; 3];
        let mut orientation = None;
        if let Some(bodies) = &bodies {
            let mut totals = [0.0_f64; 3];
            let mut angle = 0.0;
            for body in bodies {
                totals[0] = totals[0].max(body.location.subtract(bodies[0].location).dot(basis));
            }
            for pair in bodies.windows(2) {
                let delta = pair[1].location.subtract(pair[0].location);
                let forward = delta.dot(basis);
                totals[1] += (-forward).max(0.0);
                totals[2] += delta.length();
                angle += pair[0].rotation.angle_change(&pair[1].rotation)?;
            }
            distance = totals.map(Some);
            orientation = Some(angle);
        }
        for (name, value) in [
            "subject_max_advance",
            "subject_retreat_distance",
            "subject_path_distance",
        ]
        .into_iter()
        .zip(distance)
        {
            values.float(name, value)?;
        }
        values.float("subject_orientation_change", orientation)?;
        values.float(
            "subject_mean_speed",
            time_weighted_mean(metadata, Some((e.start_frame, e.end_frame)), |frame| {
                player.frames[frame]
                    .body()?
                    .linear_velocity
                    .map(Vector::length)
            }),
        )?;
        values.float(
            "subject_max_speed",
            maximum(e.start_frame, e.end_frame, |frame| {
                player.frames[frame]
                    .body()?
                    .linear_velocity
                    .map(Vector::length)
            }),
        )?;
        let used = c
            .analysis
            .outputs
            .events
            .accumulation_tracks
            .iter()
            .find(|track| track.player_id == e.player && track.quantity == "boost_used");
        let boost_used = used.and_then(|track| {
            let before = track
                .points
                .iter()
                .rev()
                .find(|point| point.frame <= e.start_frame)?;
            let after = track
                .points
                .iter()
                .rev()
                .find(|point| point.frame <= e.end_frame)?;
            Some(after.value - before.value)
        });
        if boost_used.is_some_and(|used| used < 0.0) {
            return Err("cumulative boost-used track decreased during approach".into());
        }
        values.float("subject_boost_used", boost_used)?;
        for (name, flag) in [("subject_jump_active_s", 1), ("subject_dodge_active_s", 2)] {
            let duration = (e.start_frame..e.end_frame).try_fold(0.0, |total, frame| {
                let active = match &player.frames[frame] {
                    PlayerFrame::Empty => None,
                    PlayerFrame::Data {
                        jump_active,
                        dodge_active,
                        ..
                    } => match flag {
                        1 => *jump_active,
                        _ => Some(*dodge_active),
                    },
                }?;
                Some(
                    total
                        + if active {
                            metadata[frame + 1].time - metadata[frame].time
                        } else {
                            0.0
                        },
                )
            });
            values.float(name, duration)?;
        }
        Ok(())
    }

    fn boost_target_features(
        &self,
        e: &Evidence,
        samples: &mut Samples,
        values: &mut Values,
    ) -> Result<(), String> {
        let context = &self.context;
        let metadata = &context
            .analysis
            .outputs
            .replay_data
            .frame_data
            .metadata_frames;
        let mut pickups = Some(0_u32);
        for event in context.events("boost_pickups") {
            if event.participant("player_id")? != &e.player {
                continue;
            }
            let frame = event.frame("frame")?;
            if frame < e.start_frame || frame > e.end_frame {
                continue;
            }
            if event.number("time")? != metadata[frame].time {
                return Err("native boost pickup time disagrees with its frame".into());
            }
            match event.text("pad_type")? {
                "big" => pickups = pickups.map(|count| count + 1),
                "small" => (),
                "ambiguous" => pickups = None,
                pad_type => return Err(format!("unsupported native boost pad type: {pad_type}")),
            }
        }
        count(values, "subject_big_boost_pickup_count", pickups);
        let pads = &context.analysis.outputs.replay_data.boost_pads;
        let mut indices = HashSet::new();
        let mut targets = Vec::new();
        let mut complete = true;
        for pad in pads {
            match pad.size.as_deref() {
                Some("Big") => match pad.index {
                    Some(index) => {
                        if !indices.insert(index) {
                            return Err("duplicate native Big boost-pad index".into());
                        }
                        targets.push((index, pad));
                    }
                    None => complete = false,
                },
                Some("Small") => (),
                None => complete = false,
                Some(size) => return Err(format!("unsupported native boost-pad size: {size}")),
            }
        }
        let player = context.player(&e.player)?;
        let mut nearest = None;
        if complete && let Some(car) = player.frames[e.start_frame].body() {
            for (index, pad) in targets {
                let distance = car.location.horizontal_distance(pad.position);
                if !distance.is_finite() {
                    return Err("nonfinite native Big boost-pad distance".into());
                }
                if nearest
                    .as_ref()
                    .is_none_or(|(previous_distance, previous_index, _)| {
                        (distance, index) < (*previous_distance, *previous_index)
                    })
                {
                    nearest = Some((distance, index, pad));
                }
            }
        }
        samples.nearest_big_pad_index = nearest.map(|(_, index, _)| index);
        samples.nearest_big_pad_id = nearest.and_then(|(_, _, pad)| pad.pad_id.clone());
        let mean = nearest.and_then(|(_, _, pad)| {
            time_weighted_mean(metadata, Some((e.start_frame, e.end_frame)), |frame| {
                let car = player.frames[frame].body()?;
                let velocity = car.linear_velocity?;
                let ball = context.ball(Some(frame))?;
                let to_pad = pad.position.subtract(car.location).horizontal_unit()?;
                let to_ball = ball.location.subtract(car.location).horizontal_unit()?;
                Some(velocity.dot(to_pad) - velocity.dot(to_ball))
            })
        });
        values.float("subject_nearest_big_pad_closing_advantage_mean", mean)
    }

    fn opening_features(
        &self,
        e: &Evidence,
        samples: &mut Samples,
        values: &mut Values,
    ) -> Result<(), String> {
        let c = &self.context;
        let basis = Some(team_basis(e.is_team_0));
        let subject = c.player(&e.player)?.frames[e.start_frame].body();
        let ball = c.ball(Some(e.start_frame));
        values.vector(
            "candidate_subject_velocity",
            subject.and_then(|body| body.linear_velocity),
            basis,
        )?;
        values.vector(
            "candidate_subject_heading",
            subject.map(|body| body.rotation.forward()).transpose()?,
            basis,
        )?;
        values.vector(
            "candidate_subject_position_relative_ball",
            subject
                .zip(ball)
                .map(|(car, ball)| car.location.subtract(ball.location)),
            basis,
        )?;
        values.vector(
            "candidate_ball_position",
            ball.map(|body| body.location),
            basis,
        )?;
        values.vector(
            "candidate_ball_velocity",
            ball.and_then(|body| body.linear_velocity),
            basis,
        )?;
        let metadata = &c.analysis.outputs.replay_data.frame_data.metadata_frames;
        let before = c.previous_frame(e.start_frame, metadata[e.play_start_frame].time);
        values.float("subject_boost_before_pct", self.boost(&e.player, before)?)?;
        values.float(
            "candidate_ball_own_goal_distance",
            ball.map(|body| goal_opening_distance(body.location, !e.is_team_0))
                .transpose()?,
        )?;
        let surroundings =
            surrounding_players(c, &self.roster, &e.player, e.is_team_0, Some(e.start_frame))?;
        count(
            values,
            "approaching_teammate_count",
            surroundings.approaching[0],
        );
        count(
            values,
            "approaching_opponent_count",
            surroundings.approaching[1],
        );
        let opponent = surroundings.closest_opponent;
        samples.closest_opponent_id = opponent.as_ref().map(|sample| sample.player_id.clone());
        values.float(
            "closest_opponent_ball_distance",
            opponent.as_ref().map(|sample| sample.distance),
        )?;
        values.float(
            "closest_opponent_closing_speed",
            opponent
                .as_ref()
                .zip(ball)
                .and_then(|(sample, ball)| closing_speed(sample.body, ball)),
        )?;
        values.float(
            "closest_opponent_boost_before_pct",
            opponent
                .as_ref()
                .map(|sample| self.boost(sample.player_id, before))
                .transpose()?
                .flatten(),
        )?;
        let mut nearest: Option<(String, &Value, &Body, f64)> = None;
        let mut teammate_unknown = false;
        let mut goal_side_count = ball.map(|_| 0_u32);
        let mut depth: Option<f64> = None;
        let mut goal_distance: Option<f64> = None;
        for (id, data) in &c.analysis.outputs.replay_data.frame_data.players {
            if self.roster[&id.to_string()] != e.is_team_0 {
                continue;
            }
            let state = self.activity[&id.to_string()][e.start_frame];
            if state == Some("demolished") {
                continue;
            }
            if state != Some("tracked") || data.frames[e.start_frame].body().is_none() {
                if id != &e.player {
                    teammate_unknown = true;
                    goal_side_count = None;
                }
                continue;
            }
            let car = data.frames[e.start_frame].body().unwrap();
            if id == &e.player {
                continue;
            }
            let opening_distance = point_goal_opening_distance(car.location, !e.is_team_0)?;
            goal_distance =
                Some(goal_distance.map_or(opening_distance, |old| old.min(opening_distance)));
            let y = car.location.dot(team_basis(e.is_team_0));
            depth = Some(depth.map_or(y, |old| old.min(y)));
            if let Some(ball) = ball {
                let distance = ball.location.subtract(car.location).length();
                if !distance.is_finite() {
                    return Err("nonfinite opening teammate distance".into());
                }
                let key = id.to_string();
                if nearest
                    .as_ref()
                    .is_none_or(|(old_key, _, _, old_distance)| {
                        distance < *old_distance || (distance == *old_distance && key < *old_key)
                    })
                {
                    nearest = Some((key, id, car, distance));
                }
                goal_side_count = goal_side_count.map(|count| {
                    count
                        + u32::from(
                            ball.location
                                .subtract(car.location)
                                .dot(team_basis(e.is_team_0))
                                >= 0.0,
                        )
                });
            }
        }
        if teammate_unknown {
            nearest = None;
            depth = None;
            goal_distance = None;
        }
        samples.closest_teammate_id = nearest.as_ref().map(|(_, id, _, _)| (*id).clone());
        values.float(
            "closest_teammate_ball_distance",
            nearest.as_ref().map(|(_, _, _, distance)| *distance),
        )?;
        values.float(
            "closest_teammate_closing_speed",
            nearest
                .as_ref()
                .zip(ball)
                .and_then(|((_, _, car, _), ball)| closing_speed(car, ball)),
        )?;
        values.float(
            "closest_teammate_boost_before_pct",
            nearest
                .as_ref()
                .map(|(_, id, _, _)| self.boost(id, before))
                .transpose()?
                .flatten(),
        )?;
        values.float(
            "subject_depth_lead",
            depth
                .zip(subject)
                .map(|(depth, car)| depth - car.location.dot(team_basis(e.is_team_0))),
        )?;
        values.float("teammate_goal_distance_min", goal_distance)?;
        count(values, "goal_side_teammate_count", goal_side_count);
        Ok(())
    }

    fn boost(&self, player: &Value, frame: Option<usize>) -> Result<Option<f64>, String> {
        if !self
            .context
            .analysis
            .outputs
            .events
            .accumulation_tracks
            .iter()
            .any(|track| &track.player_id == player && track.quantity == "boost_amount")
        {
            return Ok(None);
        }
        self.context.boost_before(player, frame)
    }

    fn failure_features(
        &self,
        e: &Evidence,
        last: &[&Event],
        opponent_groups: &[Vec<&Event>],
        minimum: Option<(usize, f64)>,
        values: &mut Values,
    ) -> Result<(), String> {
        let c = &self.context;
        let metadata = &c.analysis.outputs.replay_data.frame_data.metadata_frames;
        let subject = c.player(&e.player)?;
        let contact = last.first().map(|event| event.frame("frame")).transpose()?;
        for touch in last {
            c.validate_touch_sample(touch)?;
        }
        let before_withdrawal = contact
            .map(|frame| self.ball_withdrawal_distance(e, e.start_frame, frame.min(e.end_frame)))
            .transpose()?
            .flatten();
        let after_withdrawal = contact
            .map(|frame| self.ball_withdrawal_distance(e, frame.max(e.start_frame), e.end_frame))
            .transpose()?
            .flatten();
        values.float(
            "subject_ball_withdrawal_distance_before_opponent_touch",
            before_withdrawal,
        )?;
        values.float(
            "subject_ball_withdrawal_distance_after_opponent_touch",
            after_withdrawal,
        )?;
        let before =
            contact.and_then(|frame| c.previous_frame(frame, metadata[e.play_start_frame].time));
        let incoming = c.ball(before).and_then(|body| body.linear_velocity);
        let outgoing = c.ball(contact).and_then(|body| body.linear_velocity);
        values.vector(
            "opponent_touch_ball_velocity_delta",
            outgoing
                .zip(incoming)
                .map(|(after, before)| after.subtract(before)),
            Some(team_basis(e.is_team_0)),
        )?;
        values.float(
            "opponent_touch_incoming_ball_speed",
            incoming.map(Vector::length),
        )?;
        values.float(
            "opponent_touch_outgoing_ball_speed",
            outgoing.map(Vector::length),
        )?;
        values.float(
            "subject_ball_distance_at_opponent_touch",
            contact.and_then(|frame| {
                subject.frames[frame]
                    .body()
                    .zip(c.ball(Some(frame)))
                    .map(|(car, ball)| ball.location.subtract(car.location).length())
            }),
        )?;
        values.float(
            "min_subject_ball_distance",
            minimum.map(|(_, distance)| distance),
        )?;
        values.float(
            "closest_approach_delay_s",
            minimum.map(|(frame, _)| metadata[frame].time - e.start_time),
        )?;
        let closest_car = minimum.and_then(|(frame, _)| subject.frames[frame].body());
        values.float(
            "subject_height_at_closest_approach",
            closest_car.map(|body| body.location.z),
        )?;
        let alignment = match minimum.and_then(|(frame, _)| closest_car.zip(c.ball(Some(frame)))) {
            Some((car, ball)) => ball
                .location
                .subtract(car.location)
                .unit()
                .map(|toward| car.rotation.forward().map(|heading| heading.dot(toward)))
                .transpose()?,
            None => None,
        };
        values.float("subject_heading_alignment_at_closest_approach", alignment)?;
        let opponent_team = usize::from(!e.is_team_0);
        values.float(
            "opponent_ball_distance_min",
            minimum_team_distance(
                self.team_distances[e.start_frame..=e.passage_frame]
                    .iter()
                    .map(|distances| distances[opponent_team]),
            ),
        )?;
        values.float(
            "opponent_ball_distance_mean",
            feature_measurements::mean_team_distance(
                c,
                &self.team_distances,
                Some((e.start_frame, e.passage_frame)),
                !e.is_team_0,
            ),
        )?;
        let other = self.other_teammate_distances(e)?;
        values.float(
            "other_teammate_ball_distance_min",
            minimum_team_distance(other.iter().copied()),
        )?;
        values.float(
            "other_teammate_ball_distance_mean",
            time_weighted_mean(metadata, Some((e.start_frame, e.passage_frame)), |frame| {
                let distance = other[frame - e.start_frame];
                distance.nearest.filter(|_| distance.observed)
            }),
        )?;
        values.float(
            "peak_ball_height",
            maximum(e.start_frame, e.end_frame, |frame| {
                c.ball(Some(frame)).map(|body| body.location.z)
            }),
        )?;
        values.float(
            "minimum_own_goal_ball_distance",
            feature_measurements::minimum_goal_distance(
                c,
                contact.map(|frame| (frame, e.passage_frame)),
                !e.is_team_0,
            )?,
        )?;
        values.float(
            "maximum_goalward_ball_speed",
            contact.and_then(|start| {
                maximum(start, e.passage_frame, |frame| {
                    c.ball(Some(frame))?
                        .linear_velocity
                        .map(|velocity| -velocity.dot(team_basis(e.is_team_0)))
                })
            }),
        )?;
        let mut ids = HashSet::new();
        for group in opponent_groups {
            for event in group {
                ids.insert(touch_id(event)?);
            }
        }
        count_usize(values, "opponent_touch_count", ids.len())?;
        for (stream, name) in [
            ("whiff", "subject_whiff_count"),
            ("beaten_to_ball", "subject_beaten_count"),
        ] {
            let mut count = 0;
            for event in c.events(stream) {
                if event.participant("player")? != &e.player || !in_range(event.frame("frame")?, e)
                {
                    continue;
                }
                if stream == "whiff"
                    && (event.text("kind")? != "whiff"
                        || event.number("resolved_time")? > e.passage_time)
                {
                    continue;
                }
                count += 1;
            }
            count_usize(values, name, count)?;
        }
        Ok(())
    }

    fn shot_features(&self, e: &Evidence, values: &mut Values) -> Result<(), String> {
        let c = &self.context;
        let mut own_touch_frame = None;
        for (&frame, events) in self.touches.range((
            std::ops::Bound::Excluded(e.start_frame),
            std::ops::Bound::Included(e.play_end_frame),
        )) {
            let mut own_touch = false;
            for event in events {
                own_touch |= event.boolean("is_team_0")? == e.is_team_0;
            }
            if own_touch {
                own_touch_frame = Some(frame);
                break;
            }
        }
        let mut found = false;
        for stream in ["touch", "timeline"] {
            for event in c.events(stream) {
                let shot = if stream == "touch" {
                    tag(event, "action")? == Some("shot")
                } else {
                    event.text("kind")? == "Shot"
                };
                if !shot {
                    continue;
                }
                let frame = event.frame("frame")?;
                if frame >= e.start_frame
                    && frame <= e.play_end_frame
                    && own_touch_frame.is_none_or(|end| frame < end)
                    && event.boolean("is_team_0")? != e.is_team_0
                {
                    let time = event.number("time")?;
                    if c.analysis
                        .outputs
                        .replay_data
                        .frame_data
                        .metadata_frames
                        .get(frame)
                        .is_none_or(|metadata| metadata.time != time)
                    {
                        return Err("native shot time disagrees with its frame".into());
                    }
                    found = true;
                }
            }
        }
        values.boolean("opponent_shot_before_own_touch", Some(found));
        Ok(())
    }

    fn other_teammate_distances(&self, e: &Evidence) -> Result<Vec<TeamDistance>, String> {
        let c = &self.context;
        let native = &c.analysis.outputs.replay_data.frame_data;
        (e.start_frame..=e.passage_frame)
            .map(|frame| {
                let Some(ball) = c.ball(Some(frame)) else {
                    return Ok(TeamDistance {
                        observed: false,
                        nearest: None,
                    });
                };
                let mut result = TeamDistance {
                    observed: true,
                    nearest: None,
                };
                for (id, data) in &native.players {
                    if id == &e.player || self.roster[&id.to_string()] != e.is_team_0 {
                        continue;
                    }
                    match self.activity[&id.to_string()][frame] {
                        Some("demolished") => continue,
                        Some("tracked") => (),
                        None => {
                            result.observed = false;
                            continue;
                        }
                        Some(state) => {
                            return Err(format!("invalid player activity state: {state}"));
                        }
                    }
                    let Some(car) = data.frames[frame].body() else {
                        result.observed = false;
                        continue;
                    };
                    let distance = ball.location.subtract(car.location).length();
                    if !distance.is_finite() {
                        return Err("nonfinite teammate-ball distance".into());
                    }
                    result.nearest = Some(
                        result
                            .nearest
                            .map_or(distance, |previous| previous.min(distance)),
                    );
                }
                Ok(result)
            })
            .collect()
    }

    fn bump_features(
        &self,
        e: &Evidence,
        minimum: Option<(usize, f64)>,
        samples: &mut Samples,
        values: &mut Values,
    ) -> Result<(), String> {
        let c = &self.context;
        let metadata = &c.analysis.outputs.replay_data.frame_data.metadata_frames;
        let mut counts = [0_usize; 2];
        let mut strongest: Option<(&Event, f64)> = None;
        for event in c.events("bump") {
            if event.participant("victim")? != &e.player || !in_range(event.frame("frame")?, e) {
                continue;
            }
            counts[usize::from(!event.boolean("is_team_bump")?)] += 1;
            let impulse = event.number("victim_impulse")?;
            let replace = match strongest {
                None => true,
                Some((previous, old)) => {
                    impulse > old
                        || (impulse == old
                            && (event.frame("frame")?, &event.meta.id)
                                < (previous.frame("frame")?, &previous.meta.id))
                }
            };
            if replace {
                strongest = Some((event, impulse));
            }
        }
        count_usize(values, "subject_team_bump_count", counts[0])?;
        count_usize(values, "subject_opponent_bump_count", counts[1])?;
        values.float(
            "subject_bump_max_impulse",
            strongest.map(|(_, impulse)| impulse),
        )?;
        let mut delta = None;
        let mut upright = None;
        let mut timing = None;
        if let Some((bump, _)) = strongest {
            let player = c.player(&e.player)?;
            let frame = bump.frame("frame")?;
            let impact = impact_sampling::measure(
                player,
                metadata,
                frame,
                metadata[e.play_start_frame].time,
                metadata[e.play_end_frame].time,
            );
            let before = impact
                .before_frame
                .and_then(|frame| player.frames[frame].body())
                .and_then(|body| body.linear_velocity);
            let after = impact
                .after_frame
                .and_then(|frame| player.frames[frame].body());
            delta = after
                .and_then(|body| body.linear_velocity)
                .zip(before)
                .map(|(after, before)| after.subtract(before));
            upright = after.map(|body| body.rotation.upright()).transpose()?;
            let bump_time = bump.number("time")?;
            timing = minimum.map(|(closest, _)| metadata[closest].time - bump_time);
            samples.strongest_bump = Some(BumpStage {
                source_event_id: bump.meta.id.clone(),
                frame,
                raw_time: bump.number("time")?,
                impact,
            });
        }
        values.vector(
            "subject_bump_velocity_delta",
            delta,
            Some(team_basis(e.is_team_0)),
        )?;
        values.float("subject_bump_upright_after", upright)?;
        values.float("subject_bump_delay_to_closest_approach", timing)?;
        let mut demos = 0;
        for event in c.events("demolition") {
            if event.participant("attacker")? == &e.player && in_range(event.frame("frame")?, e) {
                demos += 1;
            }
        }
        count_usize(values, "subject_demo_inflicted_count", demos)?;
        Ok(())
    }
}

fn in_range(frame: usize, e: &Evidence) -> bool {
    frame >= e.start_frame && frame <= e.passage_frame
}

fn count(values: &mut Values, name: &str, count: Option<u32>) {
    values.0.insert(name.into(), count.map(Measurement::Count));
}
fn count_usize(values: &mut Values, name: &str, value: usize) -> Result<(), String> {
    count(
        values,
        name,
        Some(u32::try_from(value).map_err(|error| error.to_string())?),
    );
    Ok(())
}

fn maximum(start: usize, end: usize, mut value: impl FnMut(usize) -> Option<f64>) -> Option<f64> {
    let mut maximum = value(start)?;
    for frame in start + 1..=end {
        maximum = maximum.max(value(frame)?);
    }
    Some(maximum)
}

fn closing_speed(car: &Body, ball: &Body) -> Option<f64> {
    Some(
        car.linear_velocity?
            .dot(ball.location.subtract(car.location).unit()?),
    )
}

fn touch_id(event: &Event) -> Result<u64, String> {
    event
        .field("touch_id")?
        .as_u64()
        .ok_or_else(|| format!("invalid native touch ID: {}", event.meta.id))
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

fn followup_features(
    context: &Context<'_>,
    e: &Evidence,
    first: &[&Event],
    second: &[&Event],
    previous: &[&Event],
    values: &mut Values,
) -> Result<(), String> {
    for (prefix, events) in [("first_touch", first), ("second_touch", second)] {
        for relation in ["self", "teammate", "opponent"] {
            let mut found = false;
            for event in events {
                let actor = if event.participant("player")? == &e.player {
                    "self"
                } else if event.boolean("is_team_0")? == e.is_team_0 {
                    "teammate"
                } else {
                    "opponent"
                };
                found |= actor == relation;
            }
            values.boolean(
                &format!("{prefix}_{relation}"),
                (!events.is_empty()).then_some(found),
            );
        }
        values.float(
            &format!("{prefix}_delay_s"),
            events
                .first()
                .map(|event| event.number("time").map(|time| time - e.passage_time))
                .transpose()?,
        )?;
    }
    values.float(
        "previous_subject_touch_delay_s",
        previous
            .first()
            .map(|event| event.number("time").map(|time| e.start_time - time))
            .transpose()?,
    )?;
    let play_end = context
        .analysis
        .outputs
        .replay_data
        .frame_data
        .metadata_frames[e.play_end_frame]
        .time;
    context.goal_features(e.passage_time, play_end, e.is_team_0, values)?;
    Ok(())
}
