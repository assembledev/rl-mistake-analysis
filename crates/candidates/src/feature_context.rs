use crate::feature_input::{Body, Event, PlayerData, ReplayAnalysis};
use crate::feature_values::{Measurement, Values};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

pub(crate) struct Context<'a> {
    pub(crate) analysis: &'a ReplayAnalysis,
    pub(crate) streams: HashMap<&'a str, Vec<&'a Event>>,
}

impl<'a> Context<'a> {
    pub(crate) fn new(
        analysis: &'a ReplayAnalysis,
        checked_streams: &[&str],
    ) -> Result<Self, String> {
        analysis.validate()?;
        let mut streams: HashMap<&str, Vec<&Event>> = HashMap::new();
        for event in &analysis.outputs.events.events.events {
            let stream = event.meta.stream.as_str();
            let expected = match stream {
                "air_dribble" => "ball_carry",
                "boost_pickups" => "boost_pickup",
                stream => stream,
            };
            if checked_streams.contains(&stream) && event.payload.kind != expected {
                return Err(format!(
                    "event {} has inconsistent stream and payload kind",
                    event.meta.id
                ));
            }
            streams.entry(stream).or_default().push(event);
        }
        Ok(Self { analysis, streams })
    }

    pub(crate) fn events(&self, stream: &str) -> &[&'a Event] {
        match self.streams.get(stream) {
            Some(events) => events,
            None => &[],
        }
    }

    pub(crate) fn player(&self, id: &Value) -> Result<&'a PlayerData, String> {
        self.analysis
            .outputs
            .replay_data
            .frame_data
            .players
            .iter()
            .find(|(player, _)| player == id)
            .map(|(_, data)| data)
            .ok_or_else(|| format!("player frames are missing for {id}"))
    }

    pub(crate) fn next(
        &self,
        stream: &str,
        time_field: &str,
        start: f64,
        end: f64,
        participant: Option<(&str, &Value)>,
    ) -> Result<Option<&'a Event>, String> {
        let mut next: Option<&Event> = None;
        for event in self.events(stream) {
            if let Some((field, player)) = participant
                && event.participant(field)? != player
            {
                continue;
            }
            let time = event.number(time_field)?;
            if time >= start
                && time <= end
                && (next.is_none() || time < next.unwrap().number(time_field)?)
            {
                next = Some(event);
            }
        }
        Ok(next)
    }

    pub(crate) fn state(
        &self,
        stream: &str,
        player: Option<&Value>,
        time: f64,
    ) -> Result<Option<&'a str>, String> {
        self.state_event(stream, player, time)?
            .map(|event| {
                event.text(if stream == "possession" {
                    "possession_state"
                } else {
                    "state"
                })
            })
            .transpose()
    }

    pub(crate) fn state_event(
        &self,
        stream: &str,
        player: Option<&Value>,
        time: f64,
    ) -> Result<Option<&'a Event>, String> {
        let mut state = None;
        for event in self.events(stream) {
            if let Some(player) = player
                && event.participant("player")? != player
            {
                continue;
            }
            let start = event.number("time")?;
            let end = event.number("end_time")?;
            if start < time && time <= end {
                if state.is_some() {
                    return Err(format!("overlapping {stream} state spans at {time}"));
                }
                event.text(if stream == "possession" {
                    "possession_state"
                } else {
                    "state"
                })?;
                state = Some(*event);
            }
        }
        Ok(state)
    }

    pub(crate) fn bounds(&self, time: f64) -> Result<(f64, f64), String> {
        let metadata = &self.analysis.outputs.replay_data.frame_data.metadata_frames;
        let mut start = metadata[0].time;
        let mut end = metadata.last().unwrap().time;
        for goal in self.events("goal_context") {
            let goal_time = goal.number("time")?;
            if goal_time < time {
                start = start.max(goal_time);
            } else {
                end = end.min(goal_time);
            }
        }
        for kickoff in self.events("kickoff") {
            let kickoff_start = kickoff.number("start_time")?;
            if kickoff_start > time {
                end = end.min(kickoff_start);
            } else if kickoff_start >= start {
                start = start.max(kickoff.number("live_action_start_time")?);
            }
        }
        if start > time || end < time {
            return Err("event is outside its observed live-play segment".into());
        }
        Ok((start, end))
    }

    pub(crate) fn states_by_frame(
        &self,
        stream: &str,
        player: &Value,
    ) -> Result<Vec<Option<&'a str>>, String> {
        let frames = &self.analysis.outputs.replay_data.frame_data.metadata_frames;
        let mut states = vec![None; frames.len()];
        for event in self.events(stream) {
            if event.participant("player")? != player {
                continue;
            }
            let start = event.number("time")?;
            let end = event.number("end_time")?;
            if end < start {
                return Err(format!("reversed {stream} state span"));
            }
            let first = frames.partition_point(|frame| frame.time <= start);
            let after = frames.partition_point(|frame| frame.time <= end);
            let state = event.text("state")?;
            for current in &mut states[first..after] {
                if current.replace(state).is_some() {
                    return Err(format!("overlapping {stream} state spans"));
                }
            }
        }
        Ok(states)
    }
}

pub(crate) fn tag<'a>(event: &'a Event, group: &str) -> Result<Option<&'a str>, String> {
    let tags = event
        .field("tags")?
        .as_array()
        .ok_or("touch tags must be an array")?;
    let mut result = None;
    for tag in tags {
        let tag_group = tag
            .get("group")
            .and_then(Value::as_str)
            .ok_or("invalid touch tag group")?;
        let value = tag
            .get("value")
            .and_then(Value::as_str)
            .ok_or("invalid touch tag value")?;
        if tag_group == group && result.replace(value).is_some() {
            return Err(format!("duplicate touch tag group {group}"));
        }
    }
    Ok(result)
}

pub(crate) fn possession_state(state: &str, team: bool) -> Result<&'static str, String> {
    match state {
        "neutral" => Ok("neutral"),
        "team_zero" => Ok(if team { "own_team" } else { "opponent" }),
        "team_one" => Ok(if team { "opponent" } else { "own_team" }),
        _ => Err(format!("unsupported possession state: {state}")),
    }
}

pub(crate) fn boost_percent(boost: f64) -> f64 {
    boost * 100.0 / u8::MAX as f64
}

impl<'a> Context<'a> {
    pub(crate) fn previous_frame(&self, frame: usize, start: f64) -> Option<usize> {
        frame.checked_sub(1).filter(|index| {
            self.analysis.outputs.replay_data.frame_data.metadata_frames[*index].time >= start
        })
    }

    pub(crate) fn ball(&self, frame: Option<usize>) -> Option<&'a Body> {
        frame.and_then(|frame| {
            self.analysis
                .outputs
                .replay_data
                .frame_data
                .ball_data
                .frames[frame]
                .body()
        })
    }

    pub(crate) fn validate_touch_sample(&self, touch: &Event) -> Result<(), String> {
        let frame = touch.frame("sample_frame")?;
        let time = touch.number("sample_time")?;
        if self
            .analysis
            .outputs
            .replay_data
            .frame_data
            .metadata_frames
            .get(frame)
            .is_none_or(|metadata| metadata.time != time)
            || frame < touch.frame("frame")?
        {
            return Err("invalid native touch sampling frame/time".into());
        }
        Ok(())
    }

    pub(crate) fn boost_before(
        &self,
        player: &Value,
        frame: Option<usize>,
    ) -> Result<Option<f64>, String> {
        let track = self
            .analysis
            .outputs
            .events
            .accumulation_tracks
            .iter()
            .find(|track| &track.player_id == player && track.quantity == "boost_amount")
            .ok_or_else(|| format!("missing boost-amount track for {player}"))?;
        let boost = frame
            .and_then(|frame| track.points.iter().rev().find(|point| point.frame <= frame))
            .map(|point| point.value);
        if boost.is_some_and(|boost| !(0.0..=u8::MAX as f64).contains(&boost)) {
            return Err("boost level is outside the raw replay scale".into());
        }
        Ok(boost.map(boost_percent))
    }

    pub(crate) fn height_band(
        &self,
        player: &Value,
        frame: Option<usize>,
    ) -> Result<Option<&'a str>, String> {
        let mut height = None;
        if let Some(frame) = frame {
            for event in self.events("movement") {
                if event.participant("player")? == player
                    && event.frame("frame")? <= frame
                    && event.frame("end_frame")? >= frame
                {
                    if height.is_some() {
                        return Err("overlapping player movement spans".into());
                    }
                    height = Some(event.text("height_band")?);
                }
            }
        }
        Ok(height)
    }

    pub(crate) fn active_interval(
        &self,
        stream: &str,
        participant: &str,
        player: &Value,
        time: f64,
        play_start: Option<f64>,
    ) -> Result<bool, String> {
        let mut active = false;
        for event in self.events(stream) {
            if event.participant(participant)? != player {
                continue;
            }
            let (start, end) = match stream {
                "pass" => (event.number("start_time")?, event.number("time")?),
                "shadow_defense" => (event.number("time")?, event.number("end_time")?),
                _ => (event.number("start_time")?, event.number("end_time")?),
            };
            active |= start < time && end >= time && play_start.is_none_or(|bound| start >= bound);
        }
        Ok(active)
    }

    pub(crate) fn possession_change(
        &self,
        time: f64,
        end: f64,
        team: bool,
        before: Option<&str>,
    ) -> Result<Option<(f64, &'static str)>, String> {
        let Some(before) = before else {
            return Ok(None);
        };
        let mut change = None;
        for event in self.events("possession") {
            let start = event.number("time")?;
            let state = possession_state(event.text("possession_state")?, team)?;
            if start >= time
                && start < end
                && state != before
                && change.is_none_or(|(previous, _)| start < previous)
            {
                change = Some((start, state));
            }
        }
        Ok(change)
    }

    pub(crate) fn goal_features(
        &self,
        time: f64,
        end: f64,
        team: bool,
        values: &mut Values,
    ) -> Result<(), String> {
        let goal = self.next("goal_context", "time", time, end, None)?;
        for (name, same_team) in [("own_team", true), ("opponent", false)] {
            values.boolean(
                &format!("next_goal_{name}"),
                Some(match goal {
                    Some(event) => (event.boolean("scoring_team_is_team_0")? == team) == same_team,
                    None => false,
                }),
            );
        }
        values.float(
            "next_goal_delay_s",
            goal.map(|event| event.number("time").map(|goal| goal - time))
                .transpose()?,
        )?;
        let count = if let Some(goal) = goal {
            let mut touches = HashSet::new();
            for touch in self.events("touch") {
                let touch_time = touch.number("time")?;
                if touch_time > time && touch_time < goal.number("time")? {
                    let id = touch
                        .field("touch_id")?
                        .as_u64()
                        .ok_or("scoring sequence has an unidentified touch")?;
                    touches.insert(id);
                }
            }
            Some(Measurement::Count(
                u32::try_from(touches.len()).map_err(|error| error.to_string())?,
            ))
        } else {
            None
        };
        values.0.insert("touches_before_next_goal".into(), count);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Context;
    use crate::feature_input::ReplayAnalysis;
    use serde_json::{Value, json};

    fn event(id: &str, stream: &str, payload: Value) -> Value {
        json!({"meta": {"id": id, "stream": stream}, "payload": {"kind": stream, "payload": payload}})
    }

    fn analysis(events: Vec<Value>) -> ReplayAnalysis {
        serde_json::from_value(json!({
            "name": "subtr-actor", "schema_version": 1, "status": "ok", "revision": "supplier", "replay_sha256": "a".repeat(64),
            "outputs": {
                "events": {"replay_meta": null, "accumulation_tracks": [], "events": {"events": events}},
                "replay_data": {"boost_pad_events": [], "boost_pads": [], "frame_data": {
                    "metadata_frames": [{"time": 0.0}], "ball_data": {"frames": ["Empty"]}, "players": []
                }}
            }
        })).unwrap()
    }

    #[test]
    fn state_event_preserves_native_start_exclusive_end_inclusive_boundaries() {
        let analysis = analysis(vec![
            event(
                "before",
                "possession",
                json!({"time": 0.0, "end_time": 1.0, "possession_state": "neutral"}),
            ),
            event(
                "after",
                "possession",
                json!({"time": 1.0, "end_time": 2.0, "possession_state": "team_zero"}),
            ),
        ]);
        let context = Context::new(&analysis, &["possession"]).unwrap();
        for (time, id, state) in [
            (0.0, None, None),
            (1.0, Some("before"), Some("neutral")),
            (1.5, Some("after"), Some("team_zero")),
            (2.0, Some("after"), Some("team_zero")),
            (2.5, None, None),
        ] {
            assert_eq!(
                context
                    .state_event("possession", None, time)
                    .unwrap()
                    .map(|event| event.meta.id.as_str()),
                id
            );
            assert_eq!(context.state("possession", None, time).unwrap(), state);
        }
    }

    #[test]
    fn state_event_keeps_participant_filtering_and_overlap_errors() {
        let blue = json!({"Epic": "blue"});
        let orange = json!({"Epic": "orange"});
        let analysis = analysis(vec![
            event(
                "blue",
                "player_activity",
                json!({"time": 0.0, "end_time": 2.0, "player": blue, "state": "tracked"}),
            ),
            event(
                "orange",
                "player_activity",
                json!({"time": 0.0, "end_time": 2.0, "player": orange, "state": "demolished"}),
            ),
        ]);
        let context = Context::new(&analysis, &["player_activity"]).unwrap();
        assert_eq!(
            context
                .state_event("player_activity", Some(&blue), 1.0)
                .unwrap()
                .unwrap()
                .meta
                .id,
            "blue"
        );
        assert_eq!(
            context
                .state("player_activity", Some(&orange), 1.0)
                .unwrap(),
            Some("demolished")
        );
        let error = "overlapping player_activity state spans at 1";
        assert_eq!(
            context
                .state_event("player_activity", None, 1.0)
                .unwrap_err(),
            error
        );
        assert_eq!(
            context.state("player_activity", None, 1.0).unwrap_err(),
            error
        );
    }

    #[test]
    fn state_event_validates_matching_state_text_before_a_later_overlap() {
        let analysis = analysis(vec![
            event(
                "malformed",
                "possession",
                json!({"time": 0.0, "end_time": 2.0, "possession_state": 7}),
            ),
            event(
                "overlap",
                "possession",
                json!({"time": 0.0, "end_time": 2.0, "possession_state": "neutral"}),
            ),
        ]);
        let context = Context::new(&analysis, &["possession"]).unwrap();
        let error = "event malformed has invalid possession_state";
        assert_eq!(
            context.state_event("possession", None, 1.0).unwrap_err(),
            error
        );
        assert_eq!(context.state("possession", None, 1.0).unwrap_err(), error);
        assert!(
            context
                .state_event("possession", None, 0.0)
                .unwrap()
                .is_none()
        );
    }
}
