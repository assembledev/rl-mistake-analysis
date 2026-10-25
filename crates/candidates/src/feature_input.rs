use crate::input::is_sha256;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Deserialize)]
pub struct ReplayAnalysis {
    name: String,
    schema_version: u32,
    status: String,
    pub(crate) revision: String,
    pub(crate) replay_sha256: String,
    pub(crate) outputs: Outputs,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Outputs {
    pub events: Timeline,
    pub replay_data: ReplayData,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Timeline {
    pub replay_meta: Option<Roster>,
    pub events: Events,
    pub accumulation_tracks: Vec<Track>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Roster {
    pub team_zero: Vec<RosterPlayer>,
    pub team_one: Vec<RosterPlayer>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RosterPlayer {
    pub remote_id: Value,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Events {
    pub events: Vec<Event>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Event {
    pub meta: EventMeta,
    pub payload: Payload,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EventMeta {
    pub id: String,
    pub stream: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Payload {
    pub kind: String,
    pub payload: Value,
}

impl Event {
    pub fn field(&self, name: &str) -> Result<&Value, String> {
        self.payload
            .payload
            .get(name)
            .ok_or_else(|| format!("event {} is missing {name}", self.meta.id))
    }

    pub fn number(&self, name: &str) -> Result<f64, String> {
        self.field(name)?
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("event {} has invalid {name}", self.meta.id))
    }

    pub fn frame(&self, name: &str) -> Result<usize, String> {
        let value = self
            .field(name)?
            .as_u64()
            .ok_or_else(|| format!("event {} has invalid {name}", self.meta.id))?;
        usize::try_from(value).map_err(|error| error.to_string())
    }

    pub fn text(&self, name: &str) -> Result<&str, String> {
        self.field(name)?
            .as_str()
            .ok_or_else(|| format!("event {} has invalid {name}", self.meta.id))
    }

    pub fn boolean(&self, name: &str) -> Result<bool, String> {
        self.field(name)?
            .as_bool()
            .ok_or_else(|| format!("event {} has invalid {name}", self.meta.id))
    }

    pub fn participant(&self, name: &str) -> Result<&Value, String> {
        let value = self.field(name)?;
        if value.is_null() {
            return Err(format!("event {} has null {name}", self.meta.id));
        }
        Ok(value)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct Track {
    pub player_id: Value,
    pub quantity: String,
    pub points: Vec<Point>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Point {
    pub frame: usize,
    pub value: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReplayData {
    pub frame_data: FrameData,
    pub boost_pad_events: Vec<PadEvent>,
    pub boost_pads: Vec<Pad>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FrameData {
    pub ball_data: BallData,
    pub players: Vec<(Value, PlayerData)>,
    pub metadata_frames: Vec<Metadata>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct BallData {
    pub frames: Vec<BallFrame>,
}

#[derive(Debug, Deserialize)]
pub(crate) enum BallFrame {
    Empty,
    Data { rigid_body: Body },
}

#[derive(Debug, Deserialize)]
pub(crate) struct PlayerData {
    pub frames: Vec<PlayerFrame>,
}

#[derive(Debug, Deserialize)]
pub(crate) enum PlayerFrame {
    Empty,
    Data {
        rigid_body: Body,
        dodge_active: bool,
    },
}

impl PlayerFrame {
    pub fn body(&self) -> Option<&Body> {
        match self {
            Self::Empty => None,
            Self::Data { rigid_body, .. } => Some(rigid_body),
        }
    }
}

impl BallFrame {
    pub fn body(&self) -> Option<&Body> {
        match self {
            Self::Empty => None,
            Self::Data { rigid_body } => Some(rigid_body),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct Body {
    pub location: Vector,
    pub rotation: Rotation,
    #[serde(deserialize_with = "required_nullable")]
    pub linear_velocity: Option<Vector>,
}

fn required_nullable<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer)
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) struct Vector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector {
    pub fn length(self) -> f64 {
        self.x.hypot(self.y).hypot(self.z)
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn subtract(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub fn horizontal_distance(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }

    pub fn horizontal_unit(self) -> Option<Self> {
        let length = self.x.hypot(self.y);
        (length > 0.0).then(|| Self {
            x: self.x / length,
            y: self.y / length,
            z: 0.0,
        })
    }

    pub fn project(self, toward: Self) -> [f64; 3] {
        [
            self.x * toward.x + self.y * toward.y,
            -self.x * toward.y + self.y * toward.x,
            self.z,
        ]
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct Rotation {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

impl Rotation {
    pub fn upright(&self) -> Result<f64, String> {
        let norm = self.norm()?;
        Ok(1.0 - 2.0 * (self.x * self.x + self.y * self.y) / norm)
    }

    fn norm(&self) -> Result<f64, String> {
        let norm = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        if !norm.is_finite() || norm == 0.0 {
            return Err("invalid car rotation quaternion".into());
        }
        Ok(norm)
    }

    pub fn forward(&self) -> Result<Vector, String> {
        let norm = self.norm()?;
        Ok(Vector {
            x: 1.0 - 2.0 * (self.y * self.y + self.z * self.z) / norm,
            y: 2.0 * (self.x * self.y + self.w * self.z) / norm,
            z: 2.0 * (self.x * self.z - self.w * self.y) / norm,
        })
    }

    pub fn up(&self) -> Result<Vector, String> {
        let norm = self.norm()?;
        Ok(Vector {
            x: 2.0 * (self.x * self.z + self.w * self.y) / norm,
            y: 2.0 * (self.y * self.z - self.w * self.x) / norm,
            z: 1.0 - 2.0 * (self.x * self.x + self.y * self.y) / norm,
        })
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct Metadata {
    pub time: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PadEvent {
    pub time: f64,
    pub frame: usize,
    pub pad_id: String,
    #[serde(deserialize_with = "required_nullable")]
    pub player: Option<Value>,
    pub kind: PadEventKind,
}

#[derive(Debug, Deserialize)]
pub(crate) enum PadEventKind {
    PickedUp {
        #[serde(rename = "sequence")]
        _sequence: u8,
    },
    Available,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Pad {
    #[serde(deserialize_with = "required_nullable")]
    pub pad_id: Option<String>,
    pub position: Vector,
}

impl ReplayAnalysis {
    pub(crate) fn facts(&self) -> crate::ReplayFacts {
        crate::ReplayFacts {
            name: self.name.clone(),
            schema_version: self.schema_version,
            status: self.status.clone(),
            revision: self.revision.clone(),
            replay_sha256: self.replay_sha256.clone(),
            rocket_league_id: None,
            events: self.outputs.events.events.events.iter().map(|event| {
                crate::input::MechanicEvent {
                    meta: crate::input::EventMeta {
                        id: event.meta.id.clone(), stream: event.meta.stream.clone(),
                    },
                    payload: serde_json::json!({"kind": event.payload.kind, "payload": event.payload.payload}),
                }
            }).collect(),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.name != "subtr-actor"
            || self.schema_version != 1
            || self.status != "ok"
            || self.revision.trim().is_empty()
            || !is_sha256(&self.replay_sha256)
        {
            return Err("invalid combined subtr-actor analysis identity".into());
        }
        let frames = &self.outputs.replay_data.frame_data;
        let count = frames.metadata_frames.len();
        if count == 0
            || frames.ball_data.frames.len() != count
            || frames
                .metadata_frames
                .iter()
                .any(|frame| !frame.time.is_finite())
            || frames
                .metadata_frames
                .windows(2)
                .any(|pair| pair[0].time > pair[1].time)
        {
            return Err("missing, inconsistent, or unordered native replay frames".into());
        }
        let mut players = HashSet::new();
        for (id, data) in &frames.players {
            if id.is_null() || !players.insert(id.to_string()) || data.frames.len() != count {
                return Err("missing, duplicate, or inconsistent player frames".into());
            }
        }
        let mut events = HashSet::new();
        for event in &self.outputs.events.events.events {
            if event.meta.id.trim().is_empty()
                || event.meta.stream.trim().is_empty()
                || !events.insert(&event.meta.id)
            {
                return Err("event IDs must be nonempty and unique".into());
            }
        }
        let mut tracks = HashSet::new();
        for track in &self.outputs.events.accumulation_tracks {
            if !tracks.insert((track.player_id.to_string(), &track.quantity))
                || track
                    .points
                    .iter()
                    .any(|point| point.frame >= count || !point.value.is_finite())
                || track
                    .points
                    .windows(2)
                    .any(|pair| pair[0].frame >= pair[1].frame)
            {
                return Err("invalid or duplicate analysis numeric track".into());
            }
        }
        Ok(())
    }
}
