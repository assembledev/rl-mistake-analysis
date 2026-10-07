use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use subtr_actor::{StatsFrameResolution, StatsTimelineEventCollector};

pub fn events(bytes: &[u8]) -> Result<Value, Box<dyn std::error::Error>> {
    let replay = boxcars::ParserBuilder::new(bytes)
        .must_parse_network_data()
        .on_error_check_crc()
        .parse()?;
    let rocket_league_id = replay
        .properties
        .iter()
        .find(|(name, _)| name == "Id")
        .map(|(_, property)| property)
        .and_then(|property| match property {
            boxcars::HeaderProp::Str(value) => Some(value.clone()),
            _ => None,
        });
    let timeline = StatsTimelineEventCollector::new()
        .with_frame_resolution(StatsFrameResolution::TimeStep { seconds: 1.0 })
        .get_replay_stats_timeline_scaffold(&replay)
        .map_err(|error| format!("subtr-actor extraction failed: {error:?}"))?;
    let events = &timeline.events.events;
    Ok(json!({
        "name": "subtr-actor",
        "schema_version": 1,
        "status": "ok",
        "revision": env!("SUBTR_ACTOR_REVISION"),
        "replay_sha256": format!("{:x}", Sha256::digest(bytes)),
        "rocket_league_id": rocket_league_id,
        "events": events,
    }))
}
