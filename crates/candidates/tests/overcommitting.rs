use rl_mistake_analysis_candidates::{
    CandidateBatch, MistakeKind, feature_contract::ReplayAnalysis,
};
use serde_json::{Value, json};

fn event(id: &str, stream: &str, payload: Value) -> Value {
    json!({"meta": {"id": id, "stream": stream}, "payload": {"kind": stream, "payload": payload}})
}

fn touch(frame: usize, team: bool, shot: bool) -> Value {
    event(
        &format!("touch:{frame}"),
        "touch",
        json!({
            "frame": frame, "time": frame as f64, "touch_id": frame,
            "player": {"Epic": if team { "blue" } else { "orange" }},
            "is_team_0": team, "tags": if shot { json!([{"group": "action", "value": "shot"}]) } else { json!([]) }
        }),
    )
}

fn body(x: f64, y: f64, vy: f64) -> Value {
    json!({"Data": {"dodge_active": false, "rigid_body": {
        "location": {"x": x, "y": y, "z": 0.},
        "linear_velocity": {"x": 0., "y": vy, "z": 0.},
        "rotation": {"x": 0., "y": 0., "z": 0., "w": 1.}
    }}})
}

fn fixture() -> Value {
    let blue = json!({"Epic": "blue"});
    let orange = json!({"Epic": "orange"});
    let activity = |name: &str, player: &Value| {
        event(
            name,
            "player_activity",
            json!({"player": player, "time": -1., "end_time": 8., "state": "tracked"}),
        )
    };
    json!({
        "name": "subtr-actor", "schema_version": 1, "status": "ok", "revision": "supplier", "replay_sha256": "a".repeat(64),
        "outputs": {
            "events": {
                "replay_meta": {"team_zero": [{"remote_id": blue}], "team_one": [{"remote_id": orange}]},
                "accumulation_tracks": [],
                "events": {"events": [
                    activity("blue:activity", &blue), activity("orange:activity", &orange),
                    event("possession", "possession", json!({"time": -1., "end_time": 8., "active": true, "possession_state": "neutral"})),
                    touch(0, false, false), touch(2, false, false), touch(6, true, false)
                ]}
            },
            "replay_data": {"boost_pad_events": [], "boost_pads": [], "frame_data": {
                "metadata_frames": (0..=8).map(|frame| json!({"time": frame as f64})).collect::<Vec<_>>(),
                "ball_data": {"frames": (0..=8).map(|frame| body(0., if frame < 4 {100.} else {20.}, -1.)).collect::<Vec<_>>()},
                "players": [
                    [blue, {"frames": (0..=8).map(|frame| body(0., frame as f64 * 10., if frame == 0 {0.} else {10.})).collect::<Vec<_>>()}],
                    [orange, {"frames": (0..=8).map(|_| body(0., 200., 0.)).collect::<Vec<_>>()}]
                ]
            }}
        }
    })
}

fn select(document: Value) -> CandidateBatch {
    let analysis: ReplayAnalysis = serde_json::from_value(document).unwrap();
    MistakeKind::Overcommitting
        .candidates_from_analysis(&analysis)
        .unwrap()
}

fn events(document: &mut Value) -> &mut Vec<Value> {
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
}

fn teammate(document: &mut Value, name: &str, x: f64, y: f64, state: &str) {
    let id = json!({"Epic": name});
    document["outputs"]["events"]["replay_meta"]["team_zero"]
        .as_array_mut()
        .unwrap()
        .push(json!({"remote_id": id}));
    document["outputs"]["replay_data"]["frame_data"]["players"]
        .as_array_mut()
        .unwrap()
        .push(json!([id, {"frames": (0..=8).map(|_| body(x, y, 0.)).collect::<Vec<_>>() }]));
    events(document).push(event(
        &format!("{name}:activity"),
        "player_activity",
        json!({"player": id, "time": -1., "end_time": 8., "state": state}),
    ));
}

#[test]
fn duel_approach_finishes_at_first_passage_without_a_duration_cutoff() {
    let batch = select(fixture());
    assert_eq!(batch.candidates.len(), 1);
    let candidate = &batch.candidates[0];
    assert_eq!(candidate.frame, 1);
    assert_eq!(candidate.evidence["end_frame"], 4);
    assert_eq!(candidate.evidence["next_own_touch_frame"], 6);
    assert_eq!(candidate.evidence["opponent_touch_ids"], json!(["touch:2"]));
    assert!(candidate.features.is_empty());

    let mut document = fixture();
    for metadata in document["outputs"]["replay_data"]["frame_data"]["metadata_frames"]
        .as_array_mut()
        .unwrap()
    {
        metadata["time"] = json!(metadata["time"].as_f64().unwrap() / 100.);
    }
    for event in events(&mut document) {
        if event["meta"]["stream"] == "touch" {
            let payload = &mut event["payload"]["payload"];
            payload["time"] = json!(payload["time"].as_f64().unwrap() / 100.);
        }
    }
    let batch = select(document);
    assert_eq!(batch.candidates.len(), 1);
    assert_eq!(batch.candidates[0].raw_time, 0.01);
    assert_eq!(batch.candidates[0].evidence["end_time"], 0.04);
}

#[test]
fn touches_and_either_shot_source_are_independent_confirmation_branches() {
    for source in ["touch", "timeline", "tag"] {
        let mut document = fixture();
        if source != "touch" {
            events(&mut document).retain(|event| event["meta"]["id"] != "touch:2");
            let shot = if source == "timeline" {
                event(
                    "shot:5",
                    "timeline",
                    json!({"frame": 5, "time": 5., "kind": "Shot", "is_team_0": false}),
                )
            } else {
                touch(5, false, true)
            };
            events(&mut document).push(shot);
        }
        let batch = select(document);
        assert_eq!(batch.candidates.len(), 1, "{source}");
        if source != "touch" {
            assert_eq!(
                batch.candidates[0].evidence["opponent_touch_ids"],
                json!([])
            );
        }
    }
    let mut document = fixture();
    events(&mut document).retain(|event| event["meta"]["id"] != "touch:2");
    events(&mut document).push(event(
        "shot:6",
        "timeline",
        json!({"frame": 6, "time": 6., "kind": "Shot", "is_team_0": false}),
    ));
    assert!(select(document).candidates.is_empty());
}

#[test]
fn own_touches_and_participating_fifties_cancel_the_attempt() {
    for frame in [2, 4] {
        let mut document = fixture();
        events(&mut document).push(touch(frame, true, false));
        events(&mut document).last_mut().unwrap()["meta"]["id"] = json!(format!("own:{frame}"));
        assert!(select(document).candidates.is_empty());
    }
    for winner in ["team_zero", "team_one", "neutral"] {
        let mut document = fixture();
        events(&mut document).push(event(
            "fifty",
            "fifty_fifty",
            json!({
                "start_frame": 2, "resolve_frame": 4, "winner": winner,
                "team_zero_player": {"Epic": "blue"}, "team_one_player": {"Epic": "orange"}
            }),
        ));
        assert!(select(document).candidates.is_empty());
    }
}

#[test]
fn movement_and_role_are_opening_conditions_without_fragmenting_recovery() {
    let mut document = fixture();
    teammate(&mut document, "mate", 0., 50., "tracked");
    let players = &mut document["outputs"]["replay_data"]["frame_data"]["players"];
    for frame in 2..=8 {
        players[0][1]["frames"][frame]["Data"]["rigid_body"]["linear_velocity"]["y"] = json!(-10.);
        players[2][1]["frames"][frame]["Data"]["rigid_body"]["location"]["y"] = json!(-50.);
    }
    let batch = select(document);
    assert_eq!(batch.candidates.len(), 1);
    assert_eq!(
        (
            batch.candidates[0].frame,
            batch.candidates[0].evidence["end_frame"].as_u64().unwrap()
        ),
        (1, 4)
    );
}

#[test]
fn back_role_uses_deepest_or_farthest_goal_side_with_ties() {
    for (x, y, deepest, farthest) in [(1000., 50., true, false), (0., 0., false, true)] {
        let mut document = fixture();
        if !deepest {
            for frame in document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"]
                .as_array_mut()
                .unwrap()
            {
                frame["Data"]["rigid_body"]["location"]["x"] = json!(100.);
            }
        }
        teammate(&mut document, "mate", x, y, "tracked");
        let batch = select(document);
        assert_eq!(batch.candidates.len(), 1);
        assert_eq!(
            batch.candidates[0].evidence["opening"]["deepest_by_y"],
            deepest
        );
        assert_eq!(
            batch.candidates[0].evidence["opening"]["farthest_goal_side"],
            farthest
        );
    }
    let mut document = fixture();
    teammate(&mut document, "first", 0., 50., "tracked");
    teammate(&mut document, "cover", 1000., -50., "tracked");
    assert!(select(document).candidates.is_empty());
    let mut document = fixture();
    teammate(&mut document, "demoed", 1000., -50., "demolished");
    assert_eq!(select(document).candidates.len(), 1);
    let mut document = fixture();
    teammate(&mut document, "tied", 0., 10., "tracked");
    let batch = select(document);
    let subject = batch
        .candidates
        .iter()
        .find(|candidate| candidate.player_id == json!({"Epic": "blue"}))
        .unwrap();
    assert_eq!(subject.frame, 1);
    assert_eq!(subject.evidence["opening"]["deepest_by_y"], true);
    assert_eq!(subject.evidence["opening"]["farthest_goal_side"], true);
}

#[test]
fn unknown_geometry_aborts_without_substituting_another_frame() {
    let mut document = fixture();
    document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][3] = json!("Empty");
    assert!(select(document).candidates.is_empty());
    let mut document = fixture();
    teammate(&mut document, "mate", 0., 50., "tracked");
    document["outputs"]["replay_data"]["frame_data"]["players"][2][1]["frames"] =
        json!(vec!["Empty"; 9]);
    assert!(select(document).candidates.is_empty());
}

#[test]
fn passage_on_first_untracked_frame_cancels_the_attempt() {
    assert_eq!(select(fixture()).candidates.len(), 1);
    let mut document = fixture();
    events(&mut document)[0]["payload"]["payload"]["end_time"] = json!(3.);
    events(&mut document).push(event(
        "blue:untracked",
        "player_activity",
        json!({"player": {"Epic": "blue"}, "time": 3., "end_time": 8., "state": "untracked"}),
    ));
    assert!(select(document).candidates.is_empty());
}

#[test]
fn exact_interval_anchor_rejects_changed_boundaries_and_player() {
    let batch = select(fixture());
    let mut candidate = batch.candidates[0].clone();
    assert!(candidate.anchor().is_ok());
    candidate.evidence["end_frame"] = json!(1);
    assert!(candidate.anchor().is_err());
    candidate = batch.candidates[0].clone();
    candidate.player_id = json!({"Epic": "another"});
    assert!(candidate.anchor().is_err());
}

#[test]
fn distinct_intervals_can_share_confirmation_events() {
    let mut document = fixture();
    events(&mut document)
        .retain(|event| !matches!(event["meta"]["id"].as_str(), Some("touch:2" | "touch:6")));
    events(&mut document).push(event(
        "shot:5",
        "timeline",
        json!({"frame": 5, "time": 5., "kind": "Shot", "is_team_0": false}),
    ));
    let ball = &mut document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"];
    ball[5] = body(0., 100., -1.);
    ball[6] = body(0., 100., -1.);
    let mut batch = select(document);
    assert_eq!(batch.candidates.len(), 2);
    assert_eq!(batch.candidates[0].frame, 1);
    assert_eq!(batch.candidates[1].frame, 5);
    assert_eq!(
        batch.candidates[0].source_event_ids,
        batch.candidates[1].source_event_ids
    );
    batch.candidates.push(batch.candidates[0].clone());
    assert!(batch.validate().is_err());
}

#[test]
fn contact_and_shot_times_must_match_native_frames() {
    for source in ["touch", "timeline"] {
        let mut document = fixture();
        if source == "touch" {
            events(&mut document)[3]["payload"]["payload"]["time"] = json!(0.5);
        } else {
            events(&mut document).push(event(
                "shot:5",
                "timeline",
                json!({"frame": 5, "time": 5.5, "kind": "Shot", "is_team_0": false}),
            ));
        }
        let analysis: ReplayAnalysis = serde_json::from_value(document).unwrap();
        assert!(
            MistakeKind::Overcommitting
                .candidates_from_analysis(&analysis)
                .unwrap_err()
                .contains("time disagrees with its frame"),
            "{source}"
        );
    }
}

fn retreat_fixture() -> Value {
    let mut document = fixture();
    let frames = &mut document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"];
    frames[2] = body(0., 10., -10.);
    frames[3] = body(0., 20., 10.);
    events(&mut document).retain(|event| event["meta"]["id"] != "touch:2");
    events(&mut document).push(touch(3, false, false));
    document
}

#[test]
fn retreat_finishes_the_approach_before_later_failure_confirmation() {
    let batch = select(retreat_fixture());
    assert_eq!(batch.candidates.len(), 1);
    let candidate = &batch.candidates[0];
    assert_eq!(candidate.frame, 1);
    assert_eq!(candidate.evidence["end_frame"], 2);
    assert_eq!(candidate.evidence["end_time"], 2.);
    assert_eq!(candidate.evidence["finish"], "retreat_to_opening_depth");
    assert_eq!(candidate.evidence["passage_frame"], 4);
    assert_eq!(candidate.evidence["passage_time"], 4.);
    assert_eq!(candidate.evidence["opponent_touch_ids"], json!(["touch:3"]));
    assert_eq!(candidate.evidence["opening"]["position_y"], 10.);
}

#[test]
fn pending_retreat_still_requires_passage_before_own_touch_or_fifty() {
    let mut document = retreat_fixture();
    events(&mut document).push(touch(3, true, false));
    events(&mut document).last_mut().unwrap()["meta"]["id"] = json!("own:3");
    assert!(select(document).candidates.is_empty());

    let mut document = retreat_fixture();
    events(&mut document).push(event(
        "fifty",
        "fifty_fifty",
        json!({
            "start_frame": 3, "resolve_frame": 4,
            "team_zero_player": {"Epic": "blue"}, "team_one_player": {"Epic": "orange"}
        }),
    ));
    assert!(select(document).candidates.is_empty());

    let mut document = retreat_fixture();
    for frame in 4..=8 {
        document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][frame] =
            body(0., 100., -1.);
    }
    assert!(select(document).candidates.is_empty());
}

#[test]
fn reversal_before_starting_depth_and_active_dodges_do_not_finish_the_approach() {
    let mut document = retreat_fixture();
    document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][2] =
        body(0., 15., -10.);
    assert_eq!(select(document).candidates[0].evidence["end_frame"], 4);

    let mut document = retreat_fixture();
    document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][2]["Data"]["dodge_active"] =
        json!(true);
    assert_eq!(select(document).candidates[0].evidence["end_frame"], 4);

    let mut document = retreat_fixture();
    let frames = &mut document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"];
    frames[2]["Data"]["dodge_active"] = json!(true);
    frames[3] = body(0., 5., -10.);
    assert_eq!(select(document).candidates[0].evidence["end_frame"], 3);
}
