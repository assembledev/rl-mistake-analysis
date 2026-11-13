use rl_mistake_analysis_candidates::{
    CandidateBatch, MistakeKind,
    feature_contract::{ContextFeatures, ReplayAnalysis},
    overcommitting_features::{self, FeatureSampling, Measurement, SamplingMethod},
};
use serde_json::{Value, json};

const TIMES: [f64; 9] = [0.0, 0.5, 1.0, 1.5, 3.0, 4.0, 5.0, 6.0, 7.0];
fn event(id: &str, stream: &str, payload: Value) -> Value {
    json!({"meta": {"id": id, "stream": stream}, "payload": {"kind": stream, "payload": payload}})
}
fn body(x: f64, y: f64, z: f64, velocity: [f64; 3]) -> Value {
    json!({"Data": {"dodge_active": false, "jump_active": false, "rigid_body": {
        "location": {"x": x, "y": y, "z": z},
        "linear_velocity": {"x": velocity[0], "y": velocity[1], "z": velocity[2]},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    }}})
}
fn touch(frame: usize, team: bool, action: Option<&str>) -> Value {
    event(
        &format!("touch:{frame}"),
        "touch",
        json!({
            "frame": frame, "time": TIMES[frame], "sample_frame": frame, "sample_time": TIMES[frame], "touch_id": frame,
            "player": {"Epic": if team { "blue" } else { "orange" }}, "is_team_0": team,
            "tags": action.map(|value| vec![json!({"group": "action", "value": value})]).unwrap_or_default()
        }),
    )
}
fn fixture() -> Value {
    let blue = json!({"Epic": "blue"});
    let orange = json!({"Epic": "orange"});
    let activity = |id: &str, player: &Value| {
        event(
            id,
            "player_activity",
            json!({"player": player, "time": -1.0, "end_time": 7.0, "state": "tracked"}),
        )
    };
    json!({
        "name": "subtr-actor", "schema_version": 1, "status": "ok", "revision": "supplier", "replay_sha256": "a".repeat(64),
        "outputs": {
            "events": {
                "replay_meta": {"team_zero": [{"remote_id": blue}], "team_one": [{"remote_id": orange}]},
                "accumulation_tracks": [
                    {"player_id": blue, "quantity": "boost_amount", "points": [{"frame": 0, "value": 128.0}]},
                    {"player_id": blue, "quantity": "boost_used", "points": (0..9).map(|frame| json!({"frame": frame, "value": frame as f64 * 2.0})).collect::<Vec<_>>()}
                ],
                "events": {"events": [activity("blue:activity", &blue), activity("orange:activity", &orange),
                    event("possession", "possession", json!({"time": -1.0, "end_time": 7.0, "active": true, "possession_state": "neutral"})),
                    touch(0, false, None), touch(2, false, None), touch(6, true, None)
                ]}
            },
            "replay_data": {"boost_pad_events": [], "boost_pads": [], "frame_data": {
                "metadata_frames": TIMES.iter().map(|time| json!({"time": time})).collect::<Vec<_>>(),
                "ball_data": {"frames": (0..9).map(|frame| body(5.0, if frame < 4 {100.0} else {20.0}, 7.0, [0.0, -1.0, 0.0])).collect::<Vec<_>>()},
                "players": [
                    [blue, {"frames": (0..9).map(|frame| body(1.0, frame as f64 * 10.0, 2.0, [0.0, frame as f64 * 10.0, 0.0])).collect::<Vec<_>>() }],
                    [orange, {"frames": (0..9).map(|_| body(0.0, 200.0, 0.0, [0.0, 0.0, 0.0])).collect::<Vec<_>>() }]
                ]
            }}
        }
    })
}
fn events(document: &mut Value) -> &mut Vec<Value> {
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
}
fn analyze(document: Value) -> ReplayAnalysis {
    serde_json::from_value(document).unwrap()
}
fn sampling() -> FeatureSampling {
    FeatureSampling {
        method: SamplingMethod::NativeIntervalBoundaries,
    }
}
fn candidates(document: Value) -> CandidateBatch {
    MistakeKind::Overcommitting
        .candidates_from_analysis(&analyze(document))
        .unwrap()
}
fn extracted(document: Value) -> overcommitting_features::FeatureBatch {
    overcommitting_features::extract(&analyze(document), sampling(), None)
        .unwrap()
        .1
}
fn number(row: &overcommitting_features::FeatureRow, name: &str) -> f32 {
    match row.features[name] {
        Some(Measurement::Float(value)) => value,
        ref value => panic!("{name}: {value:?}"),
    }
}

#[test]
fn supplied_intervals_reject_roster_players_without_native_frames() {
    let document = fixture();
    let exact = candidates(document.clone());
    for index in [0, 1] {
        let mut missing = document.clone();
        missing["outputs"]["replay_data"]["frame_data"]["players"]
            .as_array_mut()
            .unwrap()
            .remove(index);
        let error =
            overcommitting_features::extract_for_candidates(&analyze(missing), sampling(), &exact)
                .unwrap_err();
        assert!(error.contains("player frames are missing"), "{error}");
    }
}

#[test]
fn native_approach_uses_irregular_time_weights_and_full_team_basis_vectors() {
    let batch = extracted(fixture());
    assert_eq!(batch.feature_definitions.len(), 88);
    assert_eq!(batch.rows.len(), 1);
    let row = &batch.rows[0];
    assert_eq!((row.frame, row.end_frame, row.passage_frame), (1, 4, 4));
    assert_eq!(number(row, "approach_duration_s"), 2.5);
    assert_eq!(number(row, "subject_mean_speed"), 29.0);
    assert_eq!(number(row, "subject_max_speed"), 40.0);
    assert_eq!(number(row, "subject_max_advance"), 30.0);
    assert_eq!(number(row, "subject_path_distance"), 30.0);
    assert_eq!(number(row, "subject_boost_used"), 6.0);
    assert_eq!(
        number(row, "candidate_subject_position_relative_ball_toward"),
        -90.0
    );
    assert_eq!(
        number(row, "candidate_subject_position_relative_ball_side"),
        4.0
    );
    assert_eq!(
        number(row, "candidate_subject_position_relative_ball_up"),
        -5.0
    );
    assert_eq!(number(row, "candidate_subject_heading_toward"), 0.0);
    assert_eq!(number(row, "candidate_subject_heading_side"), -1.0);
    assert_eq!(number(row, "candidate_subject_heading_up"), 0.0);
    assert_eq!(row.samples.closest_approach_frame, Some(4));
    assert_eq!(row.samples.first_touch.as_ref().unwrap().frame, 6);
    assert_eq!(
        row.features["first_touch_self"],
        Some(Measurement::Boolean(true))
    );
    assert_eq!(row.features["other_teammate_ball_distance_min"], None);
    assert_eq!(
        row.features["approaching_teammate_count"],
        Some(Measurement::Count(0))
    );
}

#[test]
fn observed_zero_native_flags_and_absent_measurements_are_distinct() {
    let document = fixture();
    let original = candidates(document.clone());
    let mut missing = document;
    let frames = &mut missing["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"];
    frames[2]["Data"]["rigid_body"]["linear_velocity"] = Value::Null;
    let features =
        overcommitting_features::extract_for_candidates(&analyze(missing), sampling(), &original)
            .unwrap();
    let row = &features.rows[0];
    assert_eq!(row.features["subject_mean_speed"], None);
    assert_eq!(row.features["subject_max_speed"], None);
    assert_eq!(number(row, "subject_jump_active_s"), 0.0);
    assert_eq!(number(row, "subject_path_distance"), 30.0);
    assert!(row.samples.strongest_bump.is_none());
    assert_eq!(row.features["subject_bump_velocity_delta_toward"], None);
    assert_eq!(row.features["subject_bump_max_impulse"], None);
    assert_eq!(
        row.features["subject_opponent_bump_count"],
        Some(Measurement::Count(0))
    );
    assert_eq!(row.features["closest_opponent_boost_before_pct"], None);
}

#[test]
fn supplied_interval_boundaries_and_all_sources_are_preserved_without_reselection() {
    let document = fixture();
    let mut exact = candidates(document.clone());
    exact.engine_revision = "saved-native-supplier".into();
    let candidate = &mut exact.candidates[0];
    candidate.evidence["end_frame"] = json!(2);
    candidate.evidence["end_time"] = json!(1.0);
    candidate.evidence["finish"] = json!("retreat_to_opening_depth");
    let batch = overcommitting_features::extract_for_candidates(
        &analyze(document.clone()),
        sampling(),
        &exact,
    )
    .unwrap();
    assert_eq!(batch.engine_revision, "supplier");
    assert_eq!(exact.engine_revision, "saved-native-supplier");
    let row = &batch.rows[0];
    assert_eq!((row.frame, row.end_frame, row.passage_frame), (1, 2, 4));
    assert_eq!(number(row, "approach_duration_s"), 0.5);
    assert_eq!(number(row, "subject_mean_speed"), 15.0);
    assert_eq!(number(row, "passage_delay_after_approach_s"), 2.0);
    assert_eq!(row.key().0, "overcommitting:1:2");
    assert_eq!(row.source_event_ids, exact.candidates[0].source_event_ids);
    let mut changed = exact.candidates[0].clone();
    changed.evidence["passage_frame"] = json!(5);
    changed.evidence["passage_time"] = json!(4.0);
    assert!(row.validate_candidate(&changed).is_err());
    let mut changed = exact.clone();
    changed.candidates[0]
        .source_event_ids
        .push("absent-source".into());
    assert!(
        overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &changed)
            .unwrap_err()
            .contains("source event")
    );
}

#[test]
fn either_native_shot_source_qualifies_once_before_own_touch_and_never_at_it() {
    for mode in ["timeline", "touch", "both", "at_own_touch"] {
        let mut document = fixture();
        if mode == "touch" || mode == "both" {
            events(&mut document).push(touch(5, false, Some("shot")));
        }
        if mode != "touch" {
            let frame = if mode == "at_own_touch" { 6 } else { 5 };
            events(&mut document).push(event(
                "native-shot",
                "timeline",
                json!({"frame": frame, "time": TIMES[frame], "kind": "Shot", "is_team_0": false}),
            ));
        }
        let batch = extracted(document);
        assert_eq!(
            batch.rows[0].features["opponent_shot_before_own_touch"],
            Some(Measurement::Boolean(mode != "at_own_touch")),
            "{mode}"
        );
    }
}

#[test]
fn supplied_interval_shots_stop_at_first_own_touch_strictly_after_opening() {
    for (own_frame, expected) in [(1, true), (3, false), (4, false)] {
        let mut document = fixture();
        let mut original = candidates(document.clone());
        events(&mut document).push(touch(own_frame, true, None));
        events(&mut document).push(touch(5, false, Some("shot")));
        original.candidates[0].evidence["next_own_touch_frame"] =
            json!(if own_frame > 1 { own_frame } else { 6 });
        let batch = overcommitting_features::extract_for_candidates(
            &analyze(document),
            sampling(),
            &original,
        )
        .unwrap();
        assert_eq!(
            batch.rows[0].features["opponent_shot_before_own_touch"],
            Some(Measurement::Boolean(expected)),
            "own touch at frame {own_frame}"
        );
    }
}

#[test]
fn activity_gaps_and_native_boundary_time_changes_are_rejected() {
    let document = fixture();
    let original = candidates(document.clone());
    let mut missing = document.clone();
    events(&mut missing)[0]["payload"]["payload"]["end_time"] = json!(1.0);
    let error =
        overcommitting_features::extract_for_candidates(&analyze(missing), sampling(), &original)
            .unwrap_err();
    assert!(error.contains("unknown subject activity"));
    let mut changed = original;
    changed.candidates[0].evidence["passage_time"] = json!(3.1);
    changed.candidates[0].evidence["end_time"] = json!(3.1);
    let error =
        overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &changed)
            .unwrap_err();
    assert!(error.contains("native frame time"));
}

#[test]
fn strongest_received_bump_reuses_exact_peak_then_minimum_samples() {
    let mut document = fixture();
    let original = candidates(document.clone());
    let velocities = [0.0, 0.0, 10.0, 15.0, 15.0, 17.0, 17.0, 17.0, 17.0];
    for (frame, velocity) in velocities.into_iter().enumerate() {
        document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][frame]["Data"]
            ["rigid_body"]["linear_velocity"]["y"] = json!(velocity);
    }
    events(&mut document).push(event("bump:2", "bump", json!({"frame": 2, "time": TIMES[2], "victim": {"Epic": "blue"}, "is_team_bump": false, "victim_impulse": 42.0})));
    let batch =
        overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &original)
            .unwrap();
    let row = &batch.rows[0];
    let impact = &row.samples.strongest_bump.as_ref().unwrap().impact;
    assert_eq!(
        (impact.before_frame, impact.peak_frame, impact.after_frame),
        (Some(1), Some(2), Some(4))
    );
    assert_eq!(number(row, "subject_bump_max_impulse"), 42.0);
    assert_eq!(number(row, "subject_bump_velocity_delta_toward"), 15.0);
    assert_eq!(number(row, "subject_bump_velocity_delta_side"), 0.0);
    assert_eq!(number(row, "subject_bump_velocity_delta_up"), 0.0);
    assert_eq!(number(row, "subject_bump_upright_after"), 1.0);
}

#[test]
fn incomplete_impact_retains_event_evidence_without_inventing_an_after_sample() {
    let mut document = fixture();
    let original = candidates(document.clone());
    events(&mut document).push(event("bump:4", "bump", json!({"frame": 4, "time": TIMES[4], "victim": {"Epic": "blue"}, "is_team_bump": true, "victim_impulse": 42.0})));
    let batch =
        overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &original)
            .unwrap();
    let row = &batch.rows[0];
    assert!(
        row.samples
            .strongest_bump
            .as_ref()
            .unwrap()
            .impact
            .after_frame
            .is_none()
    );
    assert_eq!(number(row, "subject_bump_max_impulse"), 42.0);
    assert_eq!(row.features["subject_bump_velocity_delta_toward"], None);
    assert_eq!(row.features["subject_bump_upright_after"], None);
    assert_eq!(
        row.features["subject_team_bump_count"],
        Some(Measurement::Count(1))
    );
}

#[test]
fn canonical_vector_triplets_are_invariant_under_team_and_field_rotation() {
    let original = extracted(fixture());
    let mut document = fixture();
    let roster = &mut document["outputs"]["events"]["replay_meta"];
    let team_zero = roster["team_zero"].clone();
    roster["team_zero"] = roster["team_one"].clone();
    roster["team_one"] = team_zero;
    for event in events(&mut document) {
        let payload = &mut event["payload"]["payload"];
        if let Some(team) = payload.get("is_team_0").and_then(Value::as_bool) {
            payload["is_team_0"] = json!(!team);
        }
    }
    let native = &mut document["outputs"]["replay_data"]["frame_data"];
    let rotate = |frame: &mut Value| {
        let body = &mut frame["Data"]["rigid_body"];
        for field in ["location", "linear_velocity"] {
            for axis in ["x", "y"] {
                body[field][axis] = json!(-body[field][axis].as_f64().unwrap());
            }
        }
        body["rotation"] = json!({"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0});
    };
    for frame in native["ball_data"]["frames"].as_array_mut().unwrap() {
        rotate(frame);
    }
    for player in native["players"].as_array_mut().unwrap() {
        for frame in player[1]["frames"].as_array_mut().unwrap() {
            rotate(frame);
        }
    }
    let transformed = extracted(document);
    for prefix in [
        "candidate_subject_velocity",
        "candidate_subject_heading",
        "candidate_subject_position_relative_ball",
        "candidate_ball_position",
        "candidate_ball_velocity",
    ] {
        for axis in ["toward", "side", "up"] {
            let name = format!("{prefix}_{axis}");
            assert_eq!(
                original.rows[0].features[&name], transformed.rows[0].features[&name],
                "{name}"
            );
        }
    }
}

#[test]
fn teammate_goal_distance_measures_the_car_center_without_ball_radius() {
    let mut document = fixture();
    let teammate = json!({"Epic": "mate"});
    document["outputs"]["events"]["replay_meta"]["team_zero"]
        .as_array_mut()
        .unwrap()
        .push(json!({"remote_id": teammate}));
    document["outputs"]["replay_data"]["frame_data"]["players"].as_array_mut().unwrap().push(json!([teammate, {"frames": (0..9).map(|_| body(0.0, 50.0, 20.0, [0.0, 0.0, 0.0])).collect::<Vec<_>>() }]));
    events(&mut document).push(event(
        "mate:activity",
        "player_activity",
        json!({"player": teammate, "time": -1.0, "end_time": 7.0, "state": "tracked"}),
    ));
    let batch = extracted(document);
    let row = batch
        .rows
        .iter()
        .find(|row| row.player_id == json!({"Epic": "blue"}))
        .unwrap();
    assert_eq!(
        number(row, "teammate_goal_distance_min"),
        subtr_actor::STANDARD_GOAL_LINE_Y + 50.0
    );
    assert_eq!(number(row, "subject_depth_lead"), 40.0);
    assert_eq!(row.samples.closest_teammate_id, Some(teammate));
}

#[test]
fn missing_native_frame_does_not_substitute_an_observed_closest_approach() {
    let document = fixture();
    let original = candidates(document.clone());
    let mut missing = document;
    missing["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][2] = json!("Empty");
    let features =
        overcommitting_features::extract_for_candidates(&analyze(missing), sampling(), &original)
            .unwrap();
    let row = &features.rows[0];
    assert_eq!(row.samples.closest_approach_frame, None);
    assert_eq!(row.features["min_subject_ball_distance"], None);
    assert_eq!(row.features["subject_path_distance"], None);
    assert_eq!(
        number(row, "candidate_subject_position_relative_ball_toward"),
        -90.0
    );
}

#[test]
fn measurement_provenance_uses_the_analysis_supplier_without_changing_saved_candidates() {
    let mut document = fixture();
    document["revision"] = json!("supplier-old");
    let exact = candidates(document.clone());
    document["revision"] = json!("supplier-new");
    let batch =
        overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
            .unwrap();
    assert_eq!(batch.engine_revision, "supplier-new");
    assert_eq!(exact.engine_revision, "supplier-old");
    assert_eq!(
        batch.rows[0].source_event_ids,
        exact.candidates[0].source_event_ids
    );
    assert_eq!(batch.rows[0].frame as u64, exact.candidates[0].frame);
}

#[test]
fn malformed_native_shot_time_and_stream_payload_are_rejected() {
    for invalid in ["time", "kind"] {
        let mut document = fixture();
        let exact = candidates(document.clone());
        let mut shot = event(
            "native-shot",
            "timeline",
            json!({"frame": 5, "time": TIMES[5], "kind": "Shot", "is_team_0": false}),
        );
        if invalid == "time" {
            shot["payload"]["payload"]["time"] = json!(TIMES[5] + 0.1);
        } else {
            shot["payload"]["kind"] = json!("touch");
        }
        events(&mut document).push(shot);
        let error =
            overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
                .unwrap_err();
        assert!(
            error.contains(if invalid == "time" {
                "native shot time disagrees"
            } else {
                "inconsistent stream and payload kind"
            }),
            "{error}"
        );
    }
}

fn boost_pickup(id: &str, frame: usize, pad_type: Option<&str>) -> Value {
    let mut pickup = event(
        id,
        "boost_pickups",
        json!({
            "frame": frame, "time": TIMES[frame], "player_id": {"Epic": "blue"},
            "collected_amount": 0.25
        }),
    );
    if let Some(pad_type) = pad_type {
        pickup["payload"]["payload"]["pad_type"] = json!(pad_type);
    }
    pickup["payload"]["kind"] = json!("boost_pickup");
    pickup
}

fn fixed_pad_fixture() -> Value {
    let mut document = fixture();
    for frame in document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"]
        .as_array_mut()
        .unwrap()
    {
        frame["Data"]["rigid_body"]["location"]["x"] = json!(0.0);
    }
    for frame in document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"]
        .as_array_mut()
        .unwrap()
    {
        frame["Data"]["rigid_body"]["location"]["x"] = json!(0.0);
    }
    document["outputs"]["replay_data"]["boost_pads"] = json!([
        {"index": 7, "pad_id": "fixed-big", "size": "Big", "position": {"x": 0.0, "y": 25.0, "z": 70.0}},
        {"index": 8, "pad_id": "later-nearer-big", "size": "Big", "position": {"x": 0.0, "y": 45.0, "z": 70.0}},
        {"index": 0, "pad_id": "small", "size": "Small", "position": {"x": 0.0, "y": 10.0, "z": 70.0}}
    ]);
    document
}

#[test]
fn explicit_big_pickups_count_at_both_approach_boundaries_even_with_small_gain() {
    let mut document = fixture();
    let exact = candidates(document.clone());
    for (id, frame, pad_type) in [
        ("before", 0, "big"),
        ("at-start", 1, "big"),
        ("small", 2, "small"),
        ("at-end", 4, "big"),
        ("after", 5, "big"),
    ] {
        events(&mut document).push(boost_pickup(id, frame, Some(pad_type)));
    }
    let batch =
        overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
            .unwrap();
    assert_eq!(batch.feature_definitions.len(), 88);
    let row = &batch.rows[0];
    assert_eq!(
        row.features["subject_big_boost_pickup_count"],
        Some(Measurement::Count(2))
    );
    assert_eq!((row.frame, row.end_frame, row.passage_frame), (1, 4, 4));
    assert_eq!(row.source_event_ids, exact.candidates[0].source_event_ids);
}

#[test]
fn ambiguous_pickup_type_is_unknown_and_observed_small_or_absent_pickups_are_zero() {
    for pad_type in [None, Some("small"), Some("ambiguous")] {
        let mut document = fixture();
        if let Some(pad_type) = pad_type {
            events(&mut document).push(boost_pickup("pickup", 2, Some(pad_type)));
        }
        let batch = extracted(document);
        assert_eq!(
            batch.rows[0].features["subject_big_boost_pickup_count"],
            if pad_type == Some("ambiguous") {
                None
            } else {
                Some(Measurement::Count(0))
            }
        );
    }
}

#[test]
fn missing_or_unsupported_pickup_type_and_incorrect_native_time_are_errors() {
    for invalid in ["missing", "unsupported", "time"] {
        let mut document = fixture();
        let exact = candidates(document.clone());
        let mut pickup = boost_pickup(
            "pickup",
            2,
            if invalid == "missing" {
                None
            } else {
                Some(if invalid == "unsupported" {
                    "large"
                } else {
                    "big"
                })
            },
        );
        if invalid == "time" {
            pickup["payload"]["payload"]["time"] = json!(TIMES[2] + 0.1);
        }
        events(&mut document).push(pickup);
        let error =
            overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
                .unwrap_err();
        assert!(
            error.contains(match invalid {
                "missing" => "missing pad_type",
                "unsupported" => "unsupported native boost pad type",
                _ => "native boost pickup time disagrees",
            }),
            "{error}"
        );
    }
}

#[test]
fn fixed_nearest_big_pad_direction_uses_irregular_native_time_weights() {
    let batch = extracted(fixed_pad_fixture());
    let row = &batch.rows[0];
    assert_eq!(row.samples.nearest_big_pad_index, Some(7));
    assert_eq!(row.samples.nearest_big_pad_id.as_deref(), Some("fixed-big"));
    assert_eq!(
        number(row, "subject_nearest_big_pad_closing_advantage_mean"),
        -24.0
    );
}

#[test]
fn equidistant_big_pad_target_uses_native_index_independent_of_metadata_order() {
    for reverse in [false, true] {
        let mut document = fixed_pad_fixture();
        let pads = &mut document["outputs"]["replay_data"]["boost_pads"];
        pads[0]["position"] = json!({"x": 10.0, "y": 10.0, "z": 70.0});
        pads[0]["index"] = json!(9);
        pads[1]["position"] = json!({"x": -10.0, "y": 10.0, "z": 70.0});
        pads[1]["index"] = json!(2);
        if reverse {
            pads.as_array_mut().unwrap().reverse();
        }
        let batch = extracted(document);
        assert_eq!(batch.rows[0].samples.nearest_big_pad_index, Some(2));
        assert_eq!(
            batch.rows[0].samples.nearest_big_pad_id.as_deref(),
            Some("later-nearer-big")
        );
    }
}

#[test]
fn incomplete_target_metadata_or_native_velocity_does_not_select_an_alternative() {
    for missing in ["size", "index", "velocity", "zero-direction"] {
        let mut document = fixed_pad_fixture();
        let exact = candidates(document.clone());
        match missing {
            "size" => document["outputs"]["replay_data"]["boost_pads"][2]["size"] = Value::Null,
            "index" => document["outputs"]["replay_data"]["boost_pads"][0]["index"] = Value::Null,
            "velocity" => {
                document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][2]["Data"]
                    ["rigid_body"]["linear_velocity"] = Value::Null
            }
            _ => document["outputs"]["replay_data"]["boost_pads"][0]["position"]["y"] = json!(20.0),
        }
        let batch =
            overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
                .unwrap();
        let row = &batch.rows[0];
        assert_eq!(
            row.features["subject_nearest_big_pad_closing_advantage_mean"], None,
            "{missing}"
        );
        assert_eq!(
            row.samples.nearest_big_pad_index,
            if missing == "size" || missing == "index" {
                None
            } else {
                Some(7)
            },
            "{missing}"
        );
    }
    let batch = extracted(fixture());
    assert_eq!(
        batch.rows[0].features["subject_nearest_big_pad_closing_advantage_mean"],
        None
    );
    assert_eq!(batch.rows[0].samples.nearest_big_pad_index, None);
}

fn with_previous_subject_touch() -> (Value, CandidateBatch) {
    let mut document = fixture();
    events(&mut document)[3] = touch(0, true, None);
    events(&mut document).push(touch(1, false, None));
    let exact = candidates(document.clone());
    (document, exact)
}

fn set_previous_possession(document: &mut Value, state: &str, owner: Option<Value>) {
    events(document).retain(|event| event["meta"]["stream"] != "possession");
    let mut payload =
        json!({"time": -1.0, "end_time": 0.0, "active": true, "possession_state": state});
    if let Some(owner) = owner {
        payload["player_id"] = owner;
    }
    events(document).push(event("possession-before-T", "possession", payload));
    events(document).push(event("possession-at-T", "possession", json!({"time": 0.0, "end_time": 7.0, "active": true, "possession_state": "neutral", "player_id": null})));
}

#[test]
fn previous_touch_possession_reads_the_span_before_t_and_preserves_unknown_identity() {
    for case in [
        "self",
        "teammate",
        "opponent",
        "neutral",
        "missing-owner",
        "null-owner",
        "opponent-no-owner",
        "missing-span",
    ] {
        let (mut document, exact) = with_previous_subject_touch();
        let (state, owner, expected) = match case {
            "self" => ("team_zero", Some(json!({"Epic": "blue"})), Some(true)),
            "teammate" => {
                let mate = json!({"Epic": "mate"});
                document["outputs"]["events"]["replay_meta"]["team_zero"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"remote_id": mate}));
                document["outputs"]["replay_data"]["frame_data"]["players"].as_array_mut().unwrap().push(json!([mate, {"frames": (0..9).map(|_| body(0.0, 50.0, 20.0, [0.0, 0.0, 0.0])).collect::<Vec<_>>() }]));
                events(&mut document).push(event(
                    "mate:activity",
                    "player_activity",
                    json!({"player": mate, "time": -1.0, "end_time": 7.0, "state": "tracked"}),
                ));
                ("team_zero", Some(mate), Some(false))
            }
            "opponent" => ("team_one", Some(json!({"Epic": "orange"})), Some(false)),
            "neutral" => ("neutral", Some(Value::Null), Some(false)),
            "null-owner" => ("team_zero", Some(Value::Null), None),
            "opponent-no-owner" => ("team_one", None, Some(false)),
            _ => ("team_zero", None, None),
        };
        set_previous_possession(&mut document, state, owner);
        if case == "missing-span" {
            events(&mut document).retain(|event| event["meta"]["id"] != "possession-before-T");
        }
        let batch =
            overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
                .unwrap();
        let row = &batch.rows[0];
        assert_eq!(
            row.features["previous_subject_touch_possession_before_self"],
            expected.map(Measurement::Boolean),
            "{case}"
        );
        assert_eq!(
            row.samples.previous_subject_touch.as_ref().unwrap().frame,
            0
        );
        assert_eq!(
            row.samples
                .previous_subject_touch_possession
                .as_ref()
                .map(|stage| stage.source_event_id.as_str()),
            if case == "missing-span" {
                None
            } else {
                Some("possession-before-T")
            },
            "{case}"
        );
    }
}

#[test]
fn previous_possession_rejects_invalid_state_or_owner_team_without_using_touch_identity() {
    for (state, owner) in [
        ("team_zero", json!({"Epic": "orange"})),
        ("team_one", json!({"Epic": "blue"})),
        ("unsupported", Value::Null),
    ] {
        let (mut document, exact) = with_previous_subject_touch();
        set_previous_possession(&mut document, state, Some(owner));
        let error =
            overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
                .unwrap_err();
        assert!(
            error.contains(if state == "unsupported" {
                "unsupported native possession state"
            } else {
                "disagrees with owner team"
            }),
            "{error}"
        );
    }
}

#[test]
fn previous_touch_following_contact_retains_simultaneous_actors_and_excludes_same_frame() {
    for simultaneous in [false, true] {
        let (mut document, exact) = with_previous_subject_touch();
        events(&mut document).retain(|event| event["meta"]["id"] != "touch:1");
        let mut own = touch(1, true, None);
        own["meta"]["id"] = json!("next-own");
        events(&mut document).push(own);
        let mut same_frame_opponent = touch(0, false, None);
        same_frame_opponent["meta"]["id"] = json!("simultaneous-previous-opponent");
        events(&mut document).push(same_frame_opponent);
        if simultaneous {
            let mut opponent = touch(1, false, None);
            opponent["meta"]["id"] = json!("next-opponent");
            events(&mut document).push(opponent);
        }
        let mut exact = exact;
        for id in &mut exact.candidates[0].source_event_ids {
            if *id == "touch:1" {
                *id = "next-own".into();
            }
        }
        let batch =
            overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
                .unwrap();
        let row = &batch.rows[0];
        assert_eq!(
            row.features["previous_subject_touch_followed_by_opponent"],
            Some(Measurement::Boolean(simultaneous))
        );
        let next = row
            .samples
            .previous_subject_touch_following_contact
            .as_ref()
            .unwrap();
        assert_eq!(next.frame, 1);
        assert_eq!(
            next.source_event_ids.len(),
            if simultaneous { 2 } else { 1 }
        );
        assert!(
            next.source_event_ids
                .iter()
                .all(|id| id.starts_with("next-"))
        );
    }
}

#[test]
fn previous_touch_stages_are_missing_without_t_or_a_following_contact_inside_saved_play() {
    let ordinary = extracted(fixture());
    assert_eq!(
        ordinary.rows[0].features["previous_subject_touch_possession_before_self"],
        None
    );
    assert_eq!(
        ordinary.rows[0].features["previous_subject_touch_followed_by_opponent"],
        None
    );
    let (mut document, mut exact) = with_previous_subject_touch();
    events(&mut document)
        .retain(|event| !matches!(event["meta"]["id"].as_str(), Some("touch:1" | "touch:2")));
    exact.candidates[0].source_event_ids = vec!["touch:0".into()];
    exact.candidates[0].evidence["play_end_frame"] = json!(4);
    let batch =
        overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
            .unwrap();
    assert_eq!(
        batch.rows[0].features["previous_subject_touch_followed_by_opponent"],
        None
    );
    assert!(
        batch.rows[0]
            .samples
            .previous_subject_touch_following_contact
            .is_none()
    );
    exact.candidates[0].evidence["play_start_frame"] = json!(1);
    let batch = overcommitting_features::extract_for_candidates(
        &analyze(with_previous_subject_touch().0),
        sampling(),
        &exact,
    )
    .unwrap();
    assert!(batch.rows[0].samples.previous_subject_touch.is_none());
}

fn withdrawal_fixture(contact: Option<usize>, end: usize) -> (Value, CandidateBatch) {
    let mut document = fixture();
    events(&mut document).retain(|event| event["meta"]["id"] != "touch:2");
    if let Some(frame) = contact {
        events(&mut document).push(touch(frame, false, None));
    } else {
        events(&mut document).push(event(
            "shot:3",
            "timeline",
            json!({"frame": 3, "time": TIMES[3], "kind": "Shot", "is_team_0": false}),
        ));
    }
    let mut exact = candidates(document.clone());
    assert_eq!(exact.candidates[0].frame, 1);
    if end != 4 {
        exact.candidates[0].evidence["end_frame"] = json!(end);
        exact.candidates[0].evidence["end_time"] = json!(TIMES[end]);
        exact.candidates[0].evidence["finish"] = json!("retreat_to_opening_depth");
    }
    let x = [0.0, 0.0, -1.0, -3.0, -12.0, -12.0, -12.0, -12.0, -12.0];
    for (frame, position_x) in x.into_iter().enumerate() {
        document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][frame] =
            body(position_x, 10.0, 2.0, [-(frame as f64) * 2.0, 0.0, 0.0]);
        document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][frame] =
            body(position_x + 10.0, 10.0, 7.0, [0.0, 0.0, 0.0]);
    }
    (document, exact)
}

#[test]
fn ball_relative_sideways_withdrawal_integrates_left_frames_and_irregular_time_steps() {
    let (document, exact) = withdrawal_fixture(Some(2), 4);
    let batch =
        overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
            .unwrap();
    let row = &batch.rows[0];
    assert_eq!(
        number(
            row,
            "subject_ball_withdrawal_distance_before_opponent_touch"
        ),
        1.0
    );
    assert_eq!(
        number(row, "subject_ball_withdrawal_distance_after_opponent_touch"),
        11.0
    );
    assert_eq!(number(row, "subject_retreat_distance"), 0.0);
}

#[test]
fn withdrawal_windows_handle_contact_at_start_end_after_approach_and_missing_contact() {
    for (contact, end, before, after) in [
        (Some(1), 4, Some(0.0), Some(12.0)),
        (Some(4), 4, Some(12.0), Some(0.0)),
        (Some(3), 2, Some(1.0), Some(0.0)),
        (None, 4, None, None),
    ] {
        let (document, exact) = withdrawal_fixture(contact, end);
        let batch =
            overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
                .unwrap();
        let row = &batch.rows[0];
        assert_eq!(
            row.features["subject_ball_withdrawal_distance_before_opponent_touch"],
            before.map(Measurement::Float),
            "{contact:?}/{end}"
        );
        assert_eq!(
            row.features["subject_ball_withdrawal_distance_after_opponent_touch"],
            after.map(Measurement::Float),
            "{contact:?}/{end}"
        );
    }
}

#[test]
fn withdrawal_missing_data_affects_only_nonempty_native_windows() {
    for missing in [
        "before-velocity",
        "after-geometry",
        "undefined-direction",
        "after-empty",
    ] {
        let (mut document, exact) = withdrawal_fixture(
            Some(if missing == "after-empty" { 3 } else { 2 }),
            if missing == "after-empty" { 2 } else { 4 },
        );
        match missing {
            "before-velocity" => {
                document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][1]["Data"]
                    ["rigid_body"]["linear_velocity"] = Value::Null
            }
            "after-geometry" => {
                document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][2] =
                    json!("Empty")
            }
            "undefined-direction" => {
                document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][2]["Data"]
                    ["rigid_body"]["location"]["x"] = json!(-1.0)
            }
            _ => {
                document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][3]["Data"]
                    ["rigid_body"]["linear_velocity"] = Value::Null
            }
        }
        let batch =
            overcommitting_features::extract_for_candidates(&analyze(document), sampling(), &exact)
                .unwrap();
        let row = &batch.rows[0];
        assert_eq!(
            row.features["subject_ball_withdrawal_distance_before_opponent_touch"],
            if missing == "before-velocity" {
                None
            } else {
                Some(Measurement::Float(1.0))
            },
            "{missing}"
        );
        assert_eq!(
            row.features["subject_ball_withdrawal_distance_after_opponent_touch"],
            match missing {
                "before-velocity" => Some(Measurement::Float(11.0)),
                "after-empty" => Some(Measurement::Float(0.0)),
                _ => None,
            },
            "{missing}"
        );
    }
}
