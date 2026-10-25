use rl_mistake_analysis_candidates::bang_features::{
    FeatureBatch, FeatureSampling, ReplayAnalysis, SamplingMethod, extract,
};
use serde_json::{Value, json};
use subtr_actor::{BALL_COLLISION_RADIUS, STANDARD_GOAL_LINE_Y, STANDARD_GOAL_MOUTH_HALF_WIDTH_X};

#[test]
fn consecutive_subject_touches_stop_at_other_players_and_play_boundaries() {
    let mut document = fixture();
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events[0] = touch(0, 0, 1, true, json!([]));
    events.push(touch(20, 1, 1, true, json!([])));
    let features = row(document.clone())["features"].clone();
    assert_eq!(features["subject_consecutive_touches_before_candidate"], 2);
    assert!(features["previous_opponent_touch_delay_s"].is_null());

    document["outputs"]["events"]["events"]["events"][0] = touch(0, 0, 3, false, json!([]));
    let features = row(document.clone())["features"].clone();
    assert_eq!(features["subject_consecutive_touches_before_candidate"], 1);
    assert_eq!(features["previous_opponent_touch_delay_s"], 1.);

    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events.push(touch(21, 1, 2, true, json!([])));
    let features = row(document.clone())["features"].clone();
    assert_eq!(features["subject_consecutive_touches_before_candidate"], 0);
    assert_eq!(features["previous_opponent_touch_delay_s"], 1.);

    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .push(event(
            "kickoff:0",
            "kickoff",
            json!({"start_time": 0., "live_action_start_time": 0.75}),
        ));
    let features = row(document)["features"].clone();
    assert_eq!(features["subject_consecutive_touches_before_candidate"], 0);
    assert!(features["previous_opponent_touch_delay_s"].is_null());
}

#[test]
fn opponent_speed_and_subject_height_use_the_candidate_contact_frame() {
    let mut document = fixture();
    let frames = &mut document["outputs"]["replay_data"]["frame_data"];
    frames["players"][2][1]["frames"][2]["Data"]["rigid_body"]["linear_velocity"] =
        json!({"x": 0., "y": 0., "z": 40.});
    frames["players"][0][1]["frames"][2]["Data"]["rigid_body"]["location"]["z"] = json!(333.);
    frames["players"][2][1]["frames"][5]["Data"]["rigid_body"]["rotation"] =
        json!({"x": 2., "y": 0., "z": 0., "w": 2.});
    let row = row(document);
    let f = &row["features"];
    assert_eq!(f["closest_opponent_speed_at_candidate"], 40.);
    assert_eq!(f["subject_height_at_candidate"], 333.);
    assert_eq!(f["closest_opponent_velocity_relative_ball_up"], 0.);
    assert_eq!(f["first_touch_opponent_heading_side"], -1.);
    assert_eq!(f["first_touch_opponent_heading_up"], 0.);
    assert_eq!(f["first_touch_opponent_up_toward"], -1.);
    assert_eq!(f["first_touch_opponent_up_up"], 0.);
    assert_eq!(
        row["samples"]["closest_opponent_at_candidate_id"],
        json!({"Steam": 3})
    );
    assert_eq!(
        row["samples"]["first_touch_opponent_id"],
        json!({"Steam": 3})
    );
}

#[test]
fn receiving_opponent_orientation_requires_one_toucher_and_its_contact_body() {
    let mut document = fixture();
    document["outputs"]["replay_data"]["frame_data"]["players"][2][1]["frames"][5] = json!("Empty");
    let features = row(document)["features"].clone();
    for direction in ["heading", "up"] {
        for axis in ["toward", "side", "up"] {
            assert!(features[format!("first_touch_opponent_{direction}_{axis}")].is_null());
        }
    }

    let mut document = fixture();
    add_opponent(&mut document, 5, body(1000., 2000., 1000.), 255.);
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .push(touch(22, 5, 5, false, json!([])));
    let row = row(document);
    assert!(row["samples"]["first_touch_opponent_id"].is_null());
    assert!(row["features"]["first_touch_opponent_heading_side"].is_null());
    assert!(row["features"]["first_touch_opponent_up_up"].is_null());
    assert_eq!(row["features"]["closest_opponent_speed_at_candidate"], 13.);
}

#[test]
fn full_opponent_possession_includes_earlier_time_and_stops_at_play_end() {
    let mut document = fixture();
    let events = &mut document["outputs"]["events"]["events"]["events"];
    events[5]["payload"]["payload"]["end_time"] = json!(0.5);
    events[6]["payload"]["payload"]["time"] = json!(0.5);
    let f = row(document)["features"].clone();
    assert_eq!(f["next_opponent_possession_duration_s"], 4.5);
    assert_eq!(f["exchange_possession_opponent_s"], 2.5);

    let f = row(fixture())["features"].clone();
    assert_eq!(f["next_opponent_possession_duration_s"], 3.);
}

#[test]
fn following_touch_possession_preserves_observed_negatives_and_unknown_gaps() {
    let mut document = fixture();
    let events = &mut document["outputs"]["events"]["events"]["events"];
    events[5]["payload"]["payload"]["end_time"] = json!(3.);
    events[6]["payload"]["payload"]["time"] = json!(3.);
    assert_eq!(
        row(document)["features"]["next_opponent_possession_duration_s"],
        0.
    );

    let mut document = fixture();
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .remove(6);
    assert!(row(document)["features"]["next_opponent_possession_duration_s"].is_null());

    let mut document = fixture();
    let events = &mut document["outputs"]["events"]["events"]["events"];
    events[5]["payload"]["payload"]["end_time"] = json!(2.5);
    events[6]["payload"]["payload"]["time"] = json!(2.5);
    assert_eq!(
        row(document)["features"]["next_opponent_possession_duration_s"],
        2.5
    );
}

#[test]
fn native_thirds_shadow_defense_and_wall_tags_are_observed_at_the_contact() {
    let mut document = fixture();
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events[1]["payload"]["payload"]["tags"]
        .as_array_mut()
        .unwrap()
        .push(json!({"group": "surface", "value": "wall"}));
    events.push(event(
        "third:1",
        "field_third",
        json!({"player": {"Steam": 1}, "time": 0., "end_time": 6.5, "state": "offensive"}),
    ));
    events.push(event(
        "shadow:1",
        "shadow_defense",
        json!({"player": {"Steam": 1}, "time": 0.5, "end_time": 1.5, "state": "shadowing"}),
    ));
    let f = row(document.clone())["features"].clone();
    assert_eq!(f["subject_field_third_at_candidate_offensive"], true);
    assert_eq!(f["subject_field_third_at_candidate_defensive"], false);
    assert_eq!(f["subject_shadow_defense_at_candidate"], true);
    assert_eq!(f["own_team_shadow_defense_at_candidate"], true);
    assert_eq!(f["candidate_surface_wall"], true);
    assert_eq!(f["candidate_surface_air"], false);

    document["outputs"]["replay_data"]["frame_data"]["players"][1][1]["frames"][2] = json!("Empty");
    let f = row(document)["features"].clone();
    assert_eq!(f["subject_shadow_defense_at_candidate"], true);
    assert!(f["own_team_shadow_defense_at_candidate"].is_null());

    let f = row(fixture())["features"].clone();
    assert!(f["candidate_surface_wall"].is_null());
    assert!(f["subject_field_third_at_candidate_offensive"].is_null());
    assert_eq!(f["subject_shadow_defense_at_candidate"], false);
}

fn event(id: &str, stream: &str, payload: Value) -> Value {
    let kind = if stream == "air_dribble" {
        "ball_carry"
    } else {
        stream
    };
    json!({"meta": {"id": id, "stream": stream}, "payload": {"kind": kind, "payload": payload}})
}

fn body(x: f64, y: f64, z: f64) -> Value {
    json!({"location": {"x": x, "y": y, "z": z}, "rotation": {"x": 0, "y": 0, "z": 0, "w": 2}, "linear_velocity": {"x": 3, "y": 4, "z": 12}})
}

fn touch(id: u64, frame: usize, player: u64, team: bool, tags: Value) -> Value {
    event(
        &format!("touch:{id}"),
        "touch",
        json!({"touch_id": id, "frame": frame,
        "time": frame as f64 / 2., "sample_frame": frame, "sample_time": frame as f64 / 2.,
        "player": {"Steam": player}, "is_team_0": team, "tags": tags}),
    )
}

fn activity(player: u64, team: bool, state: &str, start: usize, end: usize) -> Value {
    event(
        &format!("activity:{player}:{start}"),
        "player_activity",
        json!({"player": {"Steam": player}, "is_team_0": team, "state": state,
            "frame": start, "end_frame": end, "time": start as f64 / 2., "end_time": end as f64 / 2.}),
    )
}

fn fixture() -> Value {
    let players = (1..=4)
        .map(|id| {
            let frames = (0..14)
                .map(|frame| {
                    let rigid_body = match id {
                        1 => body(frame as f64 * 2., frame as f64 * 3., 10.),
                        2 => body(0., 20., 201.),
                        3 => body(8., 20., 111.),
                        _ => body(-5., 20., 101.),
                    };
                    json!({"Data": {"rigid_body": rigid_body, "dodge_active": false, "boost_amount": id as f64 * 51.}})
                })
                .collect::<Vec<_>>();
            json!([{"Steam": id}, {"frames": frames}])
        })
        .collect::<Vec<_>>();
    let ball = (0..14).map(|frame| json!({"Data": {"rigid_body": body(frame as f64 * 10., frame as f64 * 20., 100. + frame as f64)}})).collect::<Vec<_>>();
    let mut candidate = touch(
        1,
        2,
        1,
        true,
        json!([{"group": "action", "value": "clear"}, {"group": "dodge_state", "value": "dodge"}]),
    );
    candidate["payload"]["payload"]["sample_frame"] = json!(3);
    candidate["payload"]["payload"]["sample_time"] = json!(1.5);
    let mut events = vec![
        touch(0, 0, 2, true, json!([])),
        candidate,
        touch(
            2,
            12,
            2,
            true,
            json!([{"group": "action", "value": "pass"}]),
        ),
        touch(
            3,
            5,
            3,
            false,
            json!([{"group": "action", "value": "shot"}, {"group": "possession", "value": "control"}]),
        ),
        touch(
            4,
            7,
            1,
            true,
            json!([{"group": "dodge_state", "value": "no_dodge"}]),
        ),
        event(
            "possession:0",
            "possession",
            json!({"frame": 0, "end_frame": 4, "time": 0., "end_time": 2., "active": true, "possession_state": "team_zero"}),
        ),
        event(
            "possession:1",
            "possession",
            json!({"frame": 4, "end_frame": 13, "time": 2., "end_time": 6.5, "active": true, "possession_state": "team_one"}),
        ),
        event(
            "goal:10",
            "goal_context",
            json!({"frame": 10, "time": 5., "scoring_team_is_team_0": false}),
        ),
        event(
            "beaten:2",
            "beaten_to_ball",
            json!({"frame": 2, "time": 1., "player": {"Steam": 3}, "winner": {"Steam": 1}, "is_team_0": false}),
        ),
        event(
            "beaten:6",
            "beaten_to_ball",
            json!({"frame": 6, "time": 3., "player": {"Steam": 1}, "winner": {"Steam": 3}, "is_team_0": true}),
        ),
        event(
            "whiff:4",
            "whiff",
            json!({"frame": 4, "time": 2., "resolved_time": 3., "kind": "whiff", "player": {"Steam": 3}, "is_team_0": false}),
        ),
        event(
            "carry:0",
            "ball_carry",
            json!({"player_id": {"Steam": 1}, "start_time": 0.5, "end_time": 1.}),
        ),
        event(
            "pass:0",
            "pass",
            json!({"receiver": {"Steam": 1}, "start_time": 0.5, "time": 1.}),
        ),
    ];
    for id in 1..=4 {
        events.push(activity(id, id != 3, "tracked", 0, 13));
    }
    json!({"name": "subtr-actor", "schema_version": 1, "status": "ok", "revision": "fixture", "replay_sha256": "0".repeat(64), "outputs": {
        "events": {"replay_meta": {"team_zero": [{"remote_id": {"Steam": 1}}, {"remote_id": {"Steam": 2}}, {"remote_id": {"Steam": 4}}], "team_one": [{"remote_id": {"Steam": 3}}]}, "events": {"events": events}, "accumulation_tracks": (1..=4).map(|id| json!({"player_id": {"Steam": id}, "quantity": "boost_amount", "points": [{"frame": 0, "value": id as f64 * 51.}]})).collect::<Vec<_>>()},
        "replay_data": {"frame_data": {"players": players, "ball_data": {"frames": ball}, "metadata_frames": (0..14).map(|frame| json!({"time": frame as f64 / 2.})).collect::<Vec<_>>()}, "boost_pad_events": [], "boost_pads": []}
    }})
}

fn add_opponent(document: &mut Value, id: u64, rigid_body: Value, boost: f64) {
    let players = document["outputs"]["replay_data"]["frame_data"]["players"]
        .as_array_mut()
        .unwrap();
    let mut player = players[2].clone();
    player[0] = json!({"Steam": id});
    for frame in player[1]["frames"].as_array_mut().unwrap() {
        frame["Data"]["rigid_body"] = rigid_body.clone();
        frame["Data"]["boost_amount"] = json!(boost);
    }
    players.push(player);
    document["outputs"]["events"]["accumulation_tracks"].as_array_mut().unwrap().push(json!({"player_id": {"Steam": id}, "quantity": "boost_amount", "points": [{"frame": 0, "value": boost}]}));
    document["outputs"]["events"]["replay_meta"]["team_one"]
        .as_array_mut()
        .unwrap()
        .push(json!({"remote_id": {"Steam": id}}));
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .push(activity(id, false, "tracked", 0, 13));
}

fn batch(document: Value) -> FeatureBatch {
    let analysis: ReplayAnalysis = serde_json::from_value(document).unwrap();
    let (candidates, features) = extract(
        &analysis,
        FeatureSampling {
            method: SamplingMethod::NativeTouchStages,
        },
        Some(&json!({"Steam": 1})),
    )
    .unwrap();
    assert_eq!(candidates.candidates.len(), features.rows.len());
    for (candidate, row) in candidates.candidates.iter().zip(&features.rows) {
        assert_eq!(
            candidate.source_event_ids.as_slice(),
            std::slice::from_ref(&row.source_event_id)
        );
        assert_eq!(candidate.player_id, row.player_id);
        assert_eq!(candidate.frame, row.frame as u64);
    }
    features
}

fn row(document: Value) -> Value {
    serde_json::to_value(batch(document)).unwrap()["rows"][0].clone()
}

#[test]
fn exchange_extrema_and_team_distances_include_every_native_frame() {
    let mut document = fixture();
    let frames = &mut document["outputs"]["replay_data"]["frame_data"];
    for (frame, position) in [(4, [-500., 1234., 999.]), (6, [700., -222., 15.])] {
        frames["ball_data"]["frames"][frame]["Data"]["rigid_body"] =
            body(position[0], position[1], position[2]);
    }
    frames["ball_data"]["frames"][8]["Data"]["rigid_body"] = body(-9000., 9000., 2000.);
    frames["players"][3][1]["frames"][4]["Data"]["rigid_body"] = body(-497., 1234., 999.);
    frames["players"][0][1]["frames"][6]["Data"]["rigid_body"] = body(700., -222., 19.);
    frames["players"][2][1]["frames"][3]["Data"]["rigid_body"] = body(35., 60., 103.);
    frames["players"][2][1]["frames"][6]["Data"]["rigid_body"] = body(700., -220., 15.);
    let features = row(document)["features"].clone();
    for (name, expected) in [
        ("play_elapsed_s", 1.),
        ("exchange_ball_position_min_toward", -222.),
        ("exchange_ball_position_max_toward", 1234.),
        ("exchange_ball_position_min_side", -700.),
        ("exchange_ball_position_max_side", 500.),
        ("exchange_ball_position_min_up", 15.),
        ("exchange_ball_position_max_up", 999.),
        ("own_team_ball_distance_candidate_to_first_touch", 3.),
        ("own_team_ball_distance_first_to_second_touch", 4.),
        ("opponent_ball_distance_candidate_to_first_touch", 5.),
        ("opponent_ball_distance_first_to_second_touch", 2.),
        ("own_team_ball_distance_at_first_touch", 9441_f64.sqrt()),
        ("own_team_ball_distance_at_second_touch", 20061_f64.sqrt()),
        ("opponent_ball_distance_at_first_touch", 8200_f64.sqrt()),
        ("opponent_ball_distance_at_second_touch", 18260_f64.sqrt()),
    ] {
        assert!(
            (features[name].as_f64().unwrap() - expected).abs() < 0.0001,
            "{name}"
        );
    }
}

#[test]
fn team_distance_mean_weights_each_native_interval_and_includes_the_hitter() {
    let mut document = fixture();
    let frames = &mut document["outputs"]["replay_data"]["frame_data"];
    for (offset, time) in [1.0, 1.25, 2.25, 2.5].into_iter().enumerate() {
        let frame = offset + 2;
        frames["metadata_frames"][frame]["time"] = json!(time);
        frames["ball_data"]["frames"][frame]["Data"]["rigid_body"] = body(0., 0., 0.);
        for (player, distances) in [
            [10., 30., 100., 20.],
            [20., 20., 30., 200.],
            [40., 50., 60., 70.],
            [500., 500., 500., 500.],
        ]
        .into_iter()
        .enumerate()
        {
            frames["players"][player][1]["frames"][frame]["Data"]["rigid_body"] =
                body(distances[offset], 0., 0.);
        }
    }
    document["outputs"]["events"]["events"]["events"][1]["payload"]["payload"]["sample_time"] =
        json!(1.25);
    let features = row(document)["features"].clone();
    assert_eq!(
        features["own_team_ball_distance_candidate_to_first_touch"],
        10.
    );
    assert!(
        (features["own_team_ball_distance_mean_candidate_to_first_touch"]
            .as_f64()
            .unwrap()
            - 35. / 1.5)
            .abs()
            < f32::EPSILON as f64 * 35.
    );
    assert_eq!(
        features["opponent_ball_distance_mean_candidate_to_first_touch"],
        55.
    );
}

#[test]
fn missing_interval_observations_are_not_replaced_by_observed_minima() {
    let mut document = fixture();
    document["outputs"]["replay_data"]["frame_data"]["players"][3][1]["frames"][4] = json!("Empty");
    document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][6] = json!("Empty");
    let features = row(document)["features"].clone();
    for name in [
        "own_team_ball_distance_candidate_to_first_touch",
        "own_team_ball_distance_first_to_second_touch",
        "opponent_ball_distance_first_to_second_touch",
        "own_team_ball_distance_mean_candidate_to_first_touch",
        "own_team_ball_distance_mean_first_to_second_touch",
        "opponent_ball_distance_mean_first_to_second_touch",
        "ball_own_goal_distance_first_to_second_touch",
        "ball_opponent_goal_distance_first_to_second_touch",
        "exchange_ball_position_min_toward",
        "exchange_ball_position_max_up",
    ] {
        assert!(features[name].is_null(), "{name}");
    }
    assert!(features["opponent_ball_distance_candidate_to_first_touch"].is_number());
    assert!(features["own_team_ball_distance_at_first_touch"].is_number());
    assert!(features["ball_opponent_goal_distance_at_second_touch"].is_number());
}

#[test]
fn team_minima_exclude_demolished_cars_without_treating_their_absence_as_missing_data() {
    let mut document = fixture();
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events.retain(|event| {
        event["meta"]["stream"] != "player_activity"
            || event["payload"]["payload"]["player"]["Steam"] != 3
    });
    events.extend([
        activity(3, false, "demolished", 0, 4),
        activity(3, false, "tracked", 4, 13),
    ]);
    document["outputs"]["replay_data"]["frame_data"]["players"][2][1]["frames"][3]["Data"]["rigid_body"] =
        body(30., 60., 103.);
    let features = row(document)["features"].clone();
    assert!(
        (features["opponent_ball_distance_candidate_to_first_touch"]
            .as_f64()
            .unwrap()
            - 8200_f64.sqrt())
        .abs()
            < 0.0001
    );
    assert!(features["opponent_ball_distance_mean_candidate_to_first_touch"].is_null());
    assert!(features["opponent_ball_distance_mean_first_to_second_touch"].is_number());
}

#[test]
fn zero_duration_intervals_have_no_time_average() {
    let mut document = fixture();
    for frame in 2..=5 {
        document["outputs"]["replay_data"]["frame_data"]["metadata_frames"][frame]["time"] =
            json!(1.);
    }
    for event in document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
    {
        if event["meta"]["stream"] == "touch" {
            let payload = &mut event["payload"]["payload"];
            for (frame_field, time_field) in [("frame", "time"), ("sample_frame", "sample_time")] {
                let frame = payload[frame_field].as_u64().unwrap();
                if (2..=5).contains(&frame) {
                    payload[time_field] = json!(1.);
                }
            }
        }
    }
    let features = row(document)["features"].clone();
    assert!(features["own_team_ball_distance_mean_candidate_to_first_touch"].is_null());
    assert!(features["opponent_ball_distance_mean_candidate_to_first_touch"].is_null());
    assert!(features["own_team_ball_distance_candidate_to_first_touch"].is_number());
}

#[test]
fn native_bumps_and_confirmed_subject_and_teammate_whiffs_are_distinct() {
    let mut document = fixture();
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events.extend([
        event("bump:6", "bump", json!({"frame": 6, "victim": {"Steam": 1}, "initiator": {"Steam": 3}})),
        event("whiff:subject", "whiff", json!({"frame": 3, "kind": "whiff", "player": {"Steam": 1}, "is_team_0": true, "resolved_time": 3.})),
        event("whiff:teammate", "whiff", json!({"frame": 6, "kind": "whiff", "player": {"Steam": 2}, "is_team_0": true, "resolved_time": 3.5})),
    ]);
    let features = row(document.clone())["features"].clone();
    for name in [
        "subject_bumped_during_exchange",
        "any_bump_during_exchange",
        "subject_whiff_during_exchange",
        "teammate_whiff_during_exchange",
    ] {
        assert_eq!(features[name], true, "{name}");
    }
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    let count = events.len();
    events[count - 3]["payload"]["payload"]["victim"] = json!({"Steam": 2});
    events[count - 2]["payload"]["payload"]["resolved_time"] = json!(4.);
    events[count - 1]["payload"]["payload"]["frame"] = json!(8);
    let features = row(document)["features"].clone();
    assert_eq!(features["any_bump_during_exchange"], true);
    assert_eq!(features["subject_bumped_during_exchange"], false);
    assert_eq!(features["subject_whiff_during_exchange"], false);
    assert_eq!(features["teammate_whiff_during_exchange"], false);
}

#[test]
fn possession_durations_clip_native_spans_to_the_observed_exchange() {
    let mut document = fixture();
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events[6]["payload"]["payload"]["time"] = json!(3.);
    events[6]["payload"]["payload"]["frame"] = json!(6);
    events.push(event(
        "possession:neutral",
        "possession",
        json!({"time": 2., "end_time": 3., "frame": 4, "end_frame": 6, "active": true, "possession_state": "neutral"}),
    ));
    let features = row(document.clone())["features"].clone();
    assert_eq!(features["exchange_possession_own_team_s"], 1.);
    assert_eq!(features["exchange_possession_opponent_s"], 0.5);
    assert_eq!(features["exchange_possession_neutral_s"], 1.);
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events.last_mut().unwrap()["payload"]["payload"]["end_time"] = json!(2.5);
    events.last_mut().unwrap()["payload"]["payload"]["end_frame"] = json!(5);
    let features = row(document)["features"].clone();
    for team in ["own_team", "opponent", "neutral"] {
        assert!(features[format!("exchange_possession_{team}_s")].is_null());
    }
}

#[test]
fn goal_distances_allow_ball_overlap_and_direction_uses_the_native_velocity_ray() {
    let radius = f64::from(BALL_COLLISION_RADIUS);
    let goal = f64::from(STANDARD_GOAL_LINE_Y);
    let edge = f64::from(STANDARD_GOAL_MOUTH_HALF_WIDTH_X);
    let mut document = fixture();
    let frames = &mut document["outputs"]["replay_data"]["frame_data"];
    frames["ball_data"]["frames"][2]["Data"]["rigid_body"] = body(edge + radius / 2., 0., 100.);
    frames["ball_data"]["frames"][2]["Data"]["rigid_body"]["linear_velocity"] =
        json!({"x": 0., "y": 100., "z": 0.});
    frames["ball_data"]["frames"][4]["Data"]["rigid_body"] = body(0., goal - radius / 2., 100.);
    frames["ball_data"]["frames"][5]["Data"]["rigid_body"] = body(0., goal - radius * 3., 100.);
    frames["ball_data"]["frames"][7]["Data"]["rigid_body"] = body(0., -goal + radius / 2., 100.);
    let features = row(document.clone())["features"].clone();
    assert_eq!(features["candidate_outgoing_ball_velocity_toward"], 100.);
    assert_eq!(features["candidate_directed_at_opponent_goal"], true);
    assert_eq!(features["candidate_directed_at_own_goal"], false);
    assert_eq!(
        features["ball_opponent_goal_distance_candidate_to_first_touch"],
        0.
    );
    assert_eq!(
        features["ball_opponent_goal_distance_at_first_touch"],
        radius * 2.
    );
    assert_eq!(features["ball_own_goal_distance_at_second_touch"], 0.);
    document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][2]["Data"]["rigid_body"]
        ["location"]["x"] = json!(edge + radius * 2.);
    assert_eq!(
        row(document.clone())["features"]["candidate_directed_at_opponent_goal"],
        false
    );
    document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][2]["Data"]["rigid_body"]
        ["linear_velocity"] = Value::Null;
    assert!(row(document)["features"]["candidate_directed_at_opponent_goal"].is_null());
}

#[test]
fn native_stages_preserve_simultaneous_touches_and_candidate_provenance() {
    let mut document = fixture();
    add_opponent(&mut document, 5, body(10000., 20000., 10000.), 255.);
    document["outputs"]["events"]["events"]["events"][2] = touch(
        2,
        5,
        5,
        false,
        json!([{"group": "action", "value": "pass"}]),
    );
    let row = row(document);
    assert_eq!(row["touch_id"], 1);
    assert_eq!(row["samples"]["outgoing_sample_frame"], 2);
    assert_eq!(row["samples"]["first_touch"]["touch_ids"], json!([2, 3]));
    assert_eq!(row["samples"]["second_touch"]["touch_ids"], json!([4]));
    let f = &row["features"];
    for name in [
        "first_touch_opponent",
        "second_touch_self",
        "first_touch_action_pass",
        "first_touch_action_shot",
        "second_touch_action_unclassified",
        "first_touch_opponent_control",
        "candidate_action_clear",
        "candidate_dodging",
        "ground_carry_before",
        "receiving_pass_before",
        "opponent_beaten_at_candidate",
        "subject_beaten_during_exchange",
        "opponent_whiff_during_exchange",
        "next_goal_opponent",
        "next_possession_opponent",
    ] {
        assert_eq!(f[name], true, "{name}");
    }
    assert_eq!(f["first_touch_teammate"], false);
    for (name, value) in [
        ("candidate_incoming_ball_speed", 13.),
        ("candidate_outgoing_ball_speed", 13.),
        ("closest_opponent_position_relative_ball_toward", 0.),
        ("closest_opponent_position_relative_ball_side", 2.),
        ("closest_opponent_position_relative_ball_up", 10.),
        ("closest_opponent_heading_toward", 0.),
        ("closest_opponent_heading_side", -1.),
        ("closest_opponent_heading_up", 0.),
        ("closest_opponent_velocity_relative_ball_toward", 0.),
        ("closest_opponent_velocity_relative_ball_side", 0.),
        ("closest_opponent_velocity_relative_ball_up", 0.),
        ("closest_opponent_boost_before_pct", 60.),
        ("approaching_teammate_count", 1.),
        ("approaching_opponent_count", 0.),
        ("candidate_ball_position_toward", 40.),
        ("candidate_ball_position_side", -20.),
        ("candidate_ball_position_up", 102.),
        ("first_touch_ball_position_toward", 100.),
        ("second_touch_ball_position_up", 107.),
        ("previous_touch_delay_s", 1.),
        ("first_touch_delay_s", 1.5),
        ("second_touch_delay_s", 2.5),
        ("next_goal_delay_s", 4.),
        ("touches_before_next_goal", 3.),
        ("possession_change_delay_s", 1.),
        ("subject_next_touch_delay_s", 2.5),
        ("subject_displacement_to_first_touch_toward", 9.),
        ("subject_displacement_to_second_touch_side", -10.),
        ("subject_heading_at_first_touch_toward", 0.),
        ("subject_heading_at_first_touch_side", -1.),
        ("subject_heading_at_first_touch_up", 0.),
        ("subject_heading_at_second_touch_side", -1.),
    ] {
        assert_eq!(f[name].as_f64(), Some(value), "{name}");
    }
    assert_eq!(row["samples"]["closest_opponent_id"], json!({"Steam": 3}));
    assert!(
        (f["subject_ball_distance_at_first_touch"].as_f64().unwrap() - 17850_f64.sqrt()).abs()
            < 0.0001
    );
    assert!(
        (f["subject_ball_distance_at_second_touch"].as_f64().unwrap() - 26706_f64.sqrt()).abs()
            < 0.0001
    );
}

#[test]
fn intervals_use_all_touches_and_require_a_resolved_native_whiff() {
    let mut document = fixture();
    document["outputs"]["events"]["events"]["events"][9]["payload"]["payload"]["winner"] =
        json!({"Steam": 2});
    document["outputs"]["events"]["events"]["events"][10]["payload"]["payload"]["resolved_time"] =
        json!(4.);
    let row = row(document);
    assert_eq!(row["features"]["opponent_whiff_during_exchange"], false);
    assert_eq!(row["features"]["subject_beaten_during_exchange"], true);
    assert_eq!(row["samples"]["first_touch"]["frame"], 5);
}

#[test]
fn goals_and_kickoffs_bound_the_exchange_and_censor_unobserved_stages() {
    for boundary in ["goal", "kickoff"] {
        let mut document = fixture();
        let events = document["outputs"]["events"]["events"]["events"]
            .as_array_mut()
            .unwrap();
        events[4]["payload"]["payload"]["player"] = json!({"Steam": 2});
        if boundary == "goal" {
            events[7]["payload"]["payload"]["time"] = json!(3.);
            events[7]["payload"]["payload"]["frame"] = json!(6);
        } else {
            events.push(event(
                "kickoff:6",
                "kickoff",
                json!({"start_time": 3., "live_action_start_time": 4.}),
            ));
        }
        let row = row(document);
        assert_eq!(row["samples"]["play_end"], 3.);
        assert!(row["samples"]["second_touch"].is_null());
        for name in [
            "second_touch_delay_s",
            "opponent_whiff_during_exchange",
            "subject_beaten_during_exchange",
            "subject_next_touch_delay_s",
            "subject_displacement_to_second_touch_up",
            "exchange_ball_position_max_toward",
            "own_team_ball_distance_first_to_second_touch",
            "ball_own_goal_distance_at_second_touch",
            "exchange_possession_own_team_s",
            "subject_bumped_during_exchange",
            "teammate_whiff_during_exchange",
        ] {
            assert!(row["features"][name].is_null(), "{name}");
        }
        if boundary == "kickoff" {
            assert!(row["features"]["next_goal_delay_s"].is_null());
        }
    }
}

#[test]
fn team_coordinates_rotate_both_ground_axes_and_keep_height() {
    let mut document = fixture();
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    for event in events {
        if event["meta"]["stream"] == "touch" {
            let team = event["payload"]["payload"]["is_team_0"].as_bool().unwrap();
            event["payload"]["payload"]["is_team_0"] = json!(!team);
        }
    }
    let roster = &mut document["outputs"]["events"]["replay_meta"];
    let zero = roster["team_zero"].take();
    roster["team_zero"] = roster["team_one"].take();
    roster["team_one"] = zero;
    let row = row(document);
    assert_eq!(row["features"]["candidate_ball_position_toward"], -40.);
    assert_eq!(row["features"]["candidate_ball_position_side"], 20.);
    assert_eq!(row["features"]["candidate_ball_position_up"], 102.);
    assert_eq!(row["features"]["exchange_ball_position_min_toward"], -140.);
    assert_eq!(row["features"]["exchange_ball_position_max_toward"], -40.);
    assert_eq!(row["features"]["exchange_ball_position_min_side"], 20.);
    assert_eq!(row["features"]["exchange_ball_position_max_side"], 70.);
    assert_eq!(
        row["features"]["candidate_outgoing_ball_velocity_toward"],
        -4.
    );
    assert_eq!(
        row["features"]["closest_opponent_position_relative_ball_side"],
        -2.
    );
    assert_eq!(
        row["features"]["closest_opponent_position_relative_ball_up"],
        10.
    );
    assert_eq!(row["features"]["closest_opponent_heading_side"], 1.);
    assert_eq!(row["features"]["approaching_teammate_count"], 1);
    assert_eq!(row["features"]["approaching_opponent_count"], 0);
}

#[test]
fn missing_data_stays_null_and_a_demo_invalidates_displacement() {
    let mut document = fixture();
    document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][1]["Data"]["rigid_body"]
        ["linear_velocity"] = Value::Null;
    document["outputs"]["replay_data"]["frame_data"]["players"][2][1]["frames"][1] = json!("Empty");
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .push(event(
            "demo:6",
            "demolition",
            json!({"victim": {"Steam": 1}, "frame": 6, "time": 3.}),
        ));
    let row = row(document);
    assert!(row["features"]["candidate_incoming_ball_speed"].is_null());
    assert!(row["features"]["closest_opponent_heading_side"].is_null());
    assert!(row["features"]["approaching_opponent_count"].is_null());
    assert!(row["features"]["subject_displacement_to_first_touch_up"].is_number());
    assert!(row["features"]["subject_displacement_to_second_touch_up"].is_null());
    assert_eq!(row["features"]["subject_demo_during_exchange"], true);
}

#[test]
fn stage_headings_include_pitch_and_unavailable_bodies_stay_null() {
    let mut document = fixture();
    document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][5]["Data"]["rigid_body"]
        ["rotation"] = json!({"x": 0, "y": -1, "z": 0, "w": 1});
    document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][7] = json!("Empty");
    let row = row(document);
    assert_eq!(row["features"]["subject_heading_at_first_touch_up"], 1.);
    assert_eq!(row["features"]["subject_heading_at_first_touch_side"], 0.);
    assert!(row["features"]["subject_heading_at_second_touch_up"].is_null());
    assert!(row["features"]["subject_ball_distance_at_second_touch"].is_null());
    assert!(row["features"]["subject_displacement_to_second_touch_up"].is_null());
}

#[test]
fn a_demo_in_the_candidate_frame_is_retained_without_ordering_it() {
    let mut document = fixture();
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .push(event(
            "demo:2",
            "demolition",
            json!({"victim": {"Steam": 1}, "frame": 2, "time": 1.}),
        ));
    let row = row(document);
    assert_eq!(row["features"]["subject_demo_during_exchange"], true);
    assert!(row["features"]["subject_displacement_to_first_touch_up"].is_null());
}

#[test]
fn nearest_opponent_profile_comes_from_one_car_and_uses_3d_distance() {
    let mut document = fixture();
    let mut far = body(10., 20., 1000.);
    far["rotation"] = json!({"x": 0, "y": 1, "z": 0, "w": 1});
    far["linear_velocity"] = json!({"x": 500., "y": 500., "z": -100.});
    add_opponent(&mut document, 5, far, 255.);
    let row = row(document);
    assert_eq!(row["samples"]["closest_opponent_id"], json!({"Steam": 3}));
    assert_eq!(
        row["features"]["closest_opponent_position_relative_ball_up"],
        10.
    );
    assert_eq!(row["features"]["closest_opponent_heading_side"], -1.);
    assert_eq!(
        row["features"]["closest_opponent_velocity_relative_ball_up"],
        0.
    );
    assert_eq!(row["features"]["closest_opponent_boost_before_pct"], 60.);
    assert_eq!(row["features"]["approaching_opponent_count"], 1);
}

#[test]
fn approaching_counts_use_every_cars_own_3d_velocity_regardless_of_heading() {
    let mut document = fixture();
    let mut far = body(10., 20., 201.);
    far["linear_velocity"] = json!({"x": 0., "y": 0., "z": -2.});
    add_opponent(&mut document, 5, far, 255.);
    let frames = &mut document["outputs"]["replay_data"]["frame_data"];
    frames["ball_data"]["frames"][1]["Data"]["rigid_body"]["linear_velocity"] =
        json!({"x": 1000., "y": 0., "z": 1000.});
    for (index, velocity) in [
        (0, json!({"x": 1000., "y": 1000., "z": 1000.})),
        (1, json!({"x": 0., "y": 0., "z": -1.})),
        (2, json!({"x": 0., "y": 0., "z": 0.})),
        (3, json!({"x": 0., "y": 1., "z": 0.})),
    ] {
        frames["players"][index][1]["frames"][1]["Data"]["rigid_body"]["linear_velocity"] =
            velocity;
        frames["players"][index][1]["frames"][1]["Data"]["rigid_body"]["rotation"] =
            json!({"x": 0, "y": 0, "z": 1, "w": 0});
    }
    let row = row(document);
    assert_eq!(row["features"]["approaching_teammate_count"], 1);
    assert_eq!(row["features"]["approaching_opponent_count"], 1);
    assert_eq!(row["features"]["closest_opponent_heading_side"], 1.);
    assert_eq!(
        row["features"]["closest_opponent_velocity_relative_ball_side"],
        1000.
    );
    assert_eq!(
        row["features"]["closest_opponent_velocity_relative_ball_up"],
        -1000.
    );
}

#[test]
fn solo_players_have_zero_approaching_teammates() {
    let mut document = fixture();
    document["outputs"]["events"]["replay_meta"]["team_zero"] =
        json!([{"remote_id": {"Steam": 1}}]);
    document["outputs"]["replay_data"]["frame_data"]["players"]
        .as_array_mut()
        .unwrap()
        .retain(|player| player[0]["Steam"] != 2 && player[0]["Steam"] != 4);
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .retain(|event| {
            event["meta"]["stream"] != "touch"
                || event["payload"]["payload"]["player"]["Steam"] != 2
        });
    let row = row(document);
    assert_eq!(row["features"]["approaching_teammate_count"], 0);
    assert_eq!(row["samples"]["closest_opponent_id"], json!({"Steam": 3}));
}

#[test]
fn demolition_activity_excludes_cars_until_native_respawn() {
    let mut document = fixture();
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events.retain(|event| {
        event["meta"]["stream"] != "player_activity"
            || event["payload"]["payload"]["player"]["Steam"] != 3
    });
    events.push(activity(3, false, "demolished", 0, 1));
    events.push(activity(3, false, "tracked", 1, 13));
    events.push(touch(5, 9, 3, false, json!([])));
    for event in events.iter_mut().filter(|event| {
        event["meta"]["stream"] == "player_activity"
            && event["payload"]["payload"]["player"]["Steam"] == 4
    }) {
        event["payload"]["payload"]["state"] = json!("demolished");
    }
    let rows = serde_json::to_value(batch(document)).unwrap()["rows"]
        .as_array()
        .unwrap()
        .clone();
    assert!(rows[0]["samples"]["closest_opponent_id"].is_null());
    assert!(rows[0]["features"]["closest_opponent_position_relative_ball_side"].is_null());
    assert_eq!(rows[0]["features"]["approaching_opponent_count"], 0);
    assert_eq!(rows[0]["features"]["approaching_teammate_count"], 0);
    assert_eq!(
        rows.last().unwrap()["samples"]["closest_opponent_id"],
        json!({"Steam": 3})
    );
}

#[test]
fn missing_nearest_velocity_or_boost_does_not_select_another_opponent() {
    let mut document = fixture();
    add_opponent(&mut document, 5, body(100., 200., 300.), 255.);
    let nearest =
        &mut document["outputs"]["replay_data"]["frame_data"]["players"][2][1]["frames"][1]["Data"];
    nearest["rigid_body"]["linear_velocity"] = Value::Null;
    document["outputs"]["events"]["accumulation_tracks"][2]["points"] = json!([]);
    let row = row(document);
    assert_eq!(row["samples"]["closest_opponent_id"], json!({"Steam": 3}));
    assert_eq!(
        row["features"]["closest_opponent_position_relative_ball_side"],
        2.
    );
    assert_eq!(row["features"]["closest_opponent_heading_side"], -1.);
    assert!(row["features"]["closest_opponent_velocity_relative_ball_side"].is_null());
    assert!(row["features"]["closest_opponent_boost_before_pct"].is_null());
    assert!(row["features"]["approaching_opponent_count"].is_null());
    assert_eq!(row["features"]["approaching_teammate_count"], 1);
}

#[test]
fn missing_subject_heading_does_not_affect_surrounding_features() {
    let mut document = fixture();
    document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][1] = json!("Empty");
    let row = row(document);
    assert_eq!(row["features"]["closest_opponent_heading_side"], -1.);
    assert_eq!(row["features"]["approaching_teammate_count"], 1);
}

#[test]
fn missing_activity_does_not_treat_a_car_as_demolished() {
    let mut document = fixture();
    add_opponent(&mut document, 5, body(100., 200., 300.), 255.);
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .retain(|event| {
            event["meta"]["stream"] != "player_activity"
                || event["payload"]["payload"]["player"]["Steam"] != 3
        });
    let row = row(document);
    assert!(row["samples"]["closest_opponent_id"].is_null());
    assert!(row["features"]["closest_opponent_heading_side"].is_null());
    assert!(row["features"]["approaching_opponent_count"].is_null());
    assert_eq!(row["features"]["approaching_teammate_count"], 1);
}

#[test]
fn missing_ball_velocity_does_not_change_approaching_counts() {
    let mut document = fixture();
    document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][1]["Data"]["rigid_body"]
        ["linear_velocity"] = Value::Null;
    let row = row(document);
    assert_eq!(row["features"]["approaching_teammate_count"], 1);
    assert_eq!(row["features"]["approaching_opponent_count"], 0);
    assert!(row["features"]["closest_opponent_velocity_relative_ball_up"].is_null());
    assert_eq!(row["features"]["closest_opponent_heading_side"], -1.);
}

#[test]
fn tied_opponents_use_stable_player_identity_instead_of_input_order() {
    let mut document = fixture();
    add_opponent(&mut document, 5, body(8., 20., 111.), 255.);
    document["outputs"]["replay_data"]["frame_data"]["players"]
        .as_array_mut()
        .unwrap()
        .reverse();
    let row = row(document);
    assert_eq!(row["samples"]["closest_opponent_id"], json!({"Steam": 3}));
    assert_eq!(row["features"]["closest_opponent_boost_before_pct"], 60.);
}

#[test]
fn malformed_rosters_clocks_and_tags_fail_explicitly() {
    for change in ["roster", "time", "sample", "dodge", "boost", "activity"] {
        let mut document = fixture();
        match change {
            "roster" => {
                document["outputs"]["events"]
                    .as_object_mut()
                    .unwrap()
                    .remove("replay_meta");
            }
            "time" => {
                document["outputs"]["events"]["events"]["events"][1]["payload"]["payload"]["time"] =
                    json!(1.001)
            }
            "sample" => {
                document["outputs"]["events"]["events"]["events"][1]["payload"]["payload"]["sample_frame"] =
                    json!(1000)
            }
            "dodge" => {
                document["outputs"]["events"]["events"]["events"][1]["payload"]["payload"]["tags"]
                    [1]["value"] = json!("invalid")
            }
            "boost" => {
                document["outputs"]["events"]["accumulation_tracks"][2]["points"][0]["value"] =
                    json!(256.);
            }
            _ => {
                document["outputs"]["events"]["events"]["events"][14]["payload"]["payload"]["state"] =
                    json!("unknown");
            }
        }
        let analysis = serde_json::from_value(document).unwrap();
        assert!(
            extract(
                &analysis,
                FeatureSampling {
                    method: SamplingMethod::NativeTouchStages
                },
                Some(&json!({"Steam": 1}))
            )
            .is_err(),
            "{change}"
        );
    }
}

#[test]
fn stage_categories_use_the_judged_player_and_native_transition_times() {
    let mut document = fixture();
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    for (stream, state, start, end) in [
        ("field_half", "defensive", 0., 1.),
        ("field_half", "offensive", 1., 6.5),
        ("depth_role", "most_back", 0., 6.5),
        ("ball_depth", "ahead_of_ball", 0., 2.25),
        ("ball_depth", "behind_ball", 2.25, 3.25),
        ("ball_depth", "level_with_ball", 3.25, 6.5),
    ] {
        events.push(event(
            &format!("{stream}:{start}"),
            stream,
            json!({"player": {"Steam": 1}, "time": start, "end_time": end, "state": state}),
        ));
    }
    events.push(event(
        "movement:subject",
        "movement",
        json!({"player": {"Steam": 1}, "frame": 0, "end_frame": 13, "height_band": "low_air"}),
    ));
    events.push(event(
        "other-depth",
        "ball_depth",
        json!({"player": {"Steam": 2}, "time": 0., "end_time": 6.5, "state": "ahead_of_ball"}),
    ));
    let row = row(document);
    for name in [
        "subject_depth_role_at_candidate_most_back",
        "subject_height_band_at_candidate_low_air",
        "subject_ball_depth_at_candidate_ahead_of_ball",
        "subject_ball_depth_at_second_touch_level_with_ball",
    ] {
        assert_eq!(row["features"][name], true, "{name}");
    }
    assert_eq!(
        row["features"]["subject_field_half_at_candidate_offensive"],
        false
    );
    assert_eq!(
        row["features"]["subject_ball_depth_at_first_touch_ahead_of_ball"],
        false
    );
    assert_eq!(
        row["features"]["subject_ball_depth_at_first_touch_level_with_ball"],
        false
    );
}

#[test]
fn contact_frames_define_ball_speed_for_delayed_touch_confirmations() {
    let mut document = fixture();
    document["outputs"]["events"]["events"]["events"][3]["payload"]["payload"]["sample_frame"] =
        json!(6);
    document["outputs"]["events"]["events"]["events"][3]["payload"]["payload"]["sample_time"] =
        json!(3.);
    document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][2]["Data"]["rigid_body"]
        ["linear_velocity"] = json!({"x": 6., "y": 8., "z": 24.});
    document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][6]["Data"]["rigid_body"]
        ["linear_velocity"] = json!({"x": 0., "y": 20., "z": 21.});
    document["outputs"]["events"]["events"]["events"][4]["payload"]["payload"]["sample_frame"] =
        json!(11);
    document["outputs"]["events"]["events"]["events"][4]["payload"]["payload"]["sample_time"] =
        json!(5.5);
    let measured = row(document.clone());
    assert_eq!(measured["features"]["candidate_outgoing_ball_speed"], 26.);
    assert_eq!(measured["features"]["first_touch_outgoing_ball_speed"], 13.);
    assert_eq!(
        measured["features"]["second_touch_outgoing_ball_speed"],
        13.
    );
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert_eq!(row(document), measured);
}

#[test]
fn missing_contact_velocity_stays_null_when_classification_velocity_is_available() {
    let mut document = fixture();
    document["outputs"]["replay_data"]["frame_data"]["ball_data"]["frames"][2]["Data"]["rigid_body"]
        ["linear_velocity"] = Value::Null;
    assert!(row(document)["features"]["candidate_outgoing_ball_speed"].is_null());
}

#[test]
fn delayed_classification_still_requires_valid_native_sampling_metadata() {
    let mut document = fixture();
    document["outputs"]["events"]["events"]["events"][3]["payload"]["payload"]["sample_frame"] =
        json!(6);
    document["outputs"]["events"]["events"]["events"][3]["payload"]["payload"]["sample_time"] =
        json!(2.5);
    let analysis = serde_json::from_value(document).unwrap();
    assert!(
        extract(
            &analysis,
            FeatureSampling {
                method: SamplingMethod::NativeTouchStages
            },
            Some(&json!({"Steam": 1}))
        )
        .is_err()
    );
}

#[test]
fn boost_tracks_and_carry_episodes_are_independent_measurements() {
    for (stream, ground, air) in [("ball_carry", true, false), ("air_dribble", false, true)] {
        let mut document = fixture();
        document["outputs"]["events"]["events"]["events"][11] = event(
            "carry",
            stream,
            json!({"player_id": {"Steam": 1}, "start_time": 0.5, "end_time": 1.}),
        );
        document["outputs"]["events"]["accumulation_tracks"][0]["points"] = json!([
            {"frame": 0, "value": 0.}, {"frame": 1, "value": 127.5}, {"frame": 2, "value": 255.}
        ]);
        document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][1]["Data"]["boost_amount"] =
            json!(0.);
        let measured = row(document);
        assert_eq!(measured["features"]["subject_boost_before_pct"], 50.);
        assert_eq!(
            measured["features"]["closest_opponent_boost_before_pct"],
            60.
        );
        assert_eq!(measured["features"]["ground_carry_before"], ground);
        assert_eq!(measured["features"]["air_dribble_before"], air);
    }
}
