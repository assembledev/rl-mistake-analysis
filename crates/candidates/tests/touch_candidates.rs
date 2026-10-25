use rl_mistake_analysis_candidates::{MistakeKind, ReplayFacts, annotations::Annotation};
use serde_json::{Value, json};

fn event(stream: &str, id: &str, payload: Value) -> Value {
    json!({"meta":{"stream":stream,"id":id},"payload":{"kind":stream,"payload":payload}})
}
fn touch(id: u64, frame: u64, player: &str, team_zero: bool) -> Value {
    event(
        "touch",
        &format!("touch:{id}"),
        json!({"touch_id":id,"frame":frame,
        "time":frame as f64/30.0,"player":{"Epic":player},"is_team_0":team_zero,"tags":[]}),
    )
}
fn fifty(start: u64, end: u64, first: u64, second: u64) -> Value {
    event(
        "fifty_fifty",
        &format!("fifty:{start}"),
        json!({"start_frame":start,
        "resolve_frame":end,"team_zero_touch_frame":first,"team_one_touch_frame":second}),
    )
}
fn facts(events: Vec<Value>) -> ReplayFacts {
    serde_json::from_value(
        json!({"name":"subtr-actor","schema_version":1,"status":"ok",
        "revision":"supplier","replay_sha256":"a".repeat(64),"events":events}),
    )
    .unwrap()
}
fn kept(events: Vec<Value>) -> Vec<u64> {
    MistakeKind::BangWithTime
        .candidates(&facts(events))
        .unwrap()
        .candidates
        .iter()
        .map(|candidate| candidate.frame)
        .collect()
}

#[test]
fn keeps_only_touches_immediately_before_the_opposing_team() {
    assert_eq!(
        kept(vec![
            touch(6, 60, "b", true),
            touch(2, 20, "b", true),
            touch(0, 10, "a", true),
            touch(5, 50, "a", true),
            touch(3, 30, "c", false),
            touch(1, 11, "a", true),
            touch(4, 40, "d", false)
        ]),
        vec![20, 40]
    );
}

#[test]
fn immediate_same_player_contacts_do_not_require_a_possession_episode() {
    for (first, continuation, opponent) in [(2950, 2951, 3010), (10704, 10706, 10817)] {
        let mut next = touch(1, continuation, "a", true);
        next["payload"]["payload"]["tags"] = json!([{"group":"reception","value":"continuation"}]);
        assert_eq!(
            kept(vec![
                touch(0, first, "a", true),
                next,
                touch(2, opponent, "b", false)
            ]),
            vec![continuation]
        );
    }
}

#[test]
fn excludes_every_touch_in_a_fifty_including_its_resolving_frame() {
    assert_eq!(
        kept(vec![
            touch(0, 5, "a", true),
            touch(1, 10, "b", false),
            touch(2, 11, "a", true),
            touch(3, 12, "b", false),
            touch(4, 15, "a", true),
            touch(5, 20, "b", false),
            touch(6, 21, "a", true),
            fifty(10, 15, 11, 10)
        ]),
        vec![20]
    );
}

#[test]
fn a_clean_touch_followed_by_an_opponent_fifty_is_excluded() {
    assert_eq!(
        kept(vec![
            touch(0, 10, "a", true),
            touch(1, 20, "b", false),
            touch(2, 21, "c", true),
            touch(3, 25, "b", false),
            touch(4, 35, "a", true),
            touch(5, 45, "b", false),
            fifty(20, 25, 21, 20)
        ]),
        vec![35]
    );
}

#[test]
fn excluded_touches_are_not_skipped_when_finding_the_next_contact() {
    for excluded_by_fifty in [false, true] {
        let mut next = touch(1, 20, "b", true);
        next["payload"]["payload"]["tags"] = json!([{"group":"contested","value":"contested"}]);
        let mut events = vec![
            touch(0, 10, "a", true),
            next,
            touch(2, 30, "c", false),
            touch(3, 40, "a", true),
        ];
        if excluded_by_fifty {
            events.push(fifty(20, 22, 20, 21));
        }
        assert_eq!(kept(events), vec![30]);
    }
}

#[test]
fn simultaneous_next_contacts_are_read_as_one_native_stage() {
    let mut teammate = touch(1, 20, "b", true);
    let mut opponent = touch(2, 20, "c", false);
    for contact in [&mut teammate, &mut opponent] {
        contact["payload"]["payload"]["tags"] = json!([{"group":"contested","value":"contested"}]);
    }
    assert_eq!(
        kept(vec![
            touch(0, 10, "a", true),
            teammate,
            opponent,
            touch(3, 30, "a", true),
            touch(4, 40, "c", false),
        ]),
        vec![30]
    );
    assert!(kept(vec![touch(0, 10, "a", true), touch(1, 10, "c", false)]).is_empty());
}

#[test]
fn goals_and_kickoffs_prevent_lookahead_into_another_play() {
    for boundary in [
        event("goal_context", "goal", json!({"time":20.0/30.0})),
        event("kickoff", "kickoff", json!({"start_time":20.0/30.0})),
    ] {
        assert_eq!(
            kept(vec![
                touch(0, 10, "a", true),
                touch(1, 30, "b", false),
                touch(2, 40, "a", true),
                boundary
            ]),
            vec![30]
        );
    }
    assert_eq!(
        kept(vec![
            touch(0, 10, "a", true),
            touch(1, 20, "b", false),
            touch(2, 30, "a", true),
            event("goal_context", "goal", json!({"time":20.0/30.0}))
        ]),
        vec![10]
    );
}

#[test]
fn kickoff_starting_at_the_next_contact_belongs_to_the_next_play() {
    assert_eq!(
        kept(vec![
            touch(0, 10, "a", true),
            touch(1, 30, "b", false),
            touch(2, 40, "a", true),
            event("kickoff", "kickoff", json!({"start_time":1.0}))
        ]),
        vec![30]
    );
}

#[test]
fn malformed_native_selection_inputs_fail_explicitly() {
    for events in [
        vec![touch(0, 10, "a", true), touch(0, 20, "b", false)],
        vec![fifty(20, 10, 10, 11)],
    ] {
        assert!(
            MistakeKind::BangWithTime
                .candidates(&facts(events))
                .is_err()
        );
    }
    let mut invalid = touch(0, 10, "a", true);
    invalid["payload"]["kind"] = json!("fifty_fifty");
    assert!(
        MistakeKind::BangWithTime
            .candidates(&facts(vec![invalid]))
            .is_err()
    );
}

#[test]
fn annotations_and_batches_preserve_the_exact_selected_native_touch() {
    let mut selected = touch(7, 42, "a", true);
    let time = 13.065633773803711;
    selected["payload"]["payload"]["time"] = json!(time);
    let mut opponent = touch(8, 50, "b", false);
    opponent["payload"]["payload"]["time"] = json!(14.0);
    let batch = MistakeKind::BangWithTime
        .candidates(&facts(vec![selected.clone(), opponent]))
        .unwrap();
    batch.validate().unwrap();
    assert!(batch.feature_names.is_empty());
    assert_eq!(batch.candidates.len(), 1);
    let candidate = &batch.candidates[0];
    assert_eq!(candidate.raw_time, time);
    assert_eq!(candidate.source_event_ids, vec!["touch:7"]);
    assert_eq!(candidate.evidence, selected["payload"]["payload"]);
    let mut annotation: Annotation = serde_json::from_value(json!({"kind":"bang_with_time",
        "replay_sha256":batch.replay_sha256,"frame":42,"touch_id":7,"raw_time":candidate.raw_time,
        "player_id":candidate.player_id,"evidence":candidate.evidence,"label":1,
        "engine_revision":"reviewed-supplier","source_event_ids":candidate.source_event_ids}))
    .unwrap();
    annotation.validate().unwrap();
    annotation.incident =
        serde_json::from_value(json!({"kind":"bang_with_time","frame":43,"touch_id":7})).unwrap();
    assert!(annotation.validate().is_err());
}
