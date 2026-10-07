use rl_mistake_analysis_candidates::{CandidateBatch, MistakeKind, ReplayFacts};

fn facts() -> ReplayFacts {
    serde_json::from_str(include_str!("../../../examples/bump.json")).unwrap()
}

#[test]
fn candidate_entry_point_rejects_failed_suppliers_and_duplicate_events() {
    let kind = MistakeKind::BumpingTeammate;
    let mut replay = facts();
    replay.status = "error".into();
    assert!(kind.candidates(&replay).is_err());
    replay.status = "ok".into();
    replay.events.push(replay.events[0].clone());
    assert!(kind.candidates(&replay).is_err());
}

#[test]
fn serialized_batches_reject_changed_feature_contracts_and_duplicate_contacts() {
    let batch = MistakeKind::BumpingTeammate.candidates(&facts()).unwrap();
    let mut restored: CandidateBatch =
        serde_json::from_str(&serde_json::to_string(&batch).unwrap()).unwrap();
    restored.validate().unwrap();
    restored.feature_names.reverse();
    assert!(restored.validate().is_err());
    restored = batch.clone();
    restored.input_schema = "bumping_teammate.native.v2".into();
    assert!(restored.validate().is_err());
    restored = batch.clone();
    restored.candidates.push(restored.candidates[0].clone());
    assert!(restored.validate().is_err());
    restored = batch;
    restored.candidates[0].features[0] = f64::INFINITY;
    assert!(restored.validate().is_err());
}
