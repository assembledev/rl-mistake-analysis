pub use crate::feature_values::{DataType, FeatureDefinition};

pub fn definitions() -> Vec<FeatureDefinition> {
    let mut definitions = Vec::new();
    let mut add = |name: String, data_type, nullable, source: &str| {
        definitions.push(FeatureDefinition {
            name,
            data_type,
            nullable,
            source: source.into(),
        });
    };
    for name in [
        "approach_duration_s",
        "passage_delay_after_approach_s",
        "subject_max_advance",
        "subject_retreat_distance",
        "subject_path_distance",
        "subject_mean_speed",
        "subject_max_speed",
        "subject_orientation_change",
        "subject_boost_used",
        "subject_jump_active_s",
        "subject_dodge_active_s",
    ] {
        add(
            name.into(),
            DataType::F32,
            true,
            "Native approach frames, start through end",
        );
    }
    add(
        "subject_big_boost_pickup_count".into(),
        DataType::U32,
        true,
        "Explicit native big-pad pickups from approach start through end; ambiguous pad type is unknown",
    );
    add(
        "subject_nearest_big_pad_closing_advantage_mean".into(),
        DataType::F32,
        true,
        "Native time-weighted mean of closing speed toward the fixed nearest Big pad minus closing speed toward the ball, from approach start through end",
    );
    for prefix in [
        "candidate_subject_velocity",
        "candidate_subject_heading",
        "candidate_subject_position_relative_ball",
        "candidate_ball_position",
        "candidate_ball_velocity",
    ] {
        for axis in ["toward", "side", "up"] {
            add(
                format!("{prefix}_{axis}"),
                DataType::F32,
                true,
                "Native start frame in the team's canonical basis",
            );
        }
    }
    for name in [
        "subject_boost_before_pct",
        "candidate_ball_own_goal_distance",
        "closest_opponent_ball_distance",
        "closest_opponent_closing_speed",
        "closest_opponent_boost_before_pct",
        "closest_teammate_ball_distance",
        "closest_teammate_closing_speed",
        "closest_teammate_boost_before_pct",
        "subject_depth_lead",
        "teammate_goal_distance_min",
    ] {
        add(
            name.into(),
            DataType::F32,
            true,
            "Native opening geometry; boost uses the preceding native frame",
        );
    }
    for name in [
        "approaching_teammate_count",
        "approaching_opponent_count",
        "goal_side_teammate_count",
    ] {
        add(
            name.into(),
            DataType::U32,
            true,
            "Active players at the native start frame",
        );
    }
    for axis in ["toward", "side", "up"] {
        add(
            format!("opponent_touch_ball_velocity_delta_{axis}"),
            DataType::F32,
            true,
            "Native frames immediately before and at the last opponent contact before passage",
        );
    }
    for name in [
        "opponent_touch_incoming_ball_speed",
        "opponent_touch_outgoing_ball_speed",
        "subject_ball_distance_at_opponent_touch",
        "min_subject_ball_distance",
        "closest_approach_delay_s",
        "subject_height_at_closest_approach",
        "subject_heading_alignment_at_closest_approach",
        "opponent_ball_distance_min",
        "opponent_ball_distance_mean",
        "other_teammate_ball_distance_min",
        "other_teammate_ball_distance_mean",
        "peak_ball_height",
        "minimum_own_goal_ball_distance",
        "maximum_goalward_ball_speed",
    ] {
        add(
            name.into(),
            DataType::F32,
            true,
            "Native approach-to-passage geometry; distance means are time weighted",
        );
    }
    for name in [
        "opponent_touch_count",
        "subject_whiff_count",
        "subject_beaten_count",
        "subject_team_bump_count",
        "subject_opponent_bump_count",
        "subject_demo_inflicted_count",
    ] {
        add(
            name.into(),
            DataType::U32,
            false,
            "Native events from approach start through passage",
        );
    }
    add(
        "opponent_shot_before_own_touch".into(),
        DataType::Boolean,
        false,
        "Either native shot source from start strictly before the next own-team contact, within live play",
    );
    for name in [
        "subject_bump_max_impulse",
        "subject_bump_upright_after",
        "subject_bump_delay_to_closest_approach",
    ] {
        add(
            name.into(),
            DataType::F32,
            true,
            "Strongest received bump; exact acceleration-peak-then-minimum sampling",
        );
    }
    for axis in ["toward", "side", "up"] {
        add(
            format!("subject_bump_velocity_delta_{axis}"),
            DataType::F32,
            true,
            "Strongest bump velocity change in the team's canonical basis",
        );
    }
    for stage in ["first_touch", "second_touch"] {
        for relation in ["self", "teammate", "opponent"] {
            add(
                format!("{stage}_{relation}"),
                DataType::Boolean,
                true,
                "Native contact groups strictly after passage",
            );
        }
        add(
            format!("{stage}_delay_s"),
            DataType::F32,
            true,
            "Elapsed native time since passage",
        );
    }
    for name in [
        "previous_subject_touch_possession_before_self",
        "previous_subject_touch_followed_by_opponent",
    ] {
        add(
            name.into(),
            DataType::Boolean,
            true,
            "Native possession immediately before the previous subject touch and the first distinct following contact group within the saved live play",
        );
    }
    for name in [
        "subject_ball_withdrawal_distance_before_opponent_touch",
        "subject_ball_withdrawal_distance_after_opponent_touch",
    ] {
        add(
            name.into(),
            DataType::F32,
            true,
            "Native left-frame integral of horizontal movement away from the ball during the approach, partitioned at the last opponent touch before passage",
        );
    }
    add(
        "previous_subject_touch_delay_s".into(),
        DataType::F32,
        true,
        "Age at approach start of the preceding subject contact",
    );
    for team in ["own_team", "opponent"] {
        add(
            format!("next_goal_{team}"),
            DataType::Boolean,
            false,
            "Next native goal in the same live-play segment",
        );
    }
    add(
        "next_goal_delay_s".into(),
        DataType::F32,
        true,
        "Elapsed native time from passage to the next goal",
    );
    add(
        "touches_before_next_goal".into(),
        DataType::U32,
        true,
        "Distinct native touch IDs strictly between passage and next goal",
    );
    definitions
}
