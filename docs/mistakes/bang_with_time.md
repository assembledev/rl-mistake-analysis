# Banging the ball away with time

Commands run from the repository root. Shared formats and training APIs are described in the [format guide](../formats.md) and [training guide](../training.md).

## Candidates

```sh
target/debug/rl-mistake-analysis candidates bang_with_time < analysis.json > candidates.json
target/debug/rl-mistake-analysis contract bang_with_time
```

Candidates come from SubtrActor's original ball-touch timeline, ordered by frame and touch ID. Keep a touch when the next distinct contact frame includes an opponent. A following touch by only the same player or teammates does not qualify. Lookahead stops at the play's goal or next kickoff; a touch with no later contact in that play is excluded.

Exclude the candidate when its own touch or any touch at the next contact frame is tagged `contested=contested` or falls within a native fifty's `start_frame` through `resolve_frame`, inclusive. A bang followed by a fifty is therefore excluded too. Keep all contacts in the timeline used for lookahead; an excluded next contact does not permit selecting a later opponent touch. Simultaneous touchers at the next frame are considered together.

For example, a player's touches at frames 10 and 11 followed by an opponent at frame 20 keep only frame 11. A teammate's touch between frame 11 and the opponent instead makes that teammate's touch the candidate. Recorded player-possession intervals are not required.

Each candidate is identified by the replay checksum, frame, touch ID, and player: `(replay_sha256, frame, touch_id, player_id)`. The output uses schema `bang_with_time.candidates.v2` and keeps the original touch in `evidence`, including its `touch_id`, along with the source event ID, time, and supplier revision.

Candidate contract v2 requires the following opponent contact and excludes fifties at both contacts. It is independent of the numeric feature schema `bang_with_time.context.v1`. Model metadata can record the candidate contract alongside the numeric feature schema.

## Annotations

Each review supplies `kind: bang_with_time`, the exact touch key, `raw_time`, native `evidence`, binary `label`, and `engine_revision`. `source_event_ids` and `reject_reason` are optional. The judged player is the native toucher.

```sh
target/debug/rl-mistake-analysis validate-annotations reviews.json
python -m training.bang_dataset features.parquet reviews.json training.parquet
```

The join matches replay checksum, frame, touch ID, player ID, and replay timestamp. Supplied source event IDs must match the extracted event. It selects reviewed rows and adds the label, annotation revision, and rejection reason while preserving the feature contract and column order.

## Context extraction

`bang_with_time.context.v1` contains 163 typed measurements per selected touch. Extraction uses a combined `events,replay_data` analysis. `--player PLAYER_ID_JSON` selects one judged player; the full selection contains every candidate's feature row.

```sh
target/debug/rl-mistake-analysis bang-features examples/bang_sampling.json < analysis.json > features.json
target/debug/rl-mistake-analysis bang-feature-contract
python -m training.bang_features data/analyses features.parquet \
  --engine target/debug/rl-mistake-analysis \
  --sampling examples/bang_sampling.json
```

Each row carries its exact touch key, timestamp, source event, sample references, and typed measurements. `samples.closest_opponent_id` identifies the opponent sampled before contact; `samples.closest_opponent_at_candidate_id` identifies the closest opponent at contact. `samples.first_touch_opponent_id` identifies the unique opponent at the following contact, or is null when multiple opponents touch simultaneously. The Parquet builder stores the shared feature contract and extraction provenance described in the [format guide](../formats.md#feature-and-training-tables).

## Frame selection

Frame selection chooses which replay frames supply the measurements. The configuration is called `sampling` in files and commands. The [configuration file](../../examples/bang_sampling.json) records:

```json
{"method": "native_touch_stages"}
```

The candidate is the selected terminal touch. The previous stage is the preceding distinct contact frame; the first and second later stages are the next two distinct contact frames. A stage retains every simultaneous toucher. Stage lookup uses the original native touch timeline, including contested contacts.

Measurements stay within the observed live-play segment: from the kickoff's `live_action_start_time` or observed segment start through the next goal, kickoff start, or replay end. Goal-boundary touches are included. Delays use elapsed replay seconds from native frame timestamps.

Ball positions and outgoing speed use the shared contact frame. Incoming speed uses the preceding frame in the same segment. Native classification may confirm a toucher later; its `sample_frame` and `sample_time` are validated but do not select the ball measurement. Event contact timestamps must match their native frames.

Team axes place the arena center at the origin. `toward` points toward the opposing goal, `side` is the perpendicular ground axis, and `up` is world height. Both ground axes rotate with the judged player's team. Distances and speeds use full 3D native coordinates; headings use normalized car quaternions.

Surrounding-player measurements use the preceding frame and native tracked-activity spans. Approaching counts include tracked teammates other than the subject and tracked opponents. The opponent profile selects the smallest 3D car-to-ball distance, with lexicographically smallest serialized player ID resolving ties. Its position, heading, velocity, and boost refer to that selected player and frame.

The closest-opponent speed at the candidate uses the contact frame and the same distance and activity rules. The following opponent's orientation uses that opponent's contact frame. Its forward and up vectors preserve heading, pitch and roll in team axes; simultaneous contacts by different opponents leave these vectors unavailable.

Unavailable stages, bodies, states, or required velocities produce nullable measurements. Unknown opponent activity or an unavailable tracked opponent body makes its team's count and the closest-opponent profile unavailable. Known tracked teams have a count of the cars meeting the approach condition, including zero. Numeric observations must be finite and representable as float32.

## Feature calculations

`stage` means `first_touch` or `second_touch`; `axis` means `toward`, `side`, or `up`. The subject is the judged player. Columns retain the types and nullability declared by `bang-feature-contract`. Event flags describe the presence of the specified event within an observed interval; exchange flags require the second stage as their endpoint.

| Features | Type | Calculation | Source |
| --- | --- | --- | --- |
| `candidate_ball_position_axis` | nullable float32 | Native ball center at the candidate contact frame, expressed in team axes. | `replay_data.frame_data.ball_data` |
| `candidate_incoming_ball_speed` | nullable float32 | Length of the native ball velocity in the preceding frame. | `replay_data.frame_data.ball_data` |
| `candidate_outgoing_ball_speed` / `stage_outgoing_ball_speed` | nullable float32 | Length of the native ball velocity at the shared contact frame. Delayed classification does not change the measurement frame. | `events.touch.frame`, `replay_data.frame_data.ball_data` |
| `closest_opponent_position_relative_ball_axis` | nullable float32 | Selected opponent's car center minus the ball center in the preceding frame, expressed in team axes. | `events.replay_meta`, `events.player_activity`, `replay_data.frame_data` |
| `closest_opponent_heading_axis` | nullable float32 | Selected opponent's forward unit vector from its normalized native quaternion in the preceding frame, expressed in team axes. | `events.replay_meta`, `events.player_activity`, `replay_data.frame_data` |
| `closest_opponent_velocity_relative_ball_axis` | nullable float32 | Selected opponent's velocity minus the ball velocity in the preceding frame, expressed in team axes. | `events.replay_meta`, `events.player_activity`, `replay_data.frame_data` |
| `closest_opponent_boost_before_pct` / `subject_boost_before_pct` | nullable float32 | Latest native boost-track point at or before the preceding frame, multiplied by `100/255`. | `events.accumulation_tracks.boost_amount` |
| `approaching_teammate_count` / `approaching_opponent_count` | nullable uint32 | Count tracked cars with a strictly positive dot product between their own velocity and their car-to-ball vector. Teammates are the subject's other team members. | `events.replay_meta`, `events.player_activity`, `replay_data.frame_data` |
| `candidate_action_*` | bool | Native action tag: shot, save, clear, pass, boom, or unclassified. | `events.touch.tags.action` |
| `candidate_dodging` | bool | Native `dodge_state` equals `dodge`. | `events.touch.tags.dodge_state` |
| `subject_field_half_at_candidate_offensive` | nullable bool | Native `field_half` at candidate time: offensive is true and defensive is false. | `events.field_half` |
| `subject_depth_role_at_candidate_*` | nullable bool | Native `depth_role` at candidate time: most_back, most_forward, mid, or no_teammates. The other category sets all indicators false. | `events.depth_role` |
| `subject_height_band_at_candidate_*` | nullable bool | Native movement height band at the candidate frame: low_air and high_air have indicators; ground sets both false. | `events.movement` |
| `subject_ball_depth_at_candidate_*` / `subject_ball_depth_at_stage_*` | nullable bool | Subject's native `ball_depth` at the corresponding contact time: level_with_ball and ahead_of_ball have indicators; behind_ball sets both false. | `events.ball_depth` |
| `opponent_beaten_at_candidate` | bool | Native `beaten_to_ball` event at the candidate frame with the subject as winner and an opponent as the beaten player. | `events.beaten_to_ball` |
| `previous_touch_self/teammate/opponent` | nullable bool | Relationships represented among the preceding stage's touchers. Teammate means another member of the subject's team. | `events.touch` |
| `previous_touch_delay_s` | nullable float32 | Candidate time minus the preceding stage's native time. | `events.touch` |
| `stage_self/teammate/opponent` | nullable bool | Relationships represented among that stage's touchers. Simultaneous touches can set several flags. | `events.touch` |
| `stage_delay_s` | nullable float32 | Stage time minus candidate time. | `events.touch` |
| `stage_ball_position_axis` | nullable float32 | Native ball center at the stage's contact frame, expressed in team axes. | `replay_data.frame_data.ball_data` |
| `stage_incoming_ball_speed` | nullable float32 | Length of the native ball velocity in the frame immediately before the stage. | `replay_data.frame_data.ball_data` |
| `stage_action_*` | nullable bool | Every native action category represented among that stage's touchers, including unclassified. | `events.touch.tags.action` |
| `opponent_whiff_during_exchange` | nullable bool | Confirmed native opponent whiff whose closest-approach frame is after the candidate and through the second stage, with `resolved_time` at or before that stage. | `events.whiff` |
| `subject_beaten_during_exchange` | nullable bool | Native `beaten_to_ball` event after the candidate frame and through the second stage with the subject as the beaten player. | `events.beaten_to_ball` |
| `subject_demo_during_exchange` | nullable bool | Native demolition of the subject in the inclusive interval from the candidate frame through the second stage. | `events.demolition` |
| `first_touch_opponent_control` | nullable bool | An opponent touch at the first later stage has a native possession tag of control or advance. | `events.touch.tags.possession` |
| `possession_before_own_team/opponent/neutral` | nullable bool | Native possession span starting before and covering candidate time, expressed relative to the subject's team. | `events.possession` |
| `ground_carry_before` / `air_dribble_before` | bool | Corresponding native episode for the subject starts earlier in the current play and covers candidate time. | `events.ball_carry`, `events.air_dribble` |
| `receiving_pass_before` | bool | Native pass names the subject as receiver, starts before the candidate, and has a reception time at or after it. | `events.pass` |
| `next_possession_own_team/opponent/neutral/no_change` | nullable bool | Earliest changed possession span starting at or after candidate time and before play end. The no_change category records an unchanged observed interval. | `events.possession` |
| `possession_change_delay_s` | nullable float32 | Changed possession span's start time minus candidate time. | `events.possession` |
| `next_goal_own_team/opponent` | bool | Scoring team of the goal ending this play segment, expressed relative to the subject's team. | `events.goal_context` |
| `next_goal_delay_s` | nullable float32 | Goal time minus candidate time. | `events.goal_context` |
| `touches_before_next_goal` | nullable uint32 | Count distinct native touch IDs strictly between candidate and goal times. Simultaneous contacts retain their separate IDs. | `events.touch`, `events.goal_context` |
| `subject_next_touch_delay_s` | nullable float32 | Subject's first touch in a later contact frame in this play, measured from candidate time. | `events.touch` |
| `subject_displacement_to_stage_axis` | nullable float32 | Subject's stage position minus its candidate position, expressed in team axes. A continuous observed subject track within one life defines a valid displacement. | `replay_data.frame_data`, `events.demolition` |
| `subject_ball_distance_at_stage` | nullable float32 | Full 3D distance between the subject's car center and ball center at the stage's contact frame. | `replay_data.frame_data` |
| `subject_heading_at_stage_axis` | nullable float32 | Subject's forward unit vector from its normalized native quaternion at the stage's contact frame, expressed in team axes. | `replay_data.frame_data.players.rigid_body.rotation` |
| `play_elapsed_s` | float32 | Candidate time minus the observed live-play start. | `events.kickoff, replay_data.frame_data.metadata_frames` |
| `candidate_outgoing_ball_velocity_toward` | nullable float32 | Native outgoing ball velocity along the toward axis at touch 0. | `replay_data.frame_data.ball_data` |
| `exchange_ball_position_min_toward` | nullable float32 | Minimum ball-center toward coordinate over every native frame from touch 0 through touch 2. | `replay_data.frame_data.ball_data` |
| `exchange_ball_position_max_toward` | nullable float32 | Maximum ball-center toward coordinate over every native frame from touch 0 through touch 2. | `replay_data.frame_data.ball_data` |
| `candidate_outgoing_ball_velocity_side` | nullable float32 | Native outgoing ball velocity along the side axis at touch 0. | `replay_data.frame_data.ball_data` |
| `exchange_ball_position_min_side` | nullable float32 | Minimum ball-center side coordinate over every native frame from touch 0 through touch 2. | `replay_data.frame_data.ball_data` |
| `exchange_ball_position_max_side` | nullable float32 | Maximum ball-center side coordinate over every native frame from touch 0 through touch 2. | `replay_data.frame_data.ball_data` |
| `candidate_outgoing_ball_velocity_up` | nullable float32 | Native outgoing ball velocity along the up axis at touch 0. | `replay_data.frame_data.ball_data` |
| `exchange_ball_position_min_up` | nullable float32 | Minimum ball-center up coordinate over every native frame from touch 0 through touch 2. | `replay_data.frame_data.ball_data` |
| `exchange_ball_position_max_up` | nullable float32 | Maximum ball-center up coordinate over every native frame from touch 0 through touch 2. | `replay_data.frame_data.ball_data` |
| `candidate_directed_at_own_goal` | nullable bool | The forward ray of the outgoing ball velocity overlaps the own goal opening, accounting for the ball radius. | `replay_data.frame_data.ball_data` |
| `candidate_directed_at_opponent_goal` | nullable bool | The forward ray of the outgoing ball velocity overlaps the opponent goal opening, accounting for the ball radius. | `replay_data.frame_data.ball_data` |
| `subject_bumped_during_exchange` | nullable bool | A native bump names the judged player as victim during the inclusive interval from touch 0 through touch 2. | `events.bump` |
| `any_bump_during_exchange` | nullable bool | Any native bump occurs during the inclusive interval from touch 0 through touch 2. | `events.bump` |
| `subject_whiff_during_exchange` | nullable bool | The judged player has a native whiff anchored after touch 0 through touch 2 and confirmed by touch 2. | `events.whiff` |
| `teammate_whiff_during_exchange` | nullable bool | Another player on the judged team has a native whiff anchored after touch 0 through touch 2 and confirmed by touch 2. | `events.whiff` |
| `own_team_ball_distance_candidate_to_first_touch` | nullable float32 | Minimum over every native frame from touch 0 through touch 1 of the closest tracked own-team car-center to ball-center distance. The own team includes the judged player. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `own_team_ball_distance_mean_candidate_to_first_touch` | nullable float32 | Time-weighted mean of the closest tracked own-team car-center to ball-center distance from touch 0 through touch 1. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `opponent_ball_distance_candidate_to_first_touch` | nullable float32 | Minimum over every native frame from touch 0 through touch 1 of the closest tracked opponent car-center to ball-center distance. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `opponent_ball_distance_mean_candidate_to_first_touch` | nullable float32 | Time-weighted mean of the closest tracked opponent car-center to ball-center distance from touch 0 through touch 1. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `ball_own_goal_distance_candidate_to_first_touch` | nullable float32 | Minimum over every native frame from touch 0 through touch 1 of the ball-surface distance to the nearest point of the own goal opening. Overlap has distance zero. | `replay_data.frame_data.ball_data` |
| `ball_opponent_goal_distance_candidate_to_first_touch` | nullable float32 | Minimum over every native frame from touch 0 through touch 1 of the ball-surface distance to the nearest point of the opponent goal opening. Overlap has distance zero. | `replay_data.frame_data.ball_data` |
| `own_team_ball_distance_at_first_touch` | nullable float32 | Measurement at touch 1 of the closest tracked own-team car-center to ball-center distance. The own team includes the judged player. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `opponent_ball_distance_at_first_touch` | nullable float32 | Measurement at touch 1 of the closest tracked opponent car-center to ball-center distance. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `ball_own_goal_distance_at_first_touch` | nullable float32 | Measurement at touch 1 of the ball-surface distance to the nearest point of the own goal opening. Overlap has distance zero. | `replay_data.frame_data.ball_data` |
| `ball_opponent_goal_distance_at_first_touch` | nullable float32 | Measurement at touch 1 of the ball-surface distance to the nearest point of the opponent goal opening. Overlap has distance zero. | `replay_data.frame_data.ball_data` |
| `own_team_ball_distance_first_to_second_touch` | nullable float32 | Minimum over every native frame from touch 1 through touch 2 of the closest tracked own-team car-center to ball-center distance. The own team includes the judged player. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `own_team_ball_distance_mean_first_to_second_touch` | nullable float32 | Time-weighted mean of the closest tracked own-team car-center to ball-center distance from touch 1 through touch 2. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `opponent_ball_distance_first_to_second_touch` | nullable float32 | Minimum over every native frame from touch 1 through touch 2 of the closest tracked opponent car-center to ball-center distance. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `opponent_ball_distance_mean_first_to_second_touch` | nullable float32 | Time-weighted mean of the closest tracked opponent car-center to ball-center distance from touch 1 through touch 2. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `ball_own_goal_distance_first_to_second_touch` | nullable float32 | Minimum over every native frame from touch 1 through touch 2 of the ball-surface distance to the nearest point of the own goal opening. Overlap has distance zero. | `replay_data.frame_data.ball_data` |
| `ball_opponent_goal_distance_first_to_second_touch` | nullable float32 | Minimum over every native frame from touch 1 through touch 2 of the ball-surface distance to the nearest point of the opponent goal opening. Overlap has distance zero. | `replay_data.frame_data.ball_data` |
| `own_team_ball_distance_at_second_touch` | nullable float32 | Measurement at touch 2 of the closest tracked own-team car-center to ball-center distance. The own team includes the judged player. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `opponent_ball_distance_at_second_touch` | nullable float32 | Measurement at touch 2 of the closest tracked opponent car-center to ball-center distance. | `events.replay_meta, events.player_activity, replay_data.frame_data` |
| `ball_own_goal_distance_at_second_touch` | nullable float32 | Measurement at touch 2 of the ball-surface distance to the nearest point of the own goal opening. Overlap has distance zero. | `replay_data.frame_data.ball_data` |
| `ball_opponent_goal_distance_at_second_touch` | nullable float32 | Measurement at touch 2 of the ball-surface distance to the nearest point of the opponent goal opening. Overlap has distance zero. | `replay_data.frame_data.ball_data` |
| `exchange_possession_own_team_s` | nullable float32 | Sum of native own_team possession-span durations clipped to the interval from touch 0 through touch 2. | `events.possession` |
| `exchange_possession_opponent_s` | nullable float32 | Sum of native opponent possession-span durations clipped to the interval from touch 0 through touch 2. | `events.possession` |
| `exchange_possession_neutral_s` | nullable float32 | Sum of native neutral possession-span durations clipped to the interval from touch 0 through touch 2. | `events.possession` |
| `subject_consecutive_touches_before_candidate` | uint32 | Count distinct native subject touch IDs in the consecutive run immediately before the candidate. Stop at another player's contact or the observed play start; exclude the candidate itself. | `events.touch` |
| `previous_opponent_touch_delay_s` | nullable float32 | Candidate time minus the latest earlier opponent contact in the same observed play. Unavailable when no earlier opponent contact was observed. | `events.touch` |
| `next_opponent_possession_duration_s` | nullable float32 | Full active native opponent-possession span containing the first later contact, including its portion before the candidate, clipped to observed play boundaries. An observed own-team or neutral span gives zero; an unobserved span gives null. Span starts are inclusive and ends exclusive, except at the terminal play boundary. | `events.possession` |
| `subject_height_at_candidate` | nullable float32 | Judged player's native car-center height at the candidate contact frame. | `replay_data.frame_data.players.rigid_body.location` |
| `closest_opponent_speed_at_candidate` | nullable float32 | Length of the closest tracked opponent's native 3D velocity at the candidate contact frame. | `events.replay_meta`, `events.player_activity`, `replay_data.frame_data` |
| `subject_field_third_at_candidate_defensive` | nullable bool | Judged player occupies the native defensive third at candidate time. An observed neutral third sets both third indicators false. | `events.field_third` |
| `subject_field_third_at_candidate_offensive` | nullable bool | Judged player occupies the native offensive third at candidate time. | `events.field_third` |
| `subject_shadow_defense_at_candidate` | nullable bool | Native shadow-defense span covers candidate time for the observed judged player. | `events.shadow_defense`, `events.player_activity`, `replay_data.frame_data` |
| `own_team_shadow_defense_at_candidate` | nullable bool | At least one own-team player has a native shadow-defense span covering candidate time. Requires observed activity and tracked bodies for every active own-team player; demolished players count as false. | `events.shadow_defense`, `events.player_activity`, `replay_data.frame_data` |
| `candidate_surface_air` | nullable bool | Native candidate surface tag is air. An observed ground tag sets both surface indicators false. | `events.touch.tags.surface` |
| `candidate_surface_wall` | nullable bool | Native candidate surface tag is wall. A missing surface tag leaves both indicators null. | `events.touch.tags.surface` |
| `first_touch_opponent_heading_toward` | nullable float32 | Following opponent's forward unit vector along the subject team's toward axis at that contact frame. | `events.touch`, `replay_data.frame_data.players.rigid_body.rotation` |
| `first_touch_opponent_heading_side` | nullable float32 | Following opponent's forward unit vector along the subject team's side axis at that contact frame. | `events.touch`, `replay_data.frame_data.players.rigid_body.rotation` |
| `first_touch_opponent_heading_up` | nullable float32 | Following opponent's forward unit vector along world up at that contact frame. | `events.touch`, `replay_data.frame_data.players.rigid_body.rotation` |
| `first_touch_opponent_up_toward` | nullable float32 | Following opponent's up unit vector along the subject team's toward axis at that contact frame. | `events.touch`, `replay_data.frame_data.players.rigid_body.rotation` |
| `first_touch_opponent_up_side` | nullable float32 | Following opponent's up unit vector along the subject team's side axis at that contact frame. | `events.touch`, `replay_data.frame_data.players.rigid_body.rotation` |
| `first_touch_opponent_up_up` | nullable float32 | Following opponent's up unit vector along world up at that contact frame. | `events.touch`, `replay_data.frame_data.players.rigid_body.rotation` |

Interval aggregates require their endpoints and the relevant observations throughout the interval. A missing ball frame makes trajectory and goal-distance aggregates unavailable; an unknown activity state or missing tracked car makes that team's distance aggregate unavailable. Demolished players are excluded. A team-distance minimum is unavailable when no frame has a tracked car for that team. Possession durations are unavailable when native spans leave an unobserved gap; gaps are not counted as neutral possession.

Team-distance means use trapezoidal integration of the per-frame nearest distance, divided by elapsed replay time. The own team includes the judged player. A mean is unavailable for a zero-duration interval or when any frame has no observed tracked car for that team.

Goal openings use SubtrActor's standard Soccar dimensions and collision ball radius. Distance is the Euclidean distance from the ball center to the nearest point of the opening, minus the radius, clamped at zero. Direction checks use the measured velocity ray without gravity, defenders, later contacts, a speed threshold, or a prediction time limit. Ball-coordinate extrema use the same team axes as the contact snapshots.

## Training and scoring

### Feature selection

Models declare an ordered subset of the 163 known measurements in their manifest. The base schema is `bang_with_time.context.v1`; named selections use `bang_with_time.context.v1.NAME.v1`. Feature selection and numeric conversion use the [shared training API](../training.md#dataset-inputs).

### Scoring

```sh
target/debug/rl-mistake-analysis score MODEL_DIRECTORY bang_with_time VERSION < analysis.json
```

Rust callers receive matching candidate and feature batches from the extractor:

```rust
use rl_mistake_analysis_candidates::bang_features;
use rl_mistake_analysis_inference::Model;

let mut model = Model::load(model_directory, "bang_with_time", model_version)?;
let sampling = model.metadata().sampling.ok_or("missing context sampling contract")?;
let (candidates, features) = bang_features::extract(&analysis, sampling, None)?;
let predictions = model.score_context(candidates, features)?;
```

`analysis` is the combined export deserialized as `bang_features::ReplayAnalysis`. Extraction and scoring use the same selected touches and frame selection method, including player-specific selections.

### Model training

Prepare reviewed rows with `training.bang_dataset`, then load them with `training.dataset.load` and `arrays`. The [training guide](../training.md) covers grouped model comparison, experiment tracking, ONNX verification, and release metadata.
