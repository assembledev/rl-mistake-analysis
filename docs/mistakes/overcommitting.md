# Overcommitting

Commands run from the repository root. Shared formats and training APIs are described in the [format guide](../formats.md) and [training guide](../training.md).

An overcommitment is an unnecessary defensive challenge in which the player is beaten and leaves their team exposed without effective cover.

Candidate selection includes failed approaches that may have been justified. The model judges whether each attempt was an overcommitment.

## Candidates

```sh
target/debug/rl-mistake-analysis extract match.replay > analysis.json
target/debug/rl-mistake-analysis candidates overcommitting < analysis.json > candidates.json
target/debug/rl-mistake-analysis contract overcommitting
```

Selection requires the combined native `events,replay_data` export. Rust consumers call `MistakeKind::Overcommitting.candidates_from_analysis(&analysis)`.

1. **Open a defensive approach.** The player is tracked, on the goal side of the ball, and moving both upfield and toward the ball. Native possession is opponent or neutral, the latest contact frame contains only opponents, and the ball is moving toward the player's own end. The player must be deepest by Y **or** farthest from the ball among active goal-side teammates. These conditions apply at opening only.
2. **Finish the approach at passage or retreat.** End at the first observed ball passage, or when the player moves toward their own end and returns to or behind their opening Y position outside an active native dodge. After a retreat finish, keep the same attempt pending until passage.
3. **Confirm the failed attempt.** Ball passage must be observed before an own-team touch, participating native fifty, missing geometry, inactivity, demolition, or play boundary cancels the attempt. Require an opponent touch from opening through passage **or** an opponent shot from opening until before the next strictly later own-team touch or the play boundary. Shot evidence accepts either an `action=shot` touch tag or a native `timeline` event with `kind=Shot`.

After an own-team touch, opening may resume at the first eligible frame. A participating fifty blocks opening through its resolution frame regardless of winner.

Rank comparisons retain ties, exclude demolished teammates, and require tracked geometry for the remaining teammates. They apply in 1v1, 2v2, and 3v3.

## Annotations

The incident key is `(replay_sha256, frame, end_frame, player_id)`. `frame` is the opening frame; `end_frame` is the approach's passage or retreat finish. `passage_frame` records the later failure confirmation and equals `end_frame` when passage itself ends the approach. `raw_time` is the native opening timestamp. Reviews carry binary `label`, exact `evidence`, supplier `engine_revision`, and the contributing native `source_event_ids`.

```sh
target/debug/rl-mistake-analysis validate-annotations reviews.json
```

The candidate schema is `overcommitting.candidates.v1`; `feature_names` and `features` are empty.

## Context extraction

`overcommitting.context.v1` produces one typed feature row per candidate interval and judged player. It requires combined native `events,replay_data` analysis. `toward` points toward the opponent goal, `side` is the perpendicular ground axis, and `up` is world height.

```sh
target/debug/rl-mistake-analysis overcommitting-features examples/overcommitting_sampling.json < analysis.json > features.json
target/debug/rl-mistake-analysis overcommitting-feature-contract
python -m training.overcommitting_features data/analyses features.parquet \
  --engine target/debug/rl-mistake-analysis \
  --sampling examples/overcommitting_sampling.json
python -m training.overcommitting_dataset features.parquet reviews.json training.parquet
```

The sampling file declares `{"method": "native_interval_boundaries"}`. `--player PLAYER_ID_JSON` selects a judged player. To measure exact supplied intervals, pass `--candidates candidates.json`. The file must contain a JSON array with exactly one batch, as written by `candidates overcommitting`. Extraction validates the batch's replay checksum, players, native boundaries, timestamps, and source events without rerunning candidate selection. Python callers pass the corresponding analysis-to-candidate-file mapping as `candidate_batches` to `training.overcommitting_features.compute`.

Rows preserve `source_event_ids`. Sample references identify the boundaries, closest approach, selected players, contacts, possession, big pad, and strongest received bump. Annotation joins check the interval, player, passage, timestamps, and source events.

Missing observations and absent contact or impact stages produce `null`. Observed event counts and flags use zero or false when no matching event occurs.

## Frame selection

All comparisons use native frames without resampling. Player activity uses native spans with an exclusive opening time and inclusive ending time. Candidate-opening possession uses the latest native span starting at or before the frame time, with an exclusive ending time. Ownership immediately before the previous subject touch uses start_time < touch_time ≤ end_time. Live-play segments are the union of contiguous player activity spans, ending at the associated goal frame when present.

Let `s` be `+1` for blue and `-1` for orange. Goal-side means `s × (ball.y − player.y) ≥ 0`; passage means the same quantity becomes strictly negative. Upfield motion requires `s × player.velocity.y > 0`, and incoming ball motion requires `s × ball.velocity.y < 0`. Motion toward the ball requires the player's native 3D velocity dotted with `ball.position − player.position` to be strictly positive.

The retreat finish requires `s × player.velocity.y < 0`, `s × (player.y − opening.y) ≤ 0`, and native `dodge_active = false` at the same frame.

Deepest by Y means no tracked teammate has smaller `s × y`. Farthest goal-side means no tracked goal-side teammate has greater squared 3D distance to the ball.

## Feature calculations

The extraction catalog contains **103 scalar columns**. One interval produces one row. Models select an explicit ordered subset of this catalog.

`S` is the opening frame, `E` the approach finish, and `P` the passage confirmation. `C` is the last opponent contact frame in S..P, or unavailable when no such contact exists. `M` is the first frame attaining the minimum subject-to-ball distance in S..P. `A` is the first frame attaining that minimum in S..E, describing the approach before any later passage confirmation. `T` is the subject’s preceding contact frame before S in the same play. Frame ranges include both endpoints unless stated otherwise. Later contacts and goals stop at the saved live-play boundary.

Time-weighted means use trapezoidal integration over native timestamps. Distance minima exclude demolished players and are null when required tracking is unknown or no active car appears in the range. Means require an active car at every frame.

Opening closest-player selection uses 3D distance at S. The approach profile selects the closest opponent at A; goal-side teammate profiles select the closest other teammate satisfying s × (ball.y − teammate.y) ≥ 0 at A or P. These selections exclude the subject from teammate comparisons, ignore demolished players, and resolve ties by serialized player ID. Unknown active-player geometry leaves the corresponding profile unavailable. The shared bump sampler selects the native preceding frame, first acceleration peak, and following minimum within the same play; unavailable impact stages stay null. Whiff and beaten tags are auxiliary measurements.

Withdrawal uses the subject’s own horizontal velocity relative to the current car-to-ball direction. The before/after sums split at C and use the left frame’s measurement for each native time interval. A missing C leaves both distances null. An empty window is zero; a nonempty window with missing required geometry or velocity, or coincident horizontal car/ball centers, is null. These measurements describe movement and contact order.

An observed empty goal-side teammate set has count zero and no selected player. Unknown active-player geometry leaves the count and selected-player profile null. Closing speed is native car velocity dotted with the 3D unit car-to-ball vector at the selected frame; coincident centers leave it null. Opponent possession fraction integrates exact native possession-span intersections with S..E and divides by t(E) − t(S). Incomplete coverage leaves it null; overlapping spans are invalid.

Distances use native arena units, speeds use arena units per second, delays use replay seconds, and quaternion angles use radians. Goal geometry and ball radius come from SubtrActor constants. Missing size/index metadata, required velocities, or a coincident horizontal target leaves the mean null. The selected pad stays fixed through the interval.

| Feature | Type | Calculation | Source |
| --- | --- | --- | --- |
| `approach_duration_s` | nullable float32 | t(E) − t(S). | Native car/ball frames, metadata timestamps |
| `passage_delay_after_approach_s` | nullable float32 | t(P) − t(E). | Native car/ball frames, metadata timestamps |
| `subject_max_advance` | nullable float32 | Maximum toward displacement from the opening position across S..E, including zero at S. | Native car/ball frames, metadata timestamps |
| `subject_retreat_distance` | nullable float32 | Sum of max(−Δtoward, 0) across adjacent car positions S..E. | Native car/ball frames, metadata timestamps |
| `subject_path_distance` | nullable float32 | Sum of 3D distances between adjacent car positions S..E. | Native car/ball frames, metadata timestamps |
| `subject_mean_speed` | nullable float32 | Trapezoidal time-weighted mean of native car speed across S..E. | Native car/ball frames, metadata timestamps |
| `subject_max_speed` | nullable float32 | Maximum native 3D car speed across S..E. | Native car/ball frames, metadata timestamps |
| `subject_orientation_change` | nullable float32 | Sum of shortest 3D quaternion rotation angles, in radians, between adjacent native frames S..E. | Native car/ball frames, metadata timestamps |
| `subject_boost_used` | nullable float32 | Cumulative native boost_used at or before E minus its value at or before S, in native track units. | accumulation_tracks.boost_used |
| `subject_jump_active_s` | nullable float32 | Sum of each native frame duration whose left-hand jump_active flag is true, from S up to E. | Native car/ball frames, metadata timestamps |
| `subject_dodge_active_s` | nullable float32 | Sum of each native frame duration whose left-hand dodge_active flag is true, from S up to E. | Native car/ball frames, metadata timestamps |
| `subject_big_boost_pickup_count` | nullable uint32 | Count subject native boost_pickups with pad_type=big across S..E, independent of the collected amount. A native ambiguous pickup makes this count null. | boost_pickups.pad_type, frame, time, player_id |
| `subject_nearest_big_pad_closing_advantage_mean` | nullable float32 | At S select the nearest native Big pad by horizontal distance, breaking ties by native pad index. Across S..E take the trapezoidal mean of car velocity dotted with the horizontal unit car-to-pad vector minus car velocity dotted with the horizontal unit car-to-ball vector. Positive means greater closing speed toward that fixed pad than toward the ball. | boost_pads.size/index/position, native car and ball frames |
| `candidate_subject_velocity_toward` | nullable float32 | Subject native velocity at S, toward component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_subject_velocity_side` | nullable float32 | Subject native velocity at S, side component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_subject_velocity_up` | nullable float32 | Subject native velocity at S, up component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_subject_heading_toward` | nullable float32 | Subject forward unit vector from its normalized native quaternion at S, toward component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_subject_heading_side` | nullable float32 | Subject forward unit vector from its normalized native quaternion at S, side component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_subject_heading_up` | nullable float32 | Subject forward unit vector from its normalized native quaternion at S, up component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_subject_position_relative_ball_toward` | nullable float32 | Subject center minus ball center at S, toward component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_subject_position_relative_ball_side` | nullable float32 | Subject center minus ball center at S, side component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_subject_position_relative_ball_up` | nullable float32 | Subject center minus ball center at S, up component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_ball_position_toward` | nullable float32 | Ball center at S, toward component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_ball_position_side` | nullable float32 | Ball center at S, side component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_ball_position_up` | nullable float32 | Ball center at S, up component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_ball_velocity_toward` | nullable float32 | Ball native velocity at S, toward component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_ball_velocity_side` | nullable float32 | Ball native velocity at S, side component in team axes. | Native car/ball frames, metadata timestamps |
| `candidate_ball_velocity_up` | nullable float32 | Ball native velocity at S, up component in team axes. | Native car/ball frames, metadata timestamps |
| `subject_boost_before_pct` | nullable float32 | Latest boost_amount point at or before the native frame immediately preceding S in the same play, multiplied by 100/255. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `candidate_ball_own_goal_distance` | nullable float32 | Ball-surface distance to the nearest point of the own goal opening at S, floored at zero on overlap. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `closest_opponent_ball_distance` | nullable float32 | Minimum 3D opponent-center to ball-center distance at S. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `closest_opponent_closing_speed` | nullable float32 | Selected opponent velocity dotted with its unit car-to-ball vector at S. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `closest_opponent_boost_before_pct` | nullable float32 | Selected opponent boost before S, using the subject boost calculation. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `closest_teammate_ball_distance` | nullable float32 | Minimum 3D other-teammate-center to ball-center distance at S. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `closest_teammate_closing_speed` | nullable float32 | Selected other teammate velocity dotted with its unit car-to-ball vector at S. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `closest_teammate_boost_before_pct` | nullable float32 | Selected other teammate boost before S, using the subject boost calculation. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `subject_depth_lead` | nullable float32 | Minimum other-teammate toward position at S minus subject toward position at S. Positive means the subject is deeper than every active other teammate. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `teammate_goal_distance_min` | nullable float32 | Minimum other-teammate car-center distance to the own goal opening at S. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `approaching_teammate_count` | nullable uint32 | Count active other teammates whose native velocity dotted with their car-to-ball vector is positive at S. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `approaching_opponent_count` | nullable uint32 | Count active opponents meeting the same approach test at S. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `goal_side_teammate_count` | nullable uint32 | Count active other teammates whose toward coordinate is no greater than the ball toward coordinate at S. | replay_meta, player_activity, native frames; boost_amount tracks for boost |
| `opponent_touch_ball_velocity_delta_toward` | nullable float32 | Native ball velocity at C minus velocity in the immediately preceding frame in the same play, toward component in team axes. | touch, native ball frames |
| `opponent_touch_ball_velocity_delta_side` | nullable float32 | Native ball velocity at C minus velocity in the immediately preceding frame in the same play, side component in team axes. | touch, native ball frames |
| `opponent_touch_ball_velocity_delta_up` | nullable float32 | Native ball velocity at C minus velocity in the immediately preceding frame in the same play, up component in team axes. | touch, native ball frames |
| `opponent_touch_incoming_ball_speed` | nullable float32 | Length of native ball velocity in the frame immediately before C in the same play. | Native frames, player_activity; touch for C |
| `opponent_touch_outgoing_ball_speed` | nullable float32 | Length of native ball velocity at C. | Native frames, player_activity; touch for C |
| `subject_ball_distance_at_opponent_touch` | nullable float32 | 3D subject-center to ball-center distance at C. | Native frames, player_activity; touch for C |
| `min_subject_ball_distance` | nullable float32 | Minimum 3D subject-center to ball-center distance across S..P. | Native frames, player_activity; touch for C |
| `closest_approach_delay_s` | nullable float32 | t(M) − t(S). | Native frames, player_activity; touch for C |
| `subject_height_at_closest_approach` | nullable float32 | Subject car-center height at M. | Native frames, player_activity; touch for C |
| `subject_heading_alignment_at_closest_approach` | nullable float32 | Normalized subject forward vector dotted with its unit car-to-ball vector at M; coincident centers leave it null. | Native frames, player_activity; touch for C |
| `opponent_ball_distance_min` | nullable float32 | Minimum across S..P of the closest active opponent-center to ball-center distance. | Native frames, player_activity; touch for C |
| `opponent_ball_distance_mean` | nullable float32 | Trapezoidal time-weighted mean across S..P of the closest active opponent-center to ball-center distance. | Native frames, player_activity; touch for C |
| `other_teammate_ball_distance_min` | nullable float32 | Minimum across S..P of the closest active other-teammate-center to ball-center distance. | Native frames, player_activity; touch for C |
| `other_teammate_ball_distance_mean` | nullable float32 | Trapezoidal time-weighted mean across S..P of the closest active other-teammate-center to ball-center distance. | Native frames, player_activity; touch for C |
| `peak_ball_height` | nullable float32 | Maximum ball-center height across S..E. | Native frames, player_activity; touch for C |
| `minimum_own_goal_ball_distance` | nullable float32 | Minimum ball-surface distance to the own goal opening across C..P. | Native frames, player_activity; touch for C |
| `maximum_goalward_ball_speed` | nullable float32 | Maximum negative toward component of native ball velocity across C..P. Negative values mean every observed velocity remained upfield. | Native frames, player_activity; touch for C |
| `opponent_touch_count` | uint32 | Count distinct opponent native touch IDs across S..P. | touch |
| `subject_whiff_count` | uint32 | Count subject whiff events anchored in S..P with kind=whiff and resolved_time no later than P. | whiff |
| `subject_beaten_count` | uint32 | Count native beaten_to_ball events naming the subject as beaten, anchored in S..P. | beaten_to_ball |
| `subject_team_bump_count` | uint32 | Count received native teammate bumps anchored in S..P. | bump |
| `subject_opponent_bump_count` | uint32 | Count received native opponent bumps anchored in S..P. | bump |
| `subject_demo_inflicted_count` | uint32 | Count native demolitions naming the subject as attacker, anchored in S..P. | demolition |
| `opponent_shot_before_own_touch` | bool | True if either native touch action=shot or timeline kind=Shot names the opponent team from S strictly before the first own-team contact after S; stop at play end if there is none. | touch, timeline |
| `subject_bump_max_impulse` | nullable float32 | Maximum native victim_impulse among received bumps in S..P; ties select the earliest frame, then the lexicographically smallest event ID. | bump |
| `subject_bump_upright_after` | nullable float32 | World-up component of the subject car up vector at the selected bump’s sampled after frame. | bump, native car frames |
| `subject_bump_delay_to_closest_approach` | nullable float32 | t(M) minus the strongest received bump timestamp. | bump, metadata timestamps |
| `subject_bump_velocity_delta_toward` | nullable float32 | Subject after-frame minus before-frame native velocity for the strongest received bump, toward component in team axes. | Shared bump acceleration-peak-then-minimum sampler, native car frames |
| `subject_bump_velocity_delta_side` | nullable float32 | Subject after-frame minus before-frame native velocity for the strongest received bump, side component in team axes. | Shared bump acceleration-peak-then-minimum sampler, native car frames |
| `subject_bump_velocity_delta_up` | nullable float32 | Subject after-frame minus before-frame native velocity for the strongest received bump, up component in team axes. | Shared bump acceleration-peak-then-minimum sampler, native car frames |
| `first_touch_self` | nullable bool | True when the subject is among the simultaneous touchers at the first touch stage after P; null if the stage is absent. | touch |
| `first_touch_teammate` | nullable bool | True when another own-team player is among the simultaneous touchers at the first touch stage after P; null if the stage is absent. | touch |
| `first_touch_opponent` | nullable bool | True when an opponent is among the simultaneous touchers at the first touch stage after P; null if the stage is absent. | touch |
| `first_touch_delay_s` | nullable float32 | Native first touch stage time minus t(P). | touch, metadata timestamps |
| `second_touch_self` | nullable bool | True when the subject is among the simultaneous touchers at the second touch stage after P; null if the stage is absent. | touch |
| `second_touch_teammate` | nullable bool | True when another own-team player is among the simultaneous touchers at the second touch stage after P; null if the stage is absent. | touch |
| `second_touch_opponent` | nullable bool | True when an opponent is among the simultaneous touchers at the second touch stage after P; null if the stage is absent. | touch |
| `second_touch_delay_s` | nullable float32 | Native second touch stage time minus t(P). | touch, metadata timestamps |
| `previous_subject_touch_possession_before_self` | nullable bool | Immediately before T, true if the native possession span identifies the subject as possessor; false for observed neutral, opponent, or teammate possession. A missing span or an unidentified own-team possessor leaves the value null. Query spans with start_time < t(T) ≤ end_time. | possession, touch |
| `previous_subject_touch_followed_by_opponent` | nullable bool | True if the first distinct native contact frame strictly after T, through play end, includes any opponent. Retain simultaneous touchers. False when that stage contains only own-team players; null when T or the following stage is absent. | touch |
| `subject_ball_withdrawal_distance_before_opponent_touch` | nullable float32 | Sum max(0, −car_velocity_xy · horizontal_unit(ball_position − car_position)) × native frame duration for left frames i with S ≤ i < min(C, E). | Native car/ball frames, metadata timestamps, touch for C |
| `subject_ball_withdrawal_distance_after_opponent_touch` | nullable float32 | The same left-frame weighted sum for max(C, S) ≤ i < E. C later than E gives an observed empty window and zero. | Native car/ball frames, metadata timestamps, touch for C |
| `previous_subject_touch_delay_s` | nullable float32 | t(S) minus the subject’s preceding contact-stage time in the same play. | touch |
| `next_goal_own_team` | bool | True if the goal ending this play was scored by the own team; false if no such goal exists. | goal_context |
| `next_goal_opponent` | bool | True if the goal ending this play was scored by the opponent; false if no such goal exists. | goal_context |
| `next_goal_delay_s` | nullable float32 | Next goal time in the same play minus t(P); null if there is no goal. | goal_context |
| `touches_before_next_goal` | nullable uint32 | Count distinct native touch IDs strictly between P and the next goal; null if there is no goal. | touch, goal_context |
| `approach_min_subject_ball_distance` | nullable float32 | Subject-center to ball-center distance at A. | Native car and ball frames |
| `approach_subject_heading_alignment` | nullable float32 | Subject forward unit vector dotted with its unit car-to-ball vector at A. | Native car quaternion and positions |
| `approach_subject_ball_closing_speed` | nullable float32 | Subject velocity dotted with its unit car-to-ball vector at A. | Native car velocity and positions |
| `approach_closest_opponent_ball_distance` | nullable float32 | Closest active opponent-center to ball-center distance at A. | Native roster, activity and positions |
| `approach_closest_opponent_ball_closing_speed` | nullable float32 | Selected opponent velocity dotted with its unit car-to-ball vector at A. | Native roster, activity, positions and velocity |
| `approach_goal_side_teammate_count` | nullable uint32 | Number of active other teammates on the goal side of the ball at A. | Native roster, activity and positions |
| `approach_closest_goal_side_teammate_ball_distance` | nullable float32 | Closest active goal-side other-teammate-center to ball-center distance at A. | Native roster, activity and positions |
| `approach_closest_goal_side_teammate_ball_closing_speed` | nullable float32 | The same selected goal-side teammate velocity dotted with its unit car-to-ball vector at A. | Native roster, activity, positions and velocity |
| `passage_goal_side_teammate_count` | nullable uint32 | Number of active other teammates on the goal side of the ball at P. | Native roster, activity and positions |
| `passage_closest_goal_side_teammate_ball_distance` | nullable float32 | Closest active goal-side other-teammate-center to ball-center distance at P. | Native roster, activity and positions |
| `passage_closest_goal_side_teammate_ball_closing_speed` | nullable float32 | The same selected goal-side teammate velocity dotted with its unit car-to-ball vector at P. | Native roster, activity, positions and velocity |
| `last_opponent_touch_delay_from_approach_minimum_s` | nullable float32 | t(C) − t(A), retaining the sign; null without C or A. | Native touch and metadata timestamps |
| `approach_ball_directed_at_own_goal` | nullable bool | Whether the instantaneous native ball-velocity ray at A intersects the own goal opening, including ball radius. | Shared goal-opening intersection, native ball position and velocity |
| `approach_ball_own_goal_distance` | nullable float32 | Ball-surface distance to the nearest point of the own goal opening at A. | Shared goal-opening distance, native ball position |
| `approach_opponent_possession_fraction` | nullable float32 | Exact duration of opponent native possession intersecting S..E, divided by t(E) − t(S). | Shared possession-duration integration, native possession spans |

## Evidence

| Field | Meaning | Source |
| --- | --- | --- |
| `start_frame` | First qualifying native frame. | Native frames and opening conditions |
| `end_frame` | First passage or retreat finish frame. | Native ball and car positions, car velocity, dodge state |
| `start_time` | Native opening timestamp. | `metadata_frames` |
| `end_time` | Native approach-finish timestamp. | `metadata_frames` |
| `passage_frame` | Frame where the ball passes behind the player, confirming failure. | Native ball and car positions |
| `passage_time` | Native passage timestamp. | `metadata_frames` |
| `player` | Judged player identity. | Native roster |
| `is_team_0` | Whether the player belongs to blue. | Native roster |
| `play_start_frame` | First native frame of the live-play segment. | `player_activity` |
| `play_end_frame` | Final frame used for the live-play segment. | `player_activity`, `goal_context` |
| `finish` | `ball_passage` or `retreat_to_opening_depth`. | Native positions, car velocity, dodge state |
| `opening.position_y` | Player's Y position when the attempt opens. | Native car position |
| `opening.deepest_by_y` | Whether the deepest-by-Y opening branch qualifies. | Active teammate positions |
| `opening.farthest_goal_side` | Whether the farthest-goal-side opening branch qualifies. | Active teammate and ball positions |
| `opening.last_touch_ids` | All events at the latest contact frame at opening. | `touch` |
| `opponent_touch_ids` | Opponent touches from opening through passage. | `touch` |
| `opponent_shot_ids` | Opponent shots before the own-team-touch cutoff. | `touch`, `timeline` |
| `next_own_touch_frame` | Next strictly later own-team contact, or null before play end. | `touch` |

Feature sample references also retain the approach-specific frame and selected players:

| Sample field | Meaning | Source |
| --- | --- | --- |
| `approach_closest_approach_frame` | A, the first minimum-distance frame within S..E, or null for incomplete approach geometry. | Native subject and ball positions |
| `approach_closest_approach_time` | Native timestamp at A, or null when A is unavailable. | `metadata_frames` |
| `approach_closest_opponent_id` | Selected closest active opponent at A, or null for an absent or unknown profile. | Native roster, activity and positions |
| `approach_closest_goal_side_teammate_id` | Selected closest active goal-side other teammate at A, or null for an empty or unknown profile. | Native roster, activity and positions |
| `passage_closest_goal_side_teammate_id` | Selected closest active goal-side other teammate at P, or null for an empty or unknown profile. | Native roster, activity and positions |
