# Data contracts

These formats connect replay extraction, human review, training, and inference. A **mistake kind** defines which incidents to consider and which measurements its models need. A **feature contract** identifies those measurements and their order; models and candidate extraction must agree on it.

| Data | Format | Used for |
| --- | --- | --- |
| Selected replay outputs | JSON | Analysis timeline and structured frames for one recording |
| Replay events | JSON | Input to candidate extraction |
| Candidates | JSON | Incidents for review or scoring |
| Human annotations | JSON | Reviewed incidents and recording references |
| Feature table | Parquet | Typed measurements and extraction provenance |
| Training table | Parquet | Features and labels for model fitting |
| Model catalog | `manifest.json` | Available releases and required artifacts |
| Model | ONNX | Classifier computation and learned parameters |

## Selected replay outputs

`extract FILE.replay` exports one recording as one JSON document, with `events,replay_data` selected by default. `--outputs` accepts any comma-separated selection of the five output keys below. A selection computes only its requested outputs, with all selected collectors sharing one replay pass. Multiple recordings require separate extractions and separate JSON files.

The document carries the same identity fields as the replay-events format below: `name`, `schema_version: 1`, `status`, `revision`, `replay_sha256`, and optional `rocket_league_id`. An `outputs` object contains exactly the selected payloads:

| Output key | Payload |
| --- | --- |
| `events` | Full SubtrActor analysis event timeline: event records, lightweight frame context, replay metadata, summaries, and compressed numeric tracks, including observed boost amounts |
| `replay_data` | Full SubtrActor structured replay: player and ball frame data, replay metadata, available inputs/camera state, and basic replay event streams |
| `stats` | Accumulated statistics from the default modules, keyed by module name under `modules`, with `replay_meta` |
| `stats_timeline` | Full SubtrActor statistics snapshots at each replay frame, with configuration, metadata, and events |
| `features` | Numeric matrix as JSON rows under `matrix`, with SubtrActor's replay metadata and column headers under `metadata` |

Payloads preserve the supplier's field names and structure. Analysis event records are at `outputs.events.events.events`; boost tracks are at `outputs.events.accumulation_tracks`; structured frame data is at `outputs.replay_data.frame_data`.

The feature matrix uses `NDArrayCollector<f32>`'s default feature adders: ball rigid body globally, and each player's rigid body, boost, and jump state. `metadata.column_headers` describes global and per-player columns; `metadata.replay_meta` defines the player order. Each matrix row stores the global columns followed by the player columns in that order. This supplier matrix is separate from the mistake model's feature contract. Statistics use the supplier's default modules.

Candidate extraction and scoring accept this document when the `events` output is present, using its event records with the shared identity. The other selected payloads remain available for feature calculations that use additional context.

`outputs` lists the five implemented output keys. The five forms correspond to SubtrActor's [documented primary output layers](https://github.com/rlrml/subtr-actor/blob/40ec533da4184e46418b742804818a326b50b489/src/lib.rs#L30).

## Replay events

The extraction tool reads a binary recording and returns gameplay events from subtr-actor, together with the recording's identity. The Rust type is called `ReplayFacts`. This document is the input to candidate extraction.

| Field | Meaning |
| --- | --- |
| `name` | Event supplier; must be `subtr-actor` |
| `schema_version` | Export format version; supported value: `1` |
| `status` | Extraction result; must be `ok` (success) |
| `revision` | Supplier code that produced the events |
| `replay_sha256` | Checksum of the original recording bytes |
| `rocket_league_id` | Optional recording ID from its header |
| `events` | Gameplay event records |

Each event has a `meta` object and a `payload`. Inside `meta`, `stream` identifies the event category and `id` identifies that event. The `payload` contains its time, participants, and measurements.

## Candidates

A **candidate** is an event to review for one player. A **batch** contains the candidates for one replay and one mistake kind. Each kind defines which events to keep and how to identify them. Measurements for training and scoring are calculated separately.

| Batch field | Meaning |
| --- | --- |
| `kind` | Mistake kind |
| `input_schema` | Feature contract identifier |
| `feature_names` | Ordered incident measurement names |
| `engine_revision` | Supplier revision that produced the events |
| `replay_sha256` | Checksum of the original recording bytes |
| `candidates` | Selected incidents |

Each candidate contains the mistake `kind`, frame number and time (`frame`, `raw_time`), player being evaluated (`player_id`), measurements (`features`), original event data (`evidence`), and original event IDs (`source_event_ids`). Measurements follow the order in `feature_names`.

## Context measurements

Context extractors produce one row per selected incident and subject. A batch carries `input_schema`, `engine_revision`, `replay_sha256`, `sampling`, ordered `feature_definitions`, and `rows`.

Each feature definition declares a `name`, `data_type` (`boolean`, `f32`, or `u32`), `nullable`, and `source`. Rows contain the incident key, judged player, source event, timestamp, actual sample references, and a feature-name-to-value map. JSON `null` represents an unavailable observation.

The kind's base context schema defines the measurements and units. Models select unique known names in their declared order. A named selection uses `BASE_SCHEMA.NAME.v1`, where `NAME` contains letters, digits, or underscores. Changes to measurement meaning or units use a new base schema version.

## Human annotations

A review export records human decisions so features can be prepared later. It contains `schema_version: 1` and two lists:

- `replays`: recording identifiers and where to obtain the recordings.
- `annotations`: incident identities, labels, and original event evidence.

Each recording reference has the following structure. Replace the placeholders with values from the recording and hosting service:

```json
{
  "sha256": "<64-character checksum of the original recording>",
  "rocket_league_id": "<optional ID from the recording header>",
  "download": {
    "source": "ballchasing",
    "id": "<replay ID assigned by Ballchasing>"
  }
}
```

The `id` inside the `download` object is a Ballchasing identifier. The fetch tool uses it to request the recording from Ballchasing. The header ID identifies the recording independently of its host; the checksum verifies the exact downloaded bytes.

| Annotation field | Meaning |
| --- | --- |
| `replay_sha256` | Checksum of a recording listed in `replays` |
| `kind` | Mistake kind |
| Kind-specific identity fields | Fields required to identify an incident for this mistake kind |
| `player_id` | Required identity of the judged participant |
| `raw_time` | Original event time |
| `evidence` | Original supplier event payload |
| `label` | Human decision: `0` or `1` |
| `reject_reason` | Optional reviewer rejection reason |
| `engine_revision` | Supplier revision that produced the reviewed event |
| `source_event_ids` | Optional IDs of original supplier events |

To retrieve an annotation's recording, find the entry in `replays` with the same checksum, then use that entry's download source and replay ID. The recording reference is stored once and shared by every annotation for that recording.

The fetch tool saves the original binary file unchanged. Its local filename is the recording's 64-character checksum followed by `.replay`. Locally supplied recordings use the same filename in the preparation directory.

Within a recording, each kind defines the fields that identify an incident. The judged player, incident identity, time, and evidence must agree; duplicate reviews of the same participant in an incident are errors. Each subject can have a separate decision. Source event IDs preserve extraction provenance.

`validate-annotations` checks recording references, incident and player identities, evidence, labels, timestamps, and provenance.

## Feature and training tables

Preparation combines human labels with recalculated features. One Parquet table contains one mistake kind, with one incident and subject per row:

- Recording checksum and kind-specific incident identity.
- `raw_time`, binary `label`, and optional `reject_reason`.
- `engine_revision`, `annotation_engine_revision`, and `source_event_ids`.
- A typed column for each feature.

The two revision columns distinguish preparation's supplier code from the code used when the incident was reviewed. File metadata named `rl_mistake_analysis` contains `schema_version: 1`, `kind`, `input_schema`, ordered `feature_names`, and `replays`. Readers use this metadata to select model inputs in the correct order. All selected annotations must match before the table is saved.

Context tables use metadata named `rl_mistake_analysis_features`, with ordered typed `feature_definitions` and a `sampling` contract. Feature columns retain their boolean, float32 or uint32 types and declared nullability. Feature rows retain the kind-specific incident key, judged player, source event, and actual sample references. Annotation joins preserve the feature contract and column order.

The shared dataset API converts declared features into ordered float32 model inputs. Boolean values become `0` and `1`; nullable measurements become `NaN`. It returns labels and recording checksums alongside the input matrix for grouped training and evaluation.

## Models and catalog

ONNX contains the classifier's operations, learned parameters, and exported preprocessing. `manifest.json` describes which releases exist and how to use them. It contains `schema_version: 1`, catalog `version`, and a `models` list.

| Model entry | Meaning |
| --- | --- |
| `kind` | Mistake kind |
| `version` | Model release version |
| `input_schema` | Required feature contract identifier |
| `candidate_schema` | Candidate-selection contract used for training, when declared |
| `feature_names` | Required feature names in model input order |
| `sampling` | Settings for selecting measurement frames |
| `runtime` | `onnx` |
| `model_file` | Relative path to the ONNX graph |
| `evaluation_file` | Relative path to the evaluation report |
| `keep_threshold` | Probability threshold for retaining an incident |
| `status` | `experimental` or `approved` |
| `provenance` | Training `source_revision` and supplier `engine_revisions` |
| `files` | Required artifacts, each with `path` and integrity checksum `sha256` |

Several releases can coexist. Each must declare its graph, evaluation report, and any external weight files. Consumers download the selected release's files; checksums verify them. Feature compatibility depends on the contract. Context models may select any nonempty subset of known features in their declared `feature_names` order. Adding measurements to the extractor preserves existing selections. Change the base contract version for incompatible changes to feature meaning or units. See [context measurements](#context-measurements) for named selections.

Candidate and feature contracts are independent. `candidate_schema` records which incidents qualified for training; `input_schema` identifies their measurements. Inference validates the feature schema, ordered inputs and sampling settings. Catalog schema version 1 includes candidate metadata; consumers must use a library version that understands it.

The ONNX interface accepts one float32 table `[N, F]` and returns probabilities `[N, 1]`: `N` incidents, `F` ordered features, one probability in `[0, 1]` per incident. The incident count is dynamic. Model selection and release preparation are described in the [training guide](training.md).

## Predictions

A prediction batch records the candidate contract in `input_schema` and `feature_names`, the model contract in `model_input_schema` and `model_feature_names`, recording and extraction provenance, context `sampling` where applicable, model identity, and `predictions`.

Each prediction contains the original `candidate`, probability `score`, and threshold decision `keep`. The `model` identity contains `kind`, `version`, and a checksum of the selected release metadata.

## Mistake-specific contracts

[Teammate bump](mistakes/bumping_teammate.md) documents incident identity, candidate model features, context feature extraction, and frame selection.

[Bang with time](mistakes/bang_with_time.md) documents the touch identity key, candidate selection, context measurements and annotation joins.
