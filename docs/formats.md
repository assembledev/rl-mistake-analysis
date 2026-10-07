# Data contracts

These formats connect replay extraction, human review, training, and inference. A **mistake kind** defines which incidents to consider and which measurements its models need. A **feature contract** identifies those measurements and their order; models and candidate extraction must agree on it.

| Data | Format | Used for |
| --- | --- | --- |
| Replay events | JSON | Input to candidate extraction |
| Candidates | JSON | Incidents for review or scoring |
| Human annotations | JSON | Reviewed incidents and recording references |
| Training table | Parquet | Features and labels for model fitting |
| Model catalog | `manifest.json` | Available releases and required artifacts |
| Model | ONNX | Classifier computation and learned parameters |

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

Candidate extraction selects incidents worth reviewing and calculates model features. Results are grouped by recording and mistake kind into a **batch**.

| Batch field | Meaning |
| --- | --- |
| `kind` | Mistake kind |
| `input_schema` | Feature contract identifier |
| `feature_names` | Ordered model input feature names |
| `engine_revision` | Supplier revision that produced the events |
| `replay_sha256` | Checksum of the original recording bytes |
| `candidates` | Selected incidents |

Each candidate records its `kind`, original `frame` and `raw_time`, subject `player_id`, numeric `features`, original event `evidence`, and `source_event_ids`. Features follow the batch's declared order. Evidence preserves the supplier information needed to identify and inspect the incident.

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

The `id` inside the `download` object is a Ballchasing identifier, not a filename or website address. The fetch tool uses it to request the recording from Ballchasing. The header ID identifies the recording independently of its host; the checksum verifies the exact downloaded bytes.

| Annotation field | Meaning |
| --- | --- |
| `replay_sha256` | Checksum of a recording listed in `replays` |
| `kind` | Mistake kind |
| Kind-specific identity fields | Fields required to identify an incident for this mistake kind |
| `raw_time` | Original event time |
| `evidence` | Original supplier event payload |
| `label` | Human decision: `0` or `1` |
| `reject_reason` | Optional reviewer rejection reason |
| `engine_revision` | Supplier revision that produced the reviewed event |
| `source_event_ids` | Optional IDs of original supplier events |

To retrieve an annotation's recording, find the entry in `replays` with the same checksum, then use that entry's download source and replay ID. The recording reference is stored once and shared by every annotation for that recording.

The fetch tool saves the original binary file unchanged. Its local filename is the recording's 64-character checksum followed by `.replay`. Without a download reference, supply the original file with that name in the preparation directory.

Within a recording, each kind defines the fields that identify an incident. Identity, time, and evidence must agree; duplicate incidents are errors. Preparation uses these identity fields to match labels with extracted incidents. Source event IDs are provenance, rather than the matching key.

## Training tables

Preparation combines human labels with recalculated features. One Parquet table contains one mistake kind, with one incident per row:

- Recording checksum and kind-specific incident identity.
- `raw_time`, binary `label`, and optional `reject_reason`.
- `engine_revision`, `annotation_engine_revision`, and `source_event_ids`.
- A numeric column for each feature.

The two revision columns distinguish preparation's supplier code from the code used when the incident was reviewed. File metadata named `rl_mistake_analysis` contains `schema_version: 1`, `kind`, `input_schema`, ordered `feature_names`, and `replays`. Readers use this metadata to select model inputs in the correct order. All selected annotations must match before the table is saved.

## Models and catalog

ONNX contains the classifier's operations, learned parameters, and exported preprocessing. `manifest.json` describes which releases exist and how to use them. It contains `schema_version: 1`, catalog `version`, and a `models` list.

| Model entry | Meaning |
| --- | --- |
| `kind` | Mistake kind |
| `version` | Model release version |
| `input_schema` | Required feature contract identifier |
| `feature_names` | Required feature names in model input order |
| `runtime` | `onnx` |
| `model_file` | Relative path to the ONNX graph |
| `evaluation_file` | Relative path to the evaluation report |
| `keep_threshold` | Probability threshold for retaining an incident |
| `status` | `experimental` or `approved` |
| `provenance` | Training `source_revision` and supplier `engine_revisions` |
| `files` | Required artifacts, each with `path` and integrity checksum `sha256` |

Several releases can coexist. Each must declare its graph, evaluation report, and any external weight files. Consumers download the selected release's files; checksums verify them. Feature compatibility depends on the contract, rather than an exact supplier revision. Change the contract identifier for incompatible changes to feature meaning, units, or order.

The ONNX interface accepts one float32 table `[N, F]` and returns probabilities `[N, 1]`: `N` incidents, `F` ordered features, one probability in `[0, 1]` per incident. The incident count is dynamic.

## Predictions

A prediction batch carries the recording and feature-contract metadata, a `model` object containing `kind`, `version`, and `sha256`, and a `predictions` list. Each prediction contains the original `candidate`, probability `score`, and boolean `keep`. An incident is retained when its score meets `keep_threshold`. Stored predictions identify the model that produced them.

## Example: teammate bump

`bumping_teammate` identifies an incident by `frame`, `initiator`, and `victim` within a recording. The initiator is the subject; opponent contacts are excluded. The candidate library calculates the following eight model inputs, in the order defined by `bumping_teammate.native.v1`. These values appear in each candidate's `features`, become the numeric feature columns in prepared training tables, and are supplied to the model during inference:

1. `contact_confidence`
2. `closing_speed`
3. `victim_impulse`
4. `contact_distance`
5. `strength`
6. `initiator_height`
7. `victim_height`
8. `victim_distance_to_own_goal`

The first five copy measurements from the supplier's bump event (`confidence` becomes `contact_confidence`). The last three are calculated by this project from the supplier's player positions and team information: two heights and the victim's distance to their own goal. The complete vector is this project's feature contract, rather than the raw supplier event.
