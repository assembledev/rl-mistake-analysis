import json
import subprocess

import pytest


def run(engine, *args, document):
    return subprocess.run(
        [str(engine), *map(str, args)],
        input=json.dumps(document),
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("combined", [False, True])
def test_native_events_determine_candidates_in_both_formats(engine, combined):
    events = [
        {
            "meta": {"stream": "touch", "id": f"touch:{i}"},
            "payload": {
                "kind": "touch",
                "payload": {
                    "touch_id": i,
                    "frame": i,
                    "time": float(i),
                    "player": {"Epic": player},
                    "is_team_0": team_zero,
                    "tags": [],
                },
            },
        }
        for i, (player, team_zero) in enumerate(
            [("a", True), ("a", True), ("b", False), ("b", False), ("a", True)]
        )
    ]
    document = {
        "name": "subtr-actor",
        "schema_version": 1,
        "status": "ok",
        "revision": "supplier",
        "replay_sha256": "a" * 64,
        "events": events,
    }
    if combined:
        document["outputs"] = {"events": {"events": {"events": document.pop("events")}}}
    batches = run(engine, "candidates", "--all", document=document)
    assert batches.returncode == 0, batches.stderr
    bump, bang = json.loads(batches.stdout)
    assert not bump["candidates"]
    assert [candidate["frame"] for candidate in bang["candidates"]] == [1, 3]
    for batch in (bump, bang):
        selected = run(engine, "candidates", batch["kind"], document=document)
        assert selected.returncode == 0, selected.stderr
        assert json.loads(selected.stdout) == [batch]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ((), "candidates requires KIND or --all"),
        (("--all", "bumping_teammate"), "candidates requires KIND or --all"),
        (("unknown",), "unknown mistake kind"),
    ],
)
def test_invalid_candidate_scope_fails_before_reading_input(engine, options, message):
    result = run(engine, "candidates", *options, document={})
    assert result.returncode == 1
    assert not result.stdout
    assert message in result.stderr
