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
    results = [
        run(engine, "candidates", kind, document=document)
        for kind in ("bumping_teammate", "bang_with_time")
    ]
    for result in results:
        assert result.returncode == 0, result.stderr
    bump, bang = [json.loads(result.stdout)[0] for result in results]
    assert not bump["candidates"]
    assert [candidate["frame"] for candidate in bang["candidates"]] == [1, 3]


def test_all_candidates_requires_native_frames_and_matches_each_kind(
    engine, combined_analysis
):
    document = json.loads(combined_analysis.read_text())
    result = run(engine, "candidates", "--all", document=document)
    assert result.returncode == 0, result.stderr
    batches = json.loads(result.stdout)
    assert [batch["kind"] for batch in batches] == [
        "bumping_teammate",
        "bang_with_time",
        "overcommitting",
    ]
    for batch in batches:
        selected = run(engine, "candidates", batch["kind"], document=document)
        assert selected.returncode == 0, selected.stderr
        assert json.loads(selected.stdout) == [batch]
    del document["outputs"]["replay_data"]
    for kind in ("overcommitting", "--all"):
        result = run(engine, "candidates", kind, document=document)
        assert result.returncode == 1
        assert not result.stdout
        assert "replay_data" in result.stderr


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
