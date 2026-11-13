"""Compute native overcommitting measurements for exact replay intervals."""

import json
from pathlib import Path

from training.context_features import compute as compute_context
from training.context_features import main


def compute(paths, engine, sampling, output, *, candidate_batches=None):
    """Map each analysis path to its exact saved candidate batch when supplied."""
    paths = [Path(path) for path in paths]
    command = ["overcommitting-features", str(sampling)]
    if candidate_batches is not None:
        batches = {
            Path(path).absolute(): Path(batch)
            for path, batch in candidate_batches.items()
        }
        if len(batches) != len(candidate_batches) or set(batches) != {
            path.absolute() for path in paths
        }:
            raise ValueError(
                "Candidate batches must cover exactly the selected analyses"
            )

        def command_for_path(path):
            return [*command, "--candidates", str(batches[path.absolute()])]

        command_arguments = command_for_path
    else:
        command_arguments = command
    return compute_context(
        paths,
        engine,
        output,
        kind="overcommitting",
        command=command_arguments,
        expected_sampling=json.loads(Path(sampling).read_text()),
        participant_fields=(),
        integer_fields=("end_frame", "passage_frame"),
        source_id_field="source_event_ids",
        multiple_source_ids=True,
        incident_fields=("frame", "end_frame"),
    )


if __name__ == "__main__":
    main(compute, __doc__)
