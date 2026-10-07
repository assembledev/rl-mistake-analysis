import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def engine():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "rl-mistake-analysis-cli",
            "--message-format=json",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if message.get("target", {}).get(
            "name"
        ) == "rl-mistake-analysis" and message.get("executable"):
            return Path(message["executable"])
    raise AssertionError("CLI build produced no executable")
