"""Download referenced recordings and verify their exact file checksums."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Replay download redirected", headers, fp)


def fetch(dataset_path, directory, token):
    document = json.loads(Path(dataset_path).read_bytes())
    if document.get("schema_version") != 1 or not document.get("replays"):
        raise ValueError("Invalid annotation schema or empty replay references")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    references = document["replays"]
    seen = set()
    for reference in references:
        checksum = reference.get("sha256")
        if (
            not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
            or checksum in seen
        ):
            raise ValueError("Invalid or duplicate replay fingerprint")
        seen.add(checksum)

    opener = build_opener(NoRedirects())
    downloaded = 0
    last_request = None
    for reference in references:
        checksum = reference["sha256"]
        target = directory / f"{checksum}.replay"
        if target.exists():
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                    raise ValueError(f"Replay hash mismatch: {target}")
            continue
        source = reference.get("download")
        if not isinstance(source, dict) or source.get("source") != "ballchasing":
            raise ValueError(f"Missing supported download reference for {checksum}")
        replay_id = source.get("id")
        if not isinstance(replay_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}", replay_id
        ):
            raise ValueError("Invalid Ballchasing replay ID")
        if not token or not token.strip():
            raise ValueError("BALLCHASING_TOKEN is required to download recordings")
        request = Request(
            f"https://ballchasing.com/api/replays/{replay_id}/file",
            headers={"Authorization": token},
        )
        if last_request is not None:
            delay = 1.0 - (time.monotonic() - last_request)
            if delay > 0:
                time.sleep(delay)
        last_request = time.monotonic()
        temporary = None
        try:
            digest = hashlib.sha256()
            with (
                opener.open(request, timeout=60) as response,
                tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream,
            ):
                temporary = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if digest.hexdigest() != checksum:
                raise ValueError(f"Downloaded replay hash mismatch: {replay_id}")
            os.replace(temporary, target)
            downloaded += 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return downloaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        count = fetch(
            args.annotations, args.directory, os.environ.get("BALLCHASING_TOKEN")
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({"downloaded": count}))


if __name__ == "__main__":
    main()
