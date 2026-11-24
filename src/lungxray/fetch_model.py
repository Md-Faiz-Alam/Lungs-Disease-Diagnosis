"""Download the production model and verify its SHA-256 (used by the Docker build).

MODEL_URL=... MODEL_SHA256=... python -m lungxray.fetch_model [destination]
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_sha256: str = "") -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".download")
    urllib.request.urlretrieve(url, tmp)
    digest = sha256(tmp)
    if expected_sha256 and digest != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Model checksum mismatch: expected {expected_sha256}, got {digest}")
    tmp.replace(dest)
    return digest


def main() -> None:
    url = os.environ.get("MODEL_URL", "")
    if not url:
        sys.exit("MODEL_URL is not set")
    dest = Path(
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_PATH", "models/production/model.pt")
    )
    print("model sha256", fetch(url, dest, os.environ.get("MODEL_SHA256", "")))


if __name__ == "__main__":
    main()
