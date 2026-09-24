"""Last N scheduler steps, written by the scheduler and read over HTTP.

The API process and the scheduler process do not share memory. The scheduler
appends one record per forward; ``/admin/stats`` reads the file. The supervisor
copies the same file into ``logs/verify-report.json`` while the server under
test is on the spare port.
"""

from __future__ import annotations

import json
from pathlib import Path

RING_LEN = 32


def ring_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / "logs" / "step-ring.json"


def append_step(record: dict, root: Path | None = None) -> None:
    path = ring_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps: list = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict) and isinstance(loaded.get("steps"), list):
                steps = loaded["steps"]
        except (OSError, json.JSONDecodeError):
            steps = []
    steps.append(record)
    path.write_text(json.dumps({"steps": steps[-RING_LEN:]}) + "\n")


def read_ring(root: Path | None = None) -> list:
    path = ring_path(root)
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    steps = loaded.get("steps") if isinstance(loaded, dict) else None
    return steps if isinstance(steps, list) else []
