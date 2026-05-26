"""Shared result-IO helpers for the bench/straggler/sweep scripts.

Results are appended (never overwritten) to a JSON list so repeated runs of the
same config accumulate and can be averaged / charted later. Each record carries a
git commit + dirty flag + an explicit date and ISO-8601 timestamp for
reproducibility — there was no existing helper for this (each script previously
rolled its own json.dumps, e.g. train_e2.py).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

# Bench/sweep/straggler output lives alongside the scripts (python/scripts/runs),
# NOT the top-level runs/ — that one is reserved for RL training artifacts.
# Anchored to this file so the default is the same regardless of cwd.
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def git_meta() -> dict:
    """Capture reproducibility metadata: git commit, dirty flag, date, timestamp.

    Degrades gracefully (None / "") outside a git repo or if git is unavailable.
    """
    now = datetime.now().astimezone()

    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": now.isoformat(timespec="seconds"),
    }


def append_result(path: str | Path, record: dict[str, Any]) -> Path:
    """Append `record` to the JSON list at `path` (creating it + parents).

    Reads the existing list (tolerating a missing/corrupt file → []), appends,
    and rewrites with indent=2. Returns the resolved path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, list):
                existing = loaded
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2))
    return path
