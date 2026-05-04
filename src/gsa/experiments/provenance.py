"""Capture run provenance: git, environment, hardware."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel


class Provenance(BaseModel):
    git_commit: str
    git_dirty: bool | None
    python_version: str
    platform: str
    cpu_count: int
    env_hash: str
    hardware_fingerprint: str


def _git_commit() -> tuple[str, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
        dirty = len(status.strip()) > 0
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return "unknown", None


def _env_hash() -> str:
    """Hash of installed package versions, normalized for cross-machine stability.

    The fingerprint is stable across machines for the same set of pinned
    packages, but does not capture editable installs (lines starting with
    ``-e ``), which reference machine-specific paths and are therefore
    stripped before hashing. Remaining lines are sorted to guard against
    incidental ordering differences from ``pip freeze``.
    """
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True, stderr=subprocess.DEVNULL, timeout=30,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        out = ""
    # Drop editable-install lines (machine-specific paths) and sort for determinism.
    lines = sorted(
        line.strip() for line in out.splitlines()
        if line.strip() and not line.startswith("-e ")
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _hardware_fingerprint() -> str:
    import os
    parts = [
        platform.processor(),
        platform.machine(),
        str(os.cpu_count() or 0),
        platform.system(),
        platform.release(),
    ]
    return " | ".join(parts)


def capture_provenance() -> Provenance:
    commit, dirty = _git_commit()
    return Provenance(
        git_commit=commit,
        git_dirty=dirty,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=__import__("os").cpu_count() or 1,
        env_hash=_env_hash(),
        hardware_fingerprint=_hardware_fingerprint(),
    )
