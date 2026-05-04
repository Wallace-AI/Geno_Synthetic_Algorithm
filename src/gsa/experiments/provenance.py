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
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        dirty = len(status.strip()) > 0
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", None


def _env_hash() -> str:
    """SHA256 of installed package versions for reproducibility."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
    except subprocess.CalledProcessError:
        out = ""
    return hashlib.sha256(out.encode()).hexdigest()


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
