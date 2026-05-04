import hashlib
import re
import subprocess

from gsa.experiments.provenance import (
    capture_provenance,
    Provenance,
)


def test_capture_provenance_returns_required_fields():
    p = capture_provenance()
    assert isinstance(p, Provenance)
    assert re.fullmatch(r"[a-f0-9]{40}|unknown", p.git_commit)
    assert p.git_dirty in (True, False, None)
    assert p.python_version
    assert p.platform
    assert p.cpu_count > 0
    assert p.env_hash and len(p.env_hash) == 64  # sha256 hex


def test_provenance_dict_serializable():
    p = capture_provenance()
    d = p.model_dump()
    assert "git_commit" in d
    assert "env_hash" in d


def test_capture_provenance_when_git_unavailable(monkeypatch):
    """When git binary isn't on PATH, git_commit should be 'unknown' and git_dirty None."""
    real_check_output = subprocess.check_output

    def fake_check_output(cmd, *args, **kwargs):
        if cmd[0] == "git":
            raise FileNotFoundError("git not found")
        return real_check_output(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    p = capture_provenance()
    assert p.git_commit == "unknown"
    assert p.git_dirty is None


def test_env_hash_when_pip_freeze_fails(monkeypatch):
    """When pip freeze fails, env_hash should still be a 64-char SHA256."""
    real_check_output = subprocess.check_output

    def fake_check_output(cmd, *args, **kwargs):
        if "pip" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return real_check_output(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    p = capture_provenance()
    assert len(p.env_hash) == 64
    # SHA256 of empty string after sort/filter -> sha256("") since no lines remain
    assert p.env_hash == hashlib.sha256(b"").hexdigest()


def test_env_hash_strips_editable_installs(monkeypatch):
    """env_hash should be stable across editable-install path differences."""
    def fake_check_output_a(cmd, *args, **kwargs):
        if "pip" in cmd:
            return "numpy==1.26.4\n-e file:///D:/repo-A\nscipy==1.11.4\n"
        raise FileNotFoundError

    def fake_check_output_b(cmd, *args, **kwargs):
        if "pip" in cmd:
            return "scipy==1.11.4\n-e file:///C:/repo-B\nnumpy==1.26.4\n"
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "check_output", fake_check_output_a)
    p_a = capture_provenance()

    monkeypatch.setattr(subprocess, "check_output", fake_check_output_b)
    p_b = capture_provenance()

    # Both runs should produce the same env_hash because we sort and strip -e lines
    assert p_a.env_hash == p_b.env_hash
