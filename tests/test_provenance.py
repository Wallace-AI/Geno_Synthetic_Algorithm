import re
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
