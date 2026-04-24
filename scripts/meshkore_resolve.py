"""Resolve cluster spec + credentials paths (.mechcore preferred, .meshkore legacy)."""
from __future__ import annotations

from pathlib import Path


def cluster_spec_path(root: Path) -> Path:
    me, mk = root / ".mechcore", root / ".meshkore"
    if me.exists():
        return me
    if mk.exists():
        return mk
    raise FileNotFoundError(
        "Missing `.mechcore` or `.meshkore` at repo root (cluster invite + channel_id)."
    )


def credentials_path(root: Path) -> Path | None:
    a, b = root / ".mechcore.local", root / ".meshkore.local"
    if a.exists():
        return a
    if b.exists():
        return b
    return None


def credentials_write_path(root: Path) -> Path:
    """Where `meshkore-join.sh` should write tokens (matches chosen spec file)."""
    return root / ".mechcore.local" if (root / ".mechcore").exists() else root / ".meshkore.local"
