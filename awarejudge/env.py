"""Tiny .env loader (standard library only). Reads KEY=VALUE lines from a .env file in the current
folder or the kit root, and sets them as environment variables unless already set. Never prints values."""
from __future__ import annotations

import os
from pathlib import Path


def load_env(paths: list[str | Path] | None = None) -> list[str]:
    """Return the names of variables loaded (never the values)."""
    here = Path(__file__).resolve().parent.parent
    candidates = [Path(p) for p in paths] if paths else [Path.cwd() / ".env", here / ".env"]
    loaded: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if key and val and key not in os.environ:
                os.environ[key] = val
                loaded.append(key)
    return loaded
