"""Centralized dotenv loader to keep CLI, UI and subprocesses in sync."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

from dotenv import dotenv_values

_ENV_LOADED = False
_ENV_PATH: Optional[Path] = None


_DEF_NAMES = (".env", ".env.example")


def _iter_candidates(names=_DEF_NAMES, base_dir: Optional[Path] = None, extra: Iterable[Path] = ()):  # type: ignore[misc]
    bases = []
    if base_dir:
        bases.append(Path(base_dir))
    module_dir = Path(__file__).resolve().parent
    bases.extend([module_dir, module_dir.parent])
    try:
        bases.append(Path.cwd())
    except Exception:
        pass
    try:
        bases.append(Path.home())
    except Exception:
        pass
    for p in extra:
        bases.append(Path(p))

    seen = set()
    for base in bases:
        if not base:
            continue
        for name in names:
            candidate = (base / name).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def _find_first(name: str, base_dir: Optional[Path] = None, extra: Iterable[Path] = ()) -> Optional[Path]:
    """Return first existing env file with the given name."""

    env_override = os.getenv("ENV_FILE")
    if env_override and name == ".env":
        candidate = Path(env_override).expanduser().resolve()
        if candidate.is_file():
            return candidate

    for candidate in _iter_candidates((name,), base_dir, extra):
        if candidate.is_file():
            return candidate
    return None


def load_env(base_dir: Optional[Path] = None, override: bool = False, extra: Iterable[Path] = ()) -> Optional[Path]:
    """Load environment variables with consistent priority.

    Priority (highest → lowest):
    1. Existing ``os.environ`` values
    2. Values from ``.env`` (if present)
    3. Values from ``.env.example`` (fallback)

    ``ENV_FILE`` can be used to point to a specific ``.env`` file.
    """

    global _ENV_LOADED, _ENV_PATH

    if _ENV_LOADED and not override:
        return _ENV_PATH

    primary_path = _find_first(".env", base_dir=base_dir, extra=extra)
    example_path = _find_first(".env.example", base_dir=base_dir, extra=extra)

    example_values: Dict[str, str] = {}
    if example_path:
        try:
            example_values = {k: v for k, v in (dotenv_values(example_path) or {}).items() if v is not None}
        except Exception:
            example_values = {}

    primary_values: Dict[str, str] = {}
    if primary_path:
        try:
            primary_values = {k: v for k, v in (dotenv_values(primary_path) or {}).items() if v is not None}
        except Exception:
            primary_values = {}

    merged: Dict[str, str] = {}
    merged.update(example_values)
    merged.update(primary_values)
    merged.update({k: v for k, v in os.environ.items()})

    for key, value in merged.items():
        if override or key not in os.environ:
            os.environ[key] = str(value)

    if primary_path:
        os.environ.setdefault("ENV_PATH_HINT", str(primary_path))

    _ENV_LOADED = True
    _ENV_PATH = primary_path
    return primary_path
