"""Centralized dotenv loader to keep CLI, UI and subprocesses in sync."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

_ENV_LOADED = False
_ENV_PATH: Optional[Path] = None


_DEF_NAMES = (".env",)


def _iter_candidates(base_dir: Optional[Path] = None, extra: Iterable[Path] = ()):
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
        for name in _DEF_NAMES:
            candidate = (base / name).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def find_dotenv_path(base_dir: Optional[Path] = None, extra: Iterable[Path] = ()) -> Optional[Path]:
    env_path = os.getenv("ENV_FILE")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_file():
            return candidate
    for candidate in _iter_candidates(base_dir, extra):
        if candidate.is_file():
            return candidate
    return None


def load_env(base_dir: Optional[Path] = None, override: bool = False, extra: Iterable[Path] = ()) -> Optional[Path]:
    global _ENV_LOADED, _ENV_PATH
    if _ENV_LOADED and not override:
        return _ENV_PATH

    path = find_dotenv_path(base_dir=base_dir, extra=extra)
    if path:
        load_dotenv(path, override=override)
        os.environ.setdefault("ENV_PATH_HINT", str(path))
    else:
        # fallback to default lookup so upstream behavior remains
        load_dotenv(override=override)
    _ENV_LOADED = True
    _ENV_PATH = path
    return path
