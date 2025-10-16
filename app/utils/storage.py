from __future__ import annotations

import os
import uuid
from pathlib import Path

from app.settings import settings


def ensure_trips_dir() -> Path:
    path = Path(settings.trips_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_trip_json(json_text: str) -> Path:
    path = ensure_trips_dir()
    file = path / f"trip_{uuid.uuid4().hex}.json"
    file.write_text(json_text, encoding="utf-8")
    return file
