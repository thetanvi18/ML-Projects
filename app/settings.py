from __future__ import annotations

import os
from pydantic import BaseModel


class Settings(BaseModel):
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    app_name: str = "AI Travel Planner"
    trips_dir: str = os.getenv("TRIPS_DIR", "/workspace/data/trips")


settings = Settings()
