from __future__ import annotations

import json
import os
from typing import Any

import google.generativeai as genai
from pydantic import BaseModel, Field

from app.settings import settings


ITINERARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer"},
                    "city": {"type": "string"},
                    "plan": {"type": "array", "items": {"type": "string"}},
                    "est_cost_usd": {"type": "number"},
                },
                "required": ["day", "city", "plan"],
            },
        },
        "tips": {"type": "array", "items": {"type": "string"}},
        "estimated_total_usd": {"type": "number"},
    },
    "required": ["days"],
}


class DayPlan(BaseModel):
    day: int
    city: str
    plan: list[str]
    est_cost_usd: float | None = None


class Itinerary(BaseModel):
    summary: str | None = None
    days: list[DayPlan] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    estimated_total_usd: float | None = None


def _ensure_client() -> genai.GenerativeModel:
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def build_prompt(destination: str, days: int, budget: str, interests: list[str]) -> str:
    interests_str = ", ".join(interests) if interests else "general sightseeing"
    return (
        "You are an expert 2025 travel planner. Create a practical, walkable itinerary with real "
        " neighborhoods and attractions. Optimize for time and local transit. Include budget-aware tips.\n\n"
        f"Destination: {destination}\n"
        f"Days: {days}\n"
        f"Budget: {budget}\n"
        f"Interests: {interests_str}\n\n"
        "Return concise plans for each day, with 4-6 bullet points, short sentences."
    )


def generate_itinerary(destination: str, days: int, budget: str, interests: list[str]) -> Itinerary:
    model = _ensure_client()
    prompt = build_prompt(destination, days, budget, interests)

    # Try JSON structured output first
    try:
        resp = model.generate_content([
            prompt,
            {"mime_type": "application/json", "schema": ITINERARY_SCHEMA},
        ])
        text = resp.text
        data = json.loads(text)
        return Itinerary(**data)
    except Exception:
        pass

    # Fallback: parse bullet text heuristically
    resp = model.generate_content(prompt)
    text = resp.text or ""
    days_list: list[DayPlan] = []
    current_day = 1
    current_plan: list[str] = []
    for line in text.splitlines():
        stripped = line.strip("- *")
        if stripped.lower().startswith("day "):
            if current_plan:
                days_list.append(DayPlan(day=current_day, city=destination, plan=current_plan))
                current_plan = []
                current_day += 1
        elif stripped:
            current_plan.append(stripped)
    if current_plan:
        days_list.append(DayPlan(day=current_day, city=destination, plan=current_plan))

    return Itinerary(summary=f"Itinerary for {destination}", days=days_list)
