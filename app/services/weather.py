from __future__ import annotations

import httpx
from pydantic import BaseModel

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class GeoResult(BaseModel):
    name: str
    country: str | None = None
    latitude: float
    longitude: float


async def geocode_city(city: str) -> GeoResult | None:
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(GEOCODE_URL, params=params)
        r.raise_for_status()
        data = r.json()
        if not data.get("results"):
            return None
        item = data["results"][0]
        return GeoResult(
            name=item.get("name"),
            country=item.get("country"),
            latitude=item.get("latitude"),
            longitude=item.get("longitude"),
        )


class DailyWeather(BaseModel):
    date: str
    temp_max_c: float
    temp_min_c: float
    precipitation_mm: float | None = None


async def get_daily_weather(latitude: float, longitude: float, days: int = 7) -> list[DailyWeather]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
        ],
        "timezone": "auto",
        "forecast_days": min(max(days, 1), 16),
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(FORECAST_URL, params=params)
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])

        items: list[DailyWeather] = []
        for i, date in enumerate(dates):
            items.append(
                DailyWeather(
                    date=date,
                    temp_max_c=float(tmax[i]) if i < len(tmax) else 0.0,
                    temp_min_c=float(tmin[i]) if i < len(tmin) else 0.0,
                    precipitation_mm=float(precip[i]) if i < len(precip) else None,
                )
            )
        return items
