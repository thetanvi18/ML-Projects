from __future__ import annotations

import json
import asyncio
from typing import List

import streamlit as st

from app.settings import settings
from app.services.weather import geocode_city, get_daily_weather
from app.services.currency import get_rates, convert
from app.services.gemini import generate_itinerary, Itinerary
from app.utils.storage import save_trip_json

st.set_page_config(page_title=settings.app_name, page_icon="🧭", layout="wide")

st.title("🧭 AI Travel Planner (2025)")

with st.sidebar:
    st.header("Trip Inputs")
    destination = st.text_input("Destination city or region", placeholder="e.g., Tokyo")
    num_days = st.number_input("Days", min_value=1, max_value=21, value=5)
    budget = st.selectbox("Budget", ["Backpacker", "Moderate", "Luxury"])
    interests: List[str] = st.multiselect(
        "Interests",
        [
            "Food & Markets",
            "Museums & Culture",
            "Nature & Hikes",
            "Nightlife",
            "Architecture",
            "Local Neighborhoods",
        ],
        default=["Food & Markets", "Local Neighborhoods"],
    )
    user_currency = st.text_input("Your currency code", value="USD")
    generate = st.button("Generate Itinerary")

col1, col2 = st.columns([2, 1])

if generate and destination:
    with st.spinner("Planning your trip with AI..."):
        # Run services in parallel
        async def run_parallel():
            geo = await geocode_city(destination)
            rates = await get_rates(base="USD")
            return geo, rates

        geo, rates = asyncio.run(run_parallel())
        itinerary: Itinerary | None = None
        error = None
        try:
            itinerary = generate_itinerary(destination, int(num_days), budget, interests)
        except Exception as e:
            error = str(e)

    if error:
        st.error(f"Failed to generate itinerary: {error}")

    if itinerary:
        with col1:
            st.subheader("Daily Plan")
            total_usd = 0.0
            for day in itinerary.days:
                with st.expander(f"Day {day.day}: {day.city}"):
                    for item in day.plan:
                        st.markdown(f"- {item}")
                    if day.est_cost_usd is not None:
                        total_usd += day.est_cost_usd
                        if user_currency and user_currency.upper() != "USD":
                            try:
                                converted = convert(day.est_cost_usd, "USD", user_currency.upper(), rates)
                                st.caption(f"~ {converted:,.2f} {user_currency.upper()}")
                            except Exception:
                                pass
            if itinerary.estimated_total_usd:
                total_usd = itinerary.estimated_total_usd
            st.info(f"Estimated total (USD): {total_usd:,.2f}")

        with col2:
            st.subheader("Weather Snapshot")
            if 'geo' in locals() and geo:
                try:
                    days_weather = asyncio.run(get_daily_weather(geo.latitude, geo.longitude, int(num_days)))
                    for w in days_weather[: int(num_days)]:
                        st.write(f"{w.date}: {w.temp_min_c:.1f}°C – {w.temp_max_c:.1f}°C, "+
                                 (f"{w.precipitation_mm:.0f} mm rain" if w.precipitation_mm else "low rain"))
                except Exception as e:
                    st.caption(f"Weather unavailable: {e}")
            else:
                st.caption("Enter a valid destination to view weather.")

        st.divider()
        # Save raw JSON
        as_json = itinerary.model_dump_json(indent=2)
        st.code(as_json, language="json")
        if st.button("Save trip JSON"):
            path = save_trip_json(as_json)
            st.success(f"Saved to {path}")
else:
    st.write("Use the sidebar to set your destination, days, budget, and interests, then click Generate.")
