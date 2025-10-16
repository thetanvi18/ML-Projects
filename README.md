# AI Travel Planner (Streamlit + Gemini)

A modern 2025-ready AI travel planner built with Streamlit, Google Gemini, Open-Meteo weather, and exchangerate.host for currency. Designed to be a showcase project for your resume: clean architecture, async APIs, structured AI output, Docker, and optional AWS deployment.

## Features
- AI-generated itineraries using Gemini (JSON-first with heuristic fallback)
- Live geocoding and weather (Open-Meteo)
- Currency conversion with exchangerate.host
- Streamlit UI with side-by-side plan + weather
- Save trips to JSON
- Dockerfile for container deployment

## Setup
1. Python 3.11+ recommended
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables (copy `.env.example`):
   - `GEMINI_API_KEY`: Your Google Generative AI API key
   - `TRIPS_DIR` (optional): Where saved trips JSON will go

## Run locally
```bash
streamlit run app/main.py
```

## Docker
```bash
docker build -t ai-travel-planner .
# Replace with your key
docker run -p 8501:8501 -e GEMINI_API_KEY=YOUR_KEY ai-travel-planner
```

Then open `http://localhost:8501`.

## AWS Deployment Options
- **Streamlit Community Cloud** for quick demo
- **AWS App Runner**: Deploy container directly
- **AWS Elastic Beanstalk**: Simple web app hosting
- **AWS EC2**: Self-managed VM running Docker

### App Runner (recommended)
1. Push image to ECR
2. Create App Runner service from ECR image
3. Set env vars (GEMINI_API_KEY, TRIPS_DIR=/app/data/trips)

## Folder Structure
```
app/
  main.py
  settings.py
  services/
    gemini.py
    weather.py
    currency.py
  utils/
    storage.py
```

## Notes
- Keep costs low: uses free weather/currency APIs
- Gemini model can be upgraded (e.g., `gemini-1.5-pro`) later
- Enhance with offline caching, map widgets, PDF export
