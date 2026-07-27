# Lightweight Streamlit + Python image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY app/main.py ./main.py

EXPOSE 8501

# Accept env at runtime (GEMINI_API_KEY, TRIPS_DIR)
CMD ["bash", "-lc", "streamlit run main.py --server.port=8501 --server.address=0.0.0.0"]
