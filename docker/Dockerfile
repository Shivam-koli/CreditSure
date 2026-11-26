# One image for every service (api, scheduler, bootstrap) - they differ only by command. The
# frontend (web/) is plain static HTML/CSS/JS with no build step, served directly by the api
# service itself (see the StaticFiles mount at the bottom of src/lia/api.py) - no separate
# frontend container or port needed.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY sql ./sql
COPY web ./web
COPY scripts ./scripts
EXPOSE 8000
CMD ["uvicorn", "lia.api:app", "--host", "0.0.0.0", "--port", "8000"]
