FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8058
ENV PYTHONUNBUFFERED=1
# Ensure top-level packages (agent, ingestion, worker, integrations) are always
# importable — the Celery console-script launcher does not keep /app on sys.path
# at task runtime the way `python -m` does.
ENV PYTHONPATH=/app

EXPOSE 8058

CMD ["python", "-m", "uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8058"]
