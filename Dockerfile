FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only `app/` is needed: topics and feeds moved into Postgres in phase 5, so the
# `config/` directory (and its `COPY`) is gone.
COPY app ./app

# Long-running polling worker (AGENTS.md section 8 — simple while-loop scheduler)
CMD ["python", "-m", "app.main"]
