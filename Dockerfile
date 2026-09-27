FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config

# Long-running polling worker (AGENTS.md section 8 — simple while-loop scheduler)
CMD ["python", "-m", "app.main"]
