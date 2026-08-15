FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py ingest.py ops_store.py avatar_api.py ./
COPY eval ./eval
COPY miniapp ./miniapp

CMD ["python", "bot.py"]
