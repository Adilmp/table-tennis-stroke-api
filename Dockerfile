# Pinned to Debian 13 (trixie): its package names differ from older images
# (libgl1-mesa-glx was dropped; libglib2.0-0 became libglib2.0-0t64).
FROM python:3.10-slim-trixie

WORKDIR /app

# OpenCV's runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
