# Dockerfile for Khmer OCR Telegram Bot
FROM python:3.11-slim

# Install system dependencies (Tesseract + Khmer language + Poppler)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-khm \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python deps
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app

# Run the bot in polling mode (simple, no HTTPS needed)
CMD ["python", "khmer_bot.py"]
