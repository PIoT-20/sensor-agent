FROM python:3.11-slim

WORKDIR /app

# Install small set of system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements (we add requirements file below)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app files
COPY config.json /app/config.json
COPY src/ /app/src/

EXPOSE 8080

CMD ["python", "src/__main__.py"]
