FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

# Combine apt commands and clean up
RUN apt update && apt install -y --no-install-recommends awscli \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]

