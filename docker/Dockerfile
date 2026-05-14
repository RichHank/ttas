FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    rustc \
    cargo \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    proj-data \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY . /app
RUN python run_pipeline.py --properties-per-zip-month 4 --max-points 160 --grid-size 9

EXPOSE 8050
CMD ["python", "dashboard/app.py"]
