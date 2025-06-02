ARG BASE_IMG=python:3.12-slim
FROM ${BASE_IMG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/servers/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    vim \
    htop \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /servers

RUN pip install --no-cache-dir uv
COPY pyproject.toml requirements.txt ./

RUN uv venv && uv pip install -r requirements.txt

COPY . .

EXPOSE 50001
CMD ["python", "server.py"]