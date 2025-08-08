FROM python:3.12.11

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./
COPY config.py ./
COPY instructions.json ./

COPY src ./src
COPY data/qdrant/dense ./data/qdrant/dense

CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]