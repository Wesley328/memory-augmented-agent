FROM python:3.13.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY agent ./agent
COPY api ./api
COPY core ./core
COPY memory ./memory
COPY observability ./observability
COPY service ./service
COPY tools ./tools
COPY main.py ./main.py

RUN mkdir -p /app/data && uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
