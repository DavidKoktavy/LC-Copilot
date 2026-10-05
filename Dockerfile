FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    LC_DOCS_DIR=/app/data/docs LC_SEED_SQL=/app/data/seed.sql \
    LC_DB_PATH=/app/runs/portfolio.db LC_TRACE_DIR=/app/runs
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data
RUN useradd --create-home --uid 10001 app && mkdir -p /app/runs && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health'); sys.exit(0)"
CMD ["uvicorn", "lc_copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
