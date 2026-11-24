# ─────────────────────────────────────────────────────────────────────────────
# Targets
#   runtime (default)  slim web image with the trained artifacts baked in
#   train              full pipeline image (prepare / train / evaluate), data mounted at runtime
# ─────────────────────────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1
WORKDIR /app

# ── dependencies + package, without training/dev extras ─────────────────────
FROM base AS runtime-deps
# dependency layer first: only rebuilt when pyproject.toml / uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-default-groups
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-default-groups --no-editable

# ── offline pipeline image ──────────────────────────────────────────────────
FROM base AS train
# implicit's compiled ALS kernels link against the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# dependency layer first: only rebuilt when pyproject.toml / uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-default-groups --group train
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-default-groups --group train --no-editable
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["recommender"]
CMD ["all"]

# ── web runtime ─────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=runtime-deps /app/.venv /app/.venv
COPY --chown=app:app artifacts ./artifacts
COPY --chown=app:app reports ./reports
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    WEB_CONCURRENCY=2
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz',timeout=4)"
# --preload loads the model once and shares it between workers (copy-on-write)
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY} --threads 4 --preload --access-logfile - 'recommender.web:create_app()'"]
