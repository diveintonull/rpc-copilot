FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0 \
    PATH="/app/.venv/bin:$PATH" \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    MODELSCOPE_CACHE=/home/app/.cache/modelscope

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --only-group container --no-install-project

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /home/app/.cache /app/data/visual \
    && chown -R app:app /home/app /app/data/visual

COPY --chown=app:app . .

USER app

EXPOSE 8000

CMD ["python", "-m", "api.serve"]
