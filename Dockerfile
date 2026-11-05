# Serving image. It installs only the base dependencies, so it contains no PyTorch: the model runs
# as an ONNX graph. The model bundle is mounted at run time, so the image is independent of any model.
FROM python:3.12-slim AS builder

ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
RUN pip install --no-cache-dir uv==0.9.5
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim

# libgomp1 is the OpenMP runtime that FAISS needs on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 patchsentry

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PATCHSENTRY_BUNDLE_DIR=/models/bundle

USER patchsentry
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["python", "-m", "patchsentry.serving", "--host", "0.0.0.0", "--port", "8000"]
