# LeafDx API (FastAPI + TensorFlow CPU). Also used as-is by Hugging Face Spaces (Docker SDK).
#   (cd ../ml && .venv/bin/python -m leafdx.train.export_web api-model)   # writes api/model/
#   docker build --platform linux/amd64 -t leafdx-api .
#   docker run --rm -p 7860:7860 -e LEAFDX_CORS_ORIGINS=http://localhost:3000 leafdx-api
# Docs at http://localhost:7860/docs

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860 \
    LEAFDX_MODEL=/app/model/inference.keras \
    LEAFDX_CLASSES=/app/model/classes.json

# Hugging Face Spaces runs containers as uid 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY model ./model

USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "exec uvicorn leafdx_api.main:app --host 0.0.0.0 --port ${PORT}"]
