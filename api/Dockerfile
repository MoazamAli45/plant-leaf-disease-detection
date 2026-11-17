# Build recipe used by Hugging Face Spaces to run this API (see docs/deployment.md).
# You do not need Docker installed: Hugging Face builds it on its own servers.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    LEAFDX_MODEL=/app/model/inference.keras \
    LEAFDX_CLASSES=/app/model/classes.json

# Spaces run the app as user id 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install .
COPY model ./model

USER user
EXPOSE 7860
CMD ["uvicorn", "leafdx_api.main:app", "--host", "0.0.0.0", "--port", "7860"]
