# LeafDx API

FastAPI service for server-side inference and Grad-CAM heat maps, for use outside the web app
(scripts, other clients). The web app does not need it: it computes predictions and heat maps
in the browser.

```bash
(cd ../ml && .venv/bin/python -m leafdx.train.export_web api-model)   # writes api/model/
python -m venv .venv && .venv/bin/pip install -e ".[dev]"            # Python 3.10-3.12
.venv/bin/uvicorn leafdx_api.main:app --port 8000                    # docs at /docs
.venv/bin/pytest -q && .venv/bin/ruff check .
```

| Endpoint | |
|---|---|
| `GET /health` | status and whether the model is loaded |
| `GET /classes` | the 38 classes in model output order |
| `POST /predict` | multipart `file`; query `top_k` (1-10), `gradcam` (bool). Returns `top_k`, `verdict` (`confident`, `uncertain`, `not-sure`, `not-a-leaf`) and optionally `gradcam` (PNG data URL) |

Safeguards: JPEG/PNG/WebP/BMP/GIF only, 10 MB limit, decompression-bomb guard, per-client rate
limit (30 `/predict` calls per minute), CORS allow-list. Images are processed in memory and never
stored or logged.

Configuration (environment): `LEAFDX_MODEL`, `LEAFDX_CLASSES`, `LEAFDX_MAX_UPLOAD_MB`,
`LEAFDX_RATE_LIMIT`, `LEAFDX_CORS_ORIGINS` (comma-separated).

Hosting on Hugging Face Spaces: `Dockerfile` and `huggingface/README.md` (steps in
[`../docs/deployment.md`](../docs/deployment.md)).
