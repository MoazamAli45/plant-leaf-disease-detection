"""LeafDx API.

    uvicorn leafdx_api.main:app --port 8000

Endpoints
- ``GET /health``   liveness and model status
- ``GET /classes``  the 38 classes in model output order
- ``POST /predict`` multipart ``file``; returns top-k classes, a verdict, and optionally a Grad-CAM
  overlay (``?gradcam=true``) as a PNG data URL

Uploaded images are processed in memory and never stored or logged.
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from leafdx_api import __version__
from leafdx_api.imaging import BadImageError, decode, overlay, plant_pixel_share, to_input
from leafdx_api.model import Classifier
from leafdx_api.ratelimit import RateLimiter
from leafdx_api.settings import Settings

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # noqa: ANN202
        state["classifier"] = Classifier.load(settings.model_path, settings.classes_path)
        yield
        state.clear()

    app = FastAPI(
        title="LeafDx API",
        version=__version__,
        description="Plant leaf disease classification and Grad-CAM. Research use only; "
        "not agronomic advice.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):  # noqa: ANN001, ANN202
        if request.url.path == "/predict":
            fwd = request.headers.get("x-forwarded-for")
            client = (
                fwd.split(",")[0].strip()
                if fwd
                else (request.client.host if request.client else "?")
            )
            ok, retry = limiter.allow(client)
            if not ok:
                return JSONResponse(
                    {"detail": "Too many requests. Please wait a moment and try again."},
                    status_code=429,
                    headers={"Retry-After": str(int(retry) + 1)},
                )
        return await call_next(request)

    def classifier() -> Classifier:
        c = state.get("classifier")
        if c is None:
            raise HTTPException(503, "model not loaded")
        return c

    @app.get("/health")
    def health() -> dict[str, Any]:
        c = state.get("classifier")
        return {
            "status": "ok" if c else "loading",
            "model_loaded": c is not None,
            "version": __version__,
        }

    @app.get("/classes")
    def classes() -> list[dict]:
        return classifier().classes

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(..., description="A photo of one leaf (JPEG, PNG or WebP)"),
        top_k: int = Query(3, ge=1, le=10),
        gradcam: bool = Query(False, description="Also return a Grad-CAM overlay"),
    ) -> dict[str, Any]:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, "Please upload a JPEG, PNG or WebP image.")
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                413, f"Images must be at most {settings.max_upload_bytes // 2**20} MB."
            )
        try:
            img = decode(data, settings.max_pixels)
        except BadImageError as e:
            raise HTTPException(400, f"Could not read the image: {e}.") from e

        c = classifier()
        x = to_input(img)
        probs = c.predict(x)
        order = probs.argsort()[::-1][:top_k]
        share = plant_pixel_share(img)
        top = float(probs[order[0]])
        if share < settings.min_plant_pixels:
            verdict = "not-a-leaf"
        elif top >= settings.confident:
            verdict = "confident"
        elif top >= settings.not_sure:
            verdict = "uncertain"
        else:
            verdict = "not-sure"
        result: dict[str, Any] = {
            "top_k": [
                {**c.classes[int(i)], "probability": round(float(probs[i]), 6)} for i in order
            ],
            "verdict": verdict,
            "plant_pixel_share": round(share, 4),
            "disclaimer": "Research use only; not agronomic advice.",
        }
        if gradcam:
            cam = c.gradcam(x, int(order[0]))
            buf = io.BytesIO()
            overlay(img, cam).save(buf, format="PNG")
            result["gradcam"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return result

    return app


app = create_app()
