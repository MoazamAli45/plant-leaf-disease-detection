"""API tests. Most use a tiny stand-in model so they run anywhere; the last group runs the real
exported MobileNetV1 when api/model/inference.keras exists."""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

keras = pytest.importorskip("keras")

from leafdx_api.main import create_app  # noqa: E402
from leafdx_api.ratelimit import RateLimiter  # noqa: E402
from leafdx_api.settings import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ROOT / "web" / "src" / "data" / "classes.json"
REAL_MODEL = ROOT / "api" / "model" / "inference.keras"
FIXTURES = ROOT / "web" / "e2e" / "fixtures"


def jpeg(color=(40, 140, 50), size=(320, 240)) -> bytes:
    rng = np.random.default_rng(0)
    arr = np.clip(np.array(color) + rng.normal(0, 12, (size[1], size[0], 3)), 0, 255).astype(
        np.uint8
    )
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    keras.utils.set_random_seed(0)
    inp = keras.Input((224, 224, 3))
    x = keras.layers.Rescaling(1 / 255)(inp)
    x = keras.layers.Conv2D(4, 3, activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(38, activation="softmax")(x)
    path = tmp_path_factory.mktemp("m") / "tiny.keras"
    keras.Model(inp, out).save(path)
    return path


def client_for(model_path, **overrides) -> TestClient:
    s = Settings(model_path=model_path, classes_path=CLASSES, **overrides)
    return TestClient(create_app(s))


@pytest.fixture
def client(tiny_model):
    with client_for(tiny_model, rate_limit_per_minute=1000) as c:
        yield c


def test_health_and_classes(client):
    assert client.get("/health").json()["model_loaded"] is True
    classes = client.get("/classes").json()
    assert len(classes) == 38 and classes[0]["class"] == "Apple___Apple_scab"


def test_predict_returns_sorted_top_k(client):
    r = client.post("/predict?top_k=5", files={"file": ("leaf.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    probs = [t["probability"] for t in body["top_k"]]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True)
    assert body["verdict"] in {"confident", "uncertain", "not-sure"}
    assert "gradcam" not in body


def test_gradcam_overlay_is_a_png(client):
    r = client.post("/predict?gradcam=true", files={"file": ("leaf.jpg", jpeg(), "image/jpeg")})
    data_url = r.json()["gradcam"]
    assert data_url.startswith("data:image/png;base64,")
    img = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
    assert img.size == (224, 224)


def test_grey_photo_is_not_a_leaf(client):
    r = client.post("/predict", files={"file": ("wall.jpg", jpeg((128, 128, 130)), "image/jpeg")})
    assert r.json()["verdict"] == "not-a-leaf"


@pytest.mark.parametrize(
    ("payload", "ctype", "status"),
    [
        (b"%PDF-1.4 not an image", "application/pdf", 415),
        (b"not really a jpeg", "image/jpeg", 400),
    ],
)
def test_rejects_bad_uploads(client, payload, ctype, status):
    r = client.post("/predict", files={"file": ("x", payload, ctype)})
    assert r.status_code == status
    assert "detail" in r.json()


def test_rejects_large_uploads(tiny_model):
    with client_for(tiny_model, max_upload_bytes=1000) as c:
        r = c.post("/predict", files={"file": ("big.jpg", jpeg(size=(800, 600)), "image/jpeg")})
    assert r.status_code == 413


def test_rate_limit(tiny_model):
    with client_for(tiny_model, rate_limit_per_minute=2) as c:
        codes = [
            c.post("/predict", files={"file": ("l.jpg", jpeg(), "image/jpeg")}).status_code
            for _ in range(3)
        ]
        assert c.get("/health").status_code == 200  # only /predict is limited
    assert codes == [200, 200, 429]


def test_rate_limiter_window():
    rl = RateLimiter(per_minute=1, window_s=10)
    assert rl.allow("a", now=0)[0]
    assert not rl.allow("a", now=5)[0]
    assert rl.allow("b", now=5)[0]
    assert rl.allow("a", now=10.1)[0]


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="export the model first (see api/README.md)")
@pytest.mark.parametrize(
    "name", ["tomato-early-blight.jpg", "apple-apple-scab.jpg", "grape-healthy.jpg"]
)
def test_real_model_on_fixtures(name):
    expected = json.loads((FIXTURES / "fixtures.json").read_text())[name]["class"]
    with client_for(REAL_MODEL) as c:
        r = c.post(
            "/predict?gradcam=true",
            files={"file": (name, (FIXTURES / name).read_bytes(), "image/jpeg")},
        )
    body = r.json()
    assert body["top_k"][0]["class"] == expected
    assert body["verdict"] == "confident"
