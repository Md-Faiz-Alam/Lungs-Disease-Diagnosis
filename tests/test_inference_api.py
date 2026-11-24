import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lungxray.inference.predictor import InvalidImageError, Predictor
from tests.conftest import fake_xray


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predictor_outputs_calibrated_probabilities(tiny_bundle, tmp_path):
    log = tmp_path / "pred.jsonl"
    p = Predictor(tiny_bundle, prediction_log=log)
    res = p.predict(fake_xray(3, 1), explain=True)
    assert set(res["probabilities"]) == {"COVID19", "NORMAL", "PNEUMONIA"}
    assert abs(sum(res["probabilities"].values()) - 1) < 1e-3
    assert res["gradcam_png_base64"]
    assert json.loads(log.read_text().splitlines()[0])["prediction"] == res["prediction"]


def test_predictor_rejects_colour_photo_and_tiny_image(tiny_bundle):
    p = Predictor(tiny_bundle)
    rng = np.random.default_rng(0)
    colour = Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8), "RGB")
    with pytest.raises(InvalidImageError, match="colour"):
        p.predict(colour)
    with pytest.raises(InvalidImageError, match="too small"):
        p.predict(fake_xray(1, 0, size=64))


@pytest.fixture()
def client(tiny_bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_bundle))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.delenv("API_KEY", raising=False)
    from lungxray.api import app

    with TestClient(app) as c:
        yield c


def test_api_health_and_predict(client):
    assert client.get("/health").json()["model_loaded"] is True
    r = client.post("/predict?explain=true", files={"file": ("x.png", _png(fake_xray(5, 2)), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in body["probabilities"] and body["gradcam_png_base64"]
    assert "lungxray_requests_total 1" in client.get("/metrics").text


def test_api_rejects_bad_inputs(client):
    assert client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/predict", files={"file": ("a.png", b"not an image", "image/png")}).status_code == 422


def test_api_batch(client):
    files = [("files", (f"{i}.png", _png(fake_xray(i, i % 3)), "image/png")) for i in range(3)]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 200 and len(r.json()) == 3


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    f = {"file": ("x.png", _png(fake_xray(5, 2)), "image/png")}
    assert client.post("/predict", files=f).status_code == 401
    assert client.post("/predict", files=f, headers={"X-API-Key": "secret"}).status_code == 200
