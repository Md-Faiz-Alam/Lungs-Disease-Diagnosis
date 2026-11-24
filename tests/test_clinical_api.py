"""Clinical studies API: lifecycle, DICOM de-identification, Entra auth, isolation, triage."""

import io
import time
import uuid

import jwt
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from PIL import Image

from lungxray.service import auth as auth_module
from lungxray.service.triage import triage
from tests.conftest import fake_xray

TENANT, AUDIENCE = "11111111-2222-3333-4444-555555555555", "api://lungxray-test"


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _dicom(view: str = "PA", photometric: str = "MONOCHROME2") -> bytes:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    arr = (np.asarray(fake_xray(7, 2, size=256), dtype=np.uint16) * 16).astype(np.uint16)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.PatientName, ds.PatientID, ds.PatientBirthDate = "DOE^JANE", "MRN-98765", "19800101"
    ds.InstitutionName, ds.StudyDate = "St Example Hospital", "20240101"
    ds.Modality, ds.ViewPosition, ds.BodyPartExamined = "DX", view, "CHEST"
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = arr.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


@pytest.fixture()
def demo_client(tiny_bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_bundle))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("AUTH_MODE", "demo")
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("RATE_LIMIT_INFERENCE", raising=False)
    from lungxray.api import app

    with TestClient(app, headers={"X-Demo-Session": str(uuid.uuid4())}) as c:
        yield c


def test_demo_visitors_are_isolated(demo_client):
    files = {"file": ("x.png", _png(fake_xray(4, 1, size=200)), "image/png")}
    sid = demo_client.post("/api/studies", files=files).json()["id"]
    other = {"X-Demo-Session": str(uuid.uuid4())}
    assert demo_client.get("/api/studies", headers=other).json() == []
    assert demo_client.get(f"/api/studies/{sid}", headers=other).status_code == 404
    assert demo_client.get("/api/studies", headers={"X-Demo-Session": ""}).status_code == 400
    assert demo_client.get("/api/studies", headers={"X-Demo-Session": "not-a-uuid"}).status_code == 400


def test_inference_is_rate_limited_and_size_capped(tiny_bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_bundle))
    monkeypatch.setenv("AUTH_MODE", "demo")
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("RATE_LIMIT_INFERENCE", "2/60")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    from lungxray.api import app

    files = {"file": ("x.png", _png(fake_xray(4, 1, size=200)), "image/png")}
    with TestClient(app, headers={"X-Demo-Session": str(uuid.uuid4())}) as c:
        assert c.post("/api/studies", files=files).headers["X-Request-ID"]
        assert c.post("/api/studies", files=files).status_code == 201
        limited = c.post("/api/studies", files=files)
        assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0
        assert c.get("/api/studies").status_code == 200  # reads are not limited
        big = {"file": ("big.png", b"0" * (3 * 1024 * 1024), "image/png")}
        assert c.post("/api/studies", files=big).status_code == 413


def test_sliding_window_limiter_frees_slots():
    from lungxray.service.ratelimit import SlidingWindowLimiter

    now = [0.0]
    lim = SlidingWindowLimiter(2, 10, clock=lambda: now[0])
    assert lim.check("a") == 0 and lim.check("a") == 0
    assert lim.check("a") > 0 and lim.check("b") == 0  # other clients unaffected
    now[0] = 10.5
    assert lim.check("a") == 0


def test_study_lifecycle(demo_client):
    c = demo_client
    r = c.post("/api/studies", files={"file": ("cxr.png", _png(fake_xray(1, 1, size=300)), "image/png")})
    assert r.status_code == 201, r.text
    study = r.json()
    sid = study["id"]
    assert study["triage"] in {"urgent", "review", "routine"} and study["review_status"] == "pending"
    assert abs(sum(study["probabilities"].values()) - 1) < 1e-3

    assert [s["id"] for s in c.get("/api/studies").json()] == [sid]
    img = Image.open(io.BytesIO(c.get(f"/api/studies/{sid}/image.png").content))
    heat = Image.open(io.BytesIO(c.get(f"/api/studies/{sid}/gradcam.png").content))
    assert heat.mode == "RGBA" and heat.size == img.size  # overlay is pixel-aligned with the image

    bad = c.patch(f"/api/studies/{sid}/review", json={"status": "disagreed", "label": "CANCER"})
    assert bad.status_code == 422
    ok = c.patch(
        f"/api/studies/{sid}/review", json={"status": "disagreed", "label": "NORMAL", "note": "old scar"}
    )
    assert ok.json()["review_status"] == "disagreed" and ok.json()["review_label"] == "NORMAL"

    pdf = c.get(f"/api/studies/{sid}/report.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")

    actions = [e["action"] for e in c.get("/api/audit").json()]
    assert {"study.create", "study.review.disagreed", "report.download"} <= set(actions)

    assert c.delete(f"/api/studies/{sid}").status_code == 204
    assert c.get(f"/api/studies/{sid}").status_code == 404
    assert c.get("/api/studies/..%2F..%2Fetc").status_code == 404


def test_dicom_is_deidentified(demo_client):
    r = demo_client.post("/api/studies", files={"file": ("MRN-98765_DOE.dcm", _dicom(), "application/dicom")})
    assert r.status_code == 201, r.text
    body = r.text
    for phi in ("DOE", "JANE", "MRN-98765", "19800101", "St Example", "20240101"):
        assert phi not in body, f"PHI leaked: {phi}"
    study = r.json()
    assert study["source"] == "dicom" and study["filename"] == "DICOM study"
    assert study["dicom_metadata"]["Modality"] == "DX" and study["dicom_metadata"]["ViewPosition"] == "PA"


def test_dicom_lateral_view_is_flagged(demo_client):
    study = demo_client.post(
        "/api/studies", files={"file": ("x.dcm", _dicom(view="LL"), "application/dicom")}
    )
    assert any("view position" in w for w in study.json()["warnings"])


def test_rejects_non_xray_inputs(demo_client):
    rng = np.random.default_rng(0)
    colour = Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8), "RGB")
    assert (
        demo_client.post("/api/studies", files={"file": ("p.png", _png(colour), "image/png")}).status_code
        == 422
    )
    assert (
        demo_client.post("/api/studies", files={"file": ("a.pdf", b"%PDF-1", "application/pdf")}).status_code
        == 415
    )
    assert (
        demo_client.post(
            "/api/studies", files={"file": ("x.dcm", b"x" * 300, "application/dicom")}
        ).status_code
        == 422
    )


# --- Entra ID mode ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(key, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": f"https://login.microsoftonline.com/{TENANT}/v2.0",
        "aud": AUDIENCE,
        "iat": now,
        "nbf": now,
        "exp": now + 600,
        "oid": "user-a",
        "name": "Dr A",
        "scp": "Studies.ReadWrite",
    }
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})


@pytest.fixture()
def entra_client(tiny_bundle, tmp_path, monkeypatch, signing_key):
    monkeypatch.setenv("MODEL_PATH", str(tiny_bundle))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("AUTH_MODE", "entra")
    monkeypatch.setenv("ENTRA_TENANT_ID", TENANT)
    monkeypatch.setenv("ENTRA_API_AUDIENCE", AUDIENCE)
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("API_KEY", raising=False)

    class FakeJwks:
        def get_signing_key_from_jwt(self, _token):
            return type("K", (), {"key": signing_key.public_key()})()

    monkeypatch.setattr(auth_module, "_jwks_client", lambda _tenant: FakeJwks())
    from lungxray.api import app

    with TestClient(app) as c:
        yield c


def test_entra_token_validation(entra_client, signing_key):
    c = entra_client
    assert c.get("/api/me").status_code == 401
    ok = c.get("/api/me", headers={"Authorization": f"Bearer {_token(signing_key)}"})
    assert ok.status_code == 200 and ok.json()["name"] == "Dr A" and ok.json()["auth_mode"] == "entra"
    wrong_aud = _token(signing_key, aud="api://someone-else")
    assert c.get("/api/me", headers={"Authorization": f"Bearer {wrong_aud}"}).status_code == 401
    expired = _token(signing_key, exp=int(time.time()) - 60)
    assert c.get("/api/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    no_scope = _token(signing_key, scp="User.Read")
    assert c.get("/api/me", headers={"Authorization": f"Bearer {no_scope}"}).status_code == 403
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert c.get("/api/me", headers={"Authorization": f"Bearer {_token(other_key)}"}).status_code == 401
    # legacy inference endpoint is protected too in entra mode
    f = {"file": ("x.png", _png(fake_xray(5, 2, size=200)), "image/png")}
    assert c.post("/predict", files=f).status_code == 401


def test_users_cannot_see_each_others_studies(entra_client, signing_key):
    a = {"Authorization": f"Bearer {_token(signing_key, oid='user-a')}"}
    b = {"Authorization": f"Bearer {_token(signing_key, oid='user-b', name='Dr B')}"}
    files = {"file": ("x.png", _png(fake_xray(3, 0, size=200)), "image/png")}
    sid = entra_client.post("/api/studies", files=files, headers=a).json()["id"]
    assert entra_client.get(f"/api/studies/{sid}", headers=b).status_code == 404
    assert entra_client.get(f"/api/studies/{sid}/image.png", headers=b).status_code == 404
    assert entra_client.get("/api/studies", headers=b).json() == []
    assert len(entra_client.get("/api/studies", headers=a).json()) == 1


@pytest.mark.parametrize(
    ("pred", "needs_review", "warnings", "expected"),
    [
        ("PNEUMONIA", False, [], "urgent"),
        ("COVID19", False, ["lateral view"], "urgent"),
        ("NORMAL", False, [], "routine"),
        ("NORMAL", False, ["unusual aspect"], "review"),
        ("PNEUMONIA", True, [], "review"),
    ],
)
def test_triage_rules(pred, needs_review, warnings, expected):
    assert triage(pred, needs_review, warnings) == expected
