"""Streamlit frontend. Talks to the FastAPI service (API_URL) or, if that is not set,
loads the model in-process so the demo also works as a single Hugging Face Space."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "").rstrip("/")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/production/model.pt")
SAMPLES = Path(__file__).resolve().parent.parent / "samples"

st.set_page_config(page_title="Lung X-ray Classifier", page_icon="🫁", layout="wide")


@st.cache_resource
def local_predictor():
    from lungxray.inference.predictor import Predictor

    return Predictor(MODEL_PATH)


def run_prediction(data: bytes, explain: bool) -> dict:
    if API_URL:
        headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
        r = requests.post(
            f"{API_URL}/predict",
            params={"explain": explain},
            files={"file": ("xray.png", data, "image/png")},
            headers=headers,
            timeout=60,
        )
        if r.status_code != 200:
            raise ValueError(r.json().get("detail", r.text))
        return r.json()
    from lungxray.inference.predictor import Predictor

    return local_predictor().predict(Predictor.open_image(data), explain=explain)


def model_info() -> dict:
    try:
        if API_URL:
            return requests.get(f"{API_URL}/model", timeout=10).json()
        return local_predictor().info
    except Exception:
        return {}


st.title("🫁 Chest X-ray Classifier")
st.caption("COVID-19 · Normal · Pneumonia — calibrated confidence, radiologist referral and Grad-CAM")
st.warning(
    "**Research demo only. Not a medical device.** Predictions must not be used for diagnosis.", icon="⚠️"
)

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload a frontal chest X-ray", type=["jpg", "jpeg", "png"])
    samples = sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sample = st.selectbox("…or pick a sample", ["—"] + samples)
    explain = st.toggle("Show Grad-CAM heatmap", value=True)
    info = model_info()
    if info:
        st.header("Model")
        st.write(f"**{info.get('arch')}** · run `{info.get('run_id')}`")
        m = info.get("metrics", {})
        if m:
            c1, c2 = st.columns(2)
            c1.metric("Balanced acc.", f"{m.get('balanced_accuracy', 0):.1%}")
            c2.metric("Macro AUC", f"{m.get('macro_roc_auc', 0):.3f}")
            st.caption("Measured on the leakage-free test set.")

data = None
if uploaded is not None:
    data = uploaded.getvalue()
elif sample != "—":
    data = (SAMPLES / sample).read_bytes()

if data is None:
    st.info("Upload an X-ray or choose a sample from the sidebar.")
    st.stop()

try:
    with st.spinner("Analysing…"):
        res = run_prediction(data, explain)
except Exception as e:  # invalid image, API down, ...
    st.error(f"Could not analyse this image: {e}")
    st.stop()

left, right = st.columns(2)
left.image(Image.open(io.BytesIO(data)), caption="Input", width="stretch")
if res.get("gradcam_png_base64"):
    right.image(
        base64.b64decode(res["gradcam_png_base64"]),
        caption="Grad-CAM: regions that drove the prediction",
        width="stretch",
    )

label, conf = res["prediction"], res["confidence"]
if res["needs_review"]:
    st.error(f"**{label}** — confidence {conf:.1%}. {res['recommendation']}.", icon="🩺")
else:
    st.success(f"**{label}** — confidence {conf:.1%}. {res['recommendation']}.")
for w in res.get("warnings", []):
    st.warning(w)

st.subheader("Class probabilities (temperature-calibrated)")
st.bar_chart(res["probabilities"], horizontal=True)
st.caption(f"Latency {res['latency_ms']} ms · model {res['model']['arch']}")
