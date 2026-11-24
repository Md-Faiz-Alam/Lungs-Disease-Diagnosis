"""DICOM ingestion with de-identification.

Only pixel data and a small allow-list of *technical* attributes are kept. Patient name, IDs,
birth date, dates/times, institution, physicians and every other tag are dropped before
anything is stored or logged. The original DICOM file is never persisted.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

SAFE_TAGS = (
    "Modality",
    "ViewPosition",
    "BodyPartExamined",
    "PhotometricInterpretation",
    "Rows",
    "Columns",
    "BitsStored",
    "Manufacturer",
)
CHEST_VIEWS = {"PA", "AP", ""}
XRAY_MODALITIES = {"CR", "DX", ""}


class DicomError(ValueError):
    pass


def looks_like_dicom(data: bytes) -> bool:
    return len(data) > 132 and data[128:132] == b"DICM"


def read_dicom(data: bytes) -> tuple[Image.Image, dict, list[str]]:
    """Return (8-bit grayscale image, de-identified metadata, warnings)."""
    import pydicom
    from pydicom.pixels import apply_modality_lut, apply_voi_lut

    try:
        ds = pydicom.dcmread(io.BytesIO(data), force=True)
        pixels = ds.pixel_array
    except Exception as e:  # corrupt file, unsupported transfer syntax, no pixel data
        raise DicomError(f"Could not read DICOM pixel data: {e}") from e

    if pixels.ndim == 3 and pixels.shape[-1] not in (3, 4):  # multi-frame: take the first frame
        pixels = pixels[0]
    arr = apply_modality_lut(pixels, ds)
    arr = apply_voi_lut(arr, ds, index=0) if "WindowCenter" in ds or "VOILUTSequence" in ds else arr
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = 1.0 - arr  # MONOCHROME1 stores bright as low values
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")

    meta = {tag: str(ds.get(tag, "")) for tag in SAFE_TAGS if tag in ds}
    warnings = []
    view = str(ds.get("ViewPosition", "")).upper()
    if view not in CHEST_VIEWS:
        warnings.append(f"DICOM view position is '{view}'; the model expects frontal (PA/AP) views")
    modality = str(ds.get("Modality", "")).upper()
    if modality not in XRAY_MODALITIES:
        warnings.append(f"DICOM modality is '{modality}'; the model expects plain radiographs (CR/DX)")
    body = str(ds.get("BodyPartExamined", "")).upper()
    if body and "CHEST" not in body and "THORAX" not in body and "LUNG" not in body:
        warnings.append(f"DICOM body part is '{body}'; the model expects chest radiographs")
    return img, meta, warnings
