"""/api routes used by the web app: studies, worklist, images, review, PDF report, audit."""

from __future__ import annotations

import io
import re
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from lungxray.inference.predictor import InvalidImageError, Predictor
from lungxray.service.auth import User, get_current_user
from lungxray.service.dicom_io import DicomError, looks_like_dicom, read_dicom
from lungxray.service.report import build_report
from lungxray.service.settings import Settings, get_settings
from lungxray.service.storage import AuditEvent, Study, StudyRepository, new_study_id, now_iso
from lungxray.service.triage import TRIAGE_RANK, triage

router = APIRouter(prefix="/api", tags=["clinical"])
STUDY_ID = re.compile(r"^ST-\d{8}-[0-9A-F]{8}$")
STORE_MAX_SIDE = 1024
IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}
DICOM_TYPES = {"application/dicom", "application/octet-stream", ""}


def get_repo(request: Request) -> StudyRepository:
    return request.app.state.repo


def get_predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return predictor


def _audit(repo: StudyRepository, user: User, action: str, study_id: str = "", detail: str = "") -> None:
    repo.add_audit(
        AuditEvent(user_id=user.id, user_name=user.name, action=action, study_id=study_id, detail=detail)
    )


def _load_study(repo: StudyRepository, user: User, study_id: str) -> Study:
    if not STUDY_ID.match(study_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Study not found")
    study = repo.get_study(user.id, study_id)
    if study is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Study not found")
    return study


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@router.get("/me")
def me(user: User = Depends(get_current_user), settings: Settings = Depends(get_settings)) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "is_demo": user.is_demo,
        "auth_mode": settings.auth_mode,
    }


@router.get("/model")
def model(predictor: Predictor = Depends(get_predictor), _user: User = Depends(get_current_user)) -> dict:
    return predictor.info


@router.get("/studies")
def list_studies(
    triage_filter: Literal["urgent", "review", "routine"] | None = None,
    review_status: Literal["pending", "agreed", "disagreed"] | None = None,
    user: User = Depends(get_current_user),
    repo: StudyRepository = Depends(get_repo),
) -> list[dict]:
    studies = repo.list_studies(user.id)
    if triage_filter:
        studies = [s for s in studies if s.triage == triage_filter]
    if review_status:
        studies = [s for s in studies if s.review_status == review_status]
    # worklist order: unreviewed first, then by urgency, newest first
    studies.sort(key=lambda s: s.created_at, reverse=True)
    studies.sort(key=lambda s: (s.review_status != "pending", TRIAGE_RANK.get(s.triage, 9)))
    return [s.to_dict() for s in studies]


@router.post("/studies", status_code=status.HTTP_201_CREATED)
def create_study(
    file: UploadFile = File(..., description="Chest radiograph: DICOM (.dcm), PNG or JPEG"),
    user: User = Depends(get_current_user),
    repo: StudyRepository = Depends(get_repo),
    predictor: Predictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
) -> dict:
    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {settings.max_upload_mb} MB"
        )
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    ctype = (file.content_type or "").lower()
    name = (file.filename or "upload").rsplit("/", 1)[-1][:120]

    meta: dict[str, str] = {}
    dicom_warnings: list[str] = []
    try:
        if looks_like_dicom(data) or name.lower().endswith(".dcm"):
            img, meta, dicom_warnings = read_dicom(data)
            source = "dicom"
        elif ctype in IMAGE_TYPES or ctype in DICOM_TYPES:
            img = Predictor.open_image(data)
            source = "image"
        else:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type {ctype}")
        result, heat = predictor.analyze(img)
    except (InvalidImageError, DicomError) as e:
        _audit(repo, user, "study.rejected", detail=str(e)[:200])
        raise HTTPException(422, str(e)) from e

    stored = img.convert("L")
    stored.thumbnail((STORE_MAX_SIDE, STORE_MAX_SIDE))
    warnings = dicom_warnings + result["warnings"]
    study = Study(
        id=new_study_id(),
        owner=user.id,
        created_at=now_iso(),
        filename=name if source == "image" else "DICOM study",  # DICOM filenames often embed patient IDs
        source=source,
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        needs_review=result["needs_review"],
        triage=triage(result["prediction"], result["needs_review"], warnings),
        warnings=warnings,
        dicom_metadata=meta,
        image_width=stored.width,
        image_height=stored.height,
        model_arch=result["model"]["arch"],
        model_run_id=result["model"]["run_id"] or "",
    )
    repo.put_blob(repo.blob_key(user.id, study.id, "image"), _png(stored), "image/png")
    repo.put_blob(
        repo.blob_key(user.id, study.id, "gradcam"),
        _png(predictor.heatmap_layer(heat, stored.size)),
        "image/png",
    )
    repo.save_study(study)
    _audit(repo, user, "study.create", study.id, f"{study.prediction} {study.confidence:.3f} {study.triage}")
    return study.to_dict()


@router.get("/studies/{study_id}")
def get_study(
    study_id: str, user: User = Depends(get_current_user), repo: StudyRepository = Depends(get_repo)
) -> dict:
    study = _load_study(repo, user, study_id)
    _audit(repo, user, "study.view", study.id)
    return study.to_dict()


@router.get("/studies/{study_id}/{kind}.png")
def get_study_image(
    study_id: str,
    kind: Literal["image", "gradcam"],
    user: User = Depends(get_current_user),
    repo: StudyRepository = Depends(get_repo),
) -> Response:
    study = _load_study(repo, user, study_id)
    data = repo.get_blob(repo.blob_key(user.id, study.id, kind))
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")
    return Response(data, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


class ReviewIn(BaseModel):
    status: Literal["agreed", "disagreed"]
    label: str = ""
    note: str = Field("", max_length=1000)


@router.patch("/studies/{study_id}/review")
def review_study(
    study_id: str,
    body: ReviewIn,
    user: User = Depends(get_current_user),
    repo: StudyRepository = Depends(get_repo),
    predictor: Predictor = Depends(get_predictor),
) -> dict:
    study = _load_study(repo, user, study_id)
    if body.status == "disagreed" and body.label not in predictor.classes:
        raise HTTPException(422, f"label must be one of {predictor.classes} when disagreeing")
    study.review_status = body.status
    study.review_label = study.prediction if body.status == "agreed" else body.label
    study.review_note = body.note.strip()
    study.reviewed_at = now_iso()
    study.reviewer = user.name
    repo.save_study(study)
    _audit(repo, user, f"study.review.{body.status}", study.id, study.review_label)
    return study.to_dict()


@router.get("/studies/{study_id}/report.pdf")
def report(
    study_id: str,
    user: User = Depends(get_current_user),
    repo: StudyRepository = Depends(get_repo),
    predictor: Predictor = Depends(get_predictor),
) -> Response:
    study = _load_study(repo, user, study_id)
    pdf = build_report(
        study,
        repo.get_blob(repo.blob_key(user.id, study.id, "image")),
        _flatten_overlay(repo, user, study),
        generated_for=user.name,
        generated_at=now_iso(),
        model_metrics=predictor.info.get("metrics", {}),
    )
    _audit(repo, user, "report.download", study.id)
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{study.id}.pdf"'},
    )


def _flatten_overlay(repo: StudyRepository, user: User, study: Study) -> bytes | None:
    base = repo.get_blob(repo.blob_key(user.id, study.id, "image"))
    heat = repo.get_blob(repo.blob_key(user.id, study.id, "gradcam"))
    if not base or not heat:
        return None
    img = Image.open(io.BytesIO(base)).convert("RGBA")
    layer = Image.open(io.BytesIO(heat)).convert("RGBA").resize(img.size)
    a = layer.getchannel("A").point(lambda v: int(v * 0.55))
    layer.putalpha(a)
    return _png(Image.alpha_composite(img, layer).convert("RGB"))


@router.delete("/studies/{study_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_study(
    study_id: str, user: User = Depends(get_current_user), repo: StudyRepository = Depends(get_repo)
) -> Response:
    study = _load_study(repo, user, study_id)
    for kind in ("image", "gradcam"):
        repo.delete_blob(repo.blob_key(user.id, study.id, kind))
    repo.delete_study(user.id, study.id)
    _audit(repo, user, "study.delete", study.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/audit")
def audit_log(
    user: User = Depends(get_current_user), repo: StudyRepository = Depends(get_repo)
) -> list[dict]:
    return [e.__dict__ for e in repo.list_audit(user.id)]
