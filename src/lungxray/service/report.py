"""Structured PDF report for a study (ReportLab)."""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from lungxray.service.storage import Study

TRIAGE_COLORS = {"urgent": "#b42318", "review": "#b54708", "routine": "#067647"}
LABELS = {"COVID19": "COVID-19 pattern", "NORMAL": "No acute finding", "PNEUMONIA": "Pneumonia pattern"}


def _img(data: bytes | None, width_mm: float) -> RLImage | Paragraph:
    if not data:
        return Paragraph("(image unavailable)", getSampleStyleSheet()["Normal"])
    from PIL import Image as PILImage

    jpeg = io.BytesIO()  # JPEG keeps the report small enough to e-mail (PNG made it ~1.6 MB)
    PILImage.open(io.BytesIO(data)).convert("RGB").save(jpeg, format="JPEG", quality=85)
    jpeg.seek(0)
    img = RLImage(jpeg)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_mm * mm, width_mm * mm * ratio
    return img


def build_report(
    study: Study,
    image_png: bytes | None,
    gradcam_png: bytes | None,
    generated_for: str,
    generated_at: str,
    model_metrics: dict,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"AI chest X-ray report {study.id}",
        author="LungXray",
    )
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4)
    small = ParagraphStyle(
        "small", parent=ss["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#475467")
    )
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.5, leading=13)

    grid = TableStyle(
        [
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d5dd")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f4f7")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]
    )
    story = [
        Paragraph("AI-Assisted Chest X-ray Analysis", h1),
        Paragraph(
            "Clinical decision support output - <b>research use only, not a medical device</b>. "
            "Findings must be confirmed by a qualified radiologist.",
            small,
        ),
        Spacer(1, 6),
    ]
    tcol = TRIAGE_COLORS.get(study.triage, "#475467")
    summary = [
        [
            "Study ID",
            study.id,
            "Triage",
            Paragraph(f'<font color="{tcol}"><b>{study.triage.upper()}</b></font>', body),
        ],
        ["Uploaded", study.created_at.replace("T", " "), "Source", study.source.upper()],
        [
            "AI finding",
            Paragraph(f"<b>{LABELS.get(study.prediction, study.prediction)}</b>", body),
            "Calibrated confidence",
            f"{study.confidence:.1%}",
        ],
        [
            "Model",
            f"{study.model_arch} ({study.model_run_id})",
            "Needs review",
            "Yes" if study.needs_review else "No",
        ],
    ]
    t = Table(summary, colWidths=[28 * mm, 62 * mm, 36 * mm, 52 * mm])
    t.setStyle(grid)
    t.setStyle(TableStyle([("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f2f4f7"))]))
    story += [t, Paragraph("Class probabilities (temperature-calibrated)", h2)]

    probs = [["Class", "Probability"]] + [
        [LABELS.get(k, k), f"{v:.1%}"] for k, v in sorted(study.probabilities.items(), key=lambda kv: -kv[1])
    ]
    pt = Table(probs, colWidths=[70 * mm, 30 * mm])
    pt.setStyle(grid)
    pt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f4f7")),
                ("BACKGROUND", (0, 1), (0, -1), colors.white),
            ]
        )
    )
    story.append(pt)

    story += [
        Paragraph("Images", h2),
        Table(
            [
                [_img(image_png, 84), _img(gradcam_png, 84)],
                [
                    Paragraph("Input radiograph", small),
                    Paragraph("Grad-CAM: regions that most influenced the AI finding", small),
                ],
            ],
            colWidths=[89 * mm, 89 * mm],
        ),
    ]

    if study.warnings:
        story += [Paragraph("Input quality warnings", h2)] + [
            Paragraph(f"- {w}", body) for w in study.warnings
        ]

    story.append(Paragraph("Clinician review", h2))
    if study.review_status == "pending":
        story.append(Paragraph("Not yet reviewed.", body))
    else:
        verdict = (
            "Agreed with AI finding"
            if study.review_status == "agreed"
            else (f"Disagreed - clinician label: <b>{LABELS.get(study.review_label, study.review_label)}</b>")
        )
        story.append(Paragraph(f"{verdict} - {study.reviewer}, {study.reviewed_at.replace('T', ' ')}", body))
        if study.review_note:
            story.append(Paragraph(f"Note: {study.review_note}", body))

    m = model_metrics or {}
    perf = (
        (
            f"Validated on a leakage-free held-out test set: balanced accuracy "
            f"{m.get('balanced_accuracy', 0):.1%}, macro AUC {m.get('macro_roc_auc', 0):.3f}, "
            f"disease sensitivity {m.get('abnormal_sensitivity', 0):.1%}. "
        )
        if m
        else ""
    )
    story += [
        Paragraph("Model information and limitations", h2),
        Paragraph(
            perf + "Trained on a public dataset aggregated from several sources; source-related "
            "shortcuts are documented and performance on other hospitals is not established. "
            "Grad-CAM is a qualitative aid, not a localisation guarantee.",
            small,
        ),
        Spacer(1, 8),
        Paragraph(f"Generated {generated_at.replace('T', ' ')} UTC for {generated_for}.", small),
    ]
    doc.build(story)
    return buf.getvalue()
