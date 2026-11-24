"""Study persistence behind one interface, with two backends.

* ``LocalRepository``  - SQLite + files on disk (development, tests, docker-compose)
* ``AzureRepository``  - Azure Table Storage (records, audit) + Blob Storage (images), authenticated
  with the container's managed identity (DefaultAzureCredential). No keys or connection strings.

Every study is partitioned by its owner, so one clinician can never read another's studies.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from lungxray.service.settings import Settings


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_study_id() -> str:
    return f"ST-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class Study:
    id: str
    owner: str
    created_at: str
    filename: str
    source: str  # "dicom" | "image"
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    needs_review: bool
    triage: str
    warnings: list[str] = field(default_factory=list)
    dicom_metadata: dict[str, str] = field(default_factory=dict)
    image_width: int = 0
    image_height: int = 0
    model_arch: str = ""
    model_run_id: str = ""
    review_status: str = "pending"  # pending | agreed | disagreed
    review_label: str = ""
    review_note: str = ""
    reviewed_at: str = ""
    reviewer: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


JSON_FIELDS = ("probabilities", "warnings", "dicom_metadata")


@dataclass
class AuditEvent:
    user_id: str
    user_name: str
    action: str
    study_id: str = ""
    detail: str = ""
    ts: str = field(default_factory=now_iso)


class StudyRepository(ABC):
    @abstractmethod
    def save_study(self, study: Study) -> None: ...

    @abstractmethod
    def get_study(self, owner: str, study_id: str) -> Study | None: ...

    @abstractmethod
    def list_studies(self, owner: str) -> list[Study]: ...

    @abstractmethod
    def delete_study(self, owner: str, study_id: str) -> None: ...

    @abstractmethod
    def put_blob(self, key: str, data: bytes, content_type: str) -> None: ...

    @abstractmethod
    def get_blob(self, key: str) -> bytes | None: ...

    @abstractmethod
    def delete_blob(self, key: str) -> None: ...

    @abstractmethod
    def add_audit(self, event: AuditEvent) -> None: ...

    @abstractmethod
    def list_audit(self, user_id: str, limit: int = 200) -> list[AuditEvent]: ...

    @staticmethod
    def blob_key(owner: str, study_id: str, kind: str) -> str:
        return f"{owner}/{study_id}/{kind}.png"


def _row_to_study(row: dict) -> Study:
    data = dict(row)
    for f in JSON_FIELDS:
        if isinstance(data.get(f), str):
            data[f] = json.loads(data[f])
    data["needs_review"] = bool(data["needs_review"])
    return Study(**{k: v for k, v in data.items() if k in Study.__dataclass_fields__})


def _study_to_row(study: Study) -> dict:
    row = study.to_dict()
    for f in JSON_FIELDS:
        row[f] = json.dumps(row[f])
    return row


class LocalRepository(StudyRepository):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.root / "studies.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        cols = ", ".join(f"{name} TEXT" for name in Study.__dataclass_fields__ if name not in ("id", "owner"))
        with self._lock, self._db:
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS studies (id TEXT, owner TEXT, {cols}, PRIMARY KEY (owner, id))"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS audit (user_id TEXT, user_name TEXT, action TEXT, "
                "study_id TEXT, detail TEXT, ts TEXT)"
            )

    def save_study(self, study: Study) -> None:
        row = _study_to_row(study)
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock, self._db:
            self._db.execute(f"INSERT OR REPLACE INTO studies ({cols}) VALUES ({marks})", list(row.values()))

    def _convert(self, r: sqlite3.Row) -> Study:
        d = dict(r)
        d["confidence"] = float(d["confidence"])
        d["image_width"], d["image_height"] = int(d["image_width"]), int(d["image_height"])
        d["needs_review"] = d["needs_review"] in ("1", 1, True, "True", "true")
        return _row_to_study(d)

    def get_study(self, owner: str, study_id: str) -> Study | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM studies WHERE owner=? AND id=?", (owner, study_id)).fetchone()
        return self._convert(r) if r else None

    def list_studies(self, owner: str) -> list[Study]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM studies WHERE owner=?", (owner,)).fetchall()
        return [self._convert(r) for r in rows]

    def delete_study(self, owner: str, study_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM studies WHERE owner=? AND id=?", (owner, study_id))

    def _blob_path(self, key: str) -> Path:
        path = (self.root / "blobs" / key).resolve()
        if not str(path).startswith(str((self.root / "blobs").resolve())):
            raise ValueError("Invalid blob key")
        return path

    def put_blob(self, key: str, data: bytes, content_type: str) -> None:
        path = self._blob_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_blob(self, key: str) -> bytes | None:
        path = self._blob_path(key)
        return path.read_bytes() if path.exists() else None

    def delete_blob(self, key: str) -> None:
        self._blob_path(key).unlink(missing_ok=True)

    def add_audit(self, event: AuditEvent) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO audit VALUES (?, ?, ?, ?, ?, ?)",
                (event.user_id, event.user_name, event.action, event.study_id, event.detail, event.ts),
            )

    def list_audit(self, user_id: str, limit: int = 200) -> list[AuditEvent]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM audit WHERE user_id=? ORDER BY ts DESC LIMIT ?", (user_id, limit)
            ).fetchall()
        return [AuditEvent(**dict(r)) for r in rows]


class AzureRepository(StudyRepository):
    """Azure Table Storage + Blob Storage via managed identity."""

    def __init__(self, account: str, container: str = "studies"):
        from azure.data.tables import TableServiceClient
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        cred = DefaultAzureCredential()
        tables = TableServiceClient(f"https://{account}.table.core.windows.net", credential=cred)
        self._studies = tables.get_table_client("studies")
        self._audit = tables.get_table_client("audit")
        blobs = BlobServiceClient(f"https://{account}.blob.core.windows.net", credential=cred)
        self._container = blobs.get_container_client(container)

    def save_study(self, study: Study) -> None:
        row = _study_to_row(study)
        row["PartitionKey"], row["RowKey"] = study.owner, study.id
        self._studies.upsert_entity(row)

    def get_study(self, owner: str, study_id: str) -> Study | None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return _row_to_study(self._studies.get_entity(owner, study_id))
        except ResourceNotFoundError:
            return None

    def list_studies(self, owner: str) -> list[Study]:
        rows = self._studies.query_entities("PartitionKey eq @o", parameters={"o": owner})
        return [_row_to_study(r) for r in rows]

    def delete_study(self, owner: str, study_id: str) -> None:
        self._studies.delete_entity(owner, study_id)

    def put_blob(self, key: str, data: bytes, content_type: str) -> None:
        from azure.storage.blob import ContentSettings

        self._container.upload_blob(
            key, data, overwrite=True, content_settings=ContentSettings(content_type=content_type)
        )

    def get_blob(self, key: str) -> bytes | None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return self._container.download_blob(key).readall()
        except ResourceNotFoundError:
            return None

    def delete_blob(self, key: str) -> None:
        from azure.core.exceptions import ResourceNotFoundError

        with contextlib.suppress(ResourceNotFoundError):
            self._container.delete_blob(key)

    def add_audit(self, event: AuditEvent) -> None:
        # RowKey sorts newest-first: inverted timestamp + random suffix
        inv = f"{10**12 - int(datetime.fromisoformat(event.ts).timestamp()):013d}"
        self._audit.create_entity(
            {"PartitionKey": event.user_id, "RowKey": f"{inv}-{uuid.uuid4().hex[:6]}", **asdict(event)}
        )

    def list_audit(self, user_id: str, limit: int = 200) -> list[AuditEvent]:
        rows = self._audit.query_entities(
            "PartitionKey eq @u", parameters={"u": user_id}, results_per_page=limit
        )
        out = []
        for r in rows:
            out.append(AuditEvent(**{k: r[k] for k in AuditEvent.__dataclass_fields__}))
            if len(out) >= limit:
                break
        return out


def build_repository(settings: Settings) -> StudyRepository:
    if settings.storage_mode == "azure":
        return AzureRepository(settings.azure_storage_account, settings.blob_container)
    return LocalRepository(settings.local_storage_dir)
