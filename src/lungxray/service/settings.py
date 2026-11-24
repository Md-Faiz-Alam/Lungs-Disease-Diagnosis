"""Runtime settings, read from environment variables (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    model_path: str = "models/production/model.pt"
    prediction_log: str = "logs/predictions.jsonl"
    max_upload_mb: float = 25.0

    # auth: "demo" (single built-in demo user) or "entra" (Microsoft Entra ID bearer tokens)
    auth_mode: str = "demo"
    entra_tenant_id: str = ""
    entra_api_audience: str = ""  # e.g. api://<api-app-client-id>
    required_scope: str = "Studies.ReadWrite"

    # storage: "local" (sqlite + files, for dev/tests) or "azure" (Table + Blob, managed identity)
    storage_mode: str = "local"
    local_storage_dir: str = "var/storage"
    azure_storage_account: str = ""
    blob_container: str = "studies"

    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    legacy_api_key: str = ""  # optional X-API-Key for the original /predict endpoints

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        return cls(
            model_path=e("MODEL_PATH", cls.model_path),
            prediction_log=e("PREDICTION_LOG", cls.prediction_log),
            max_upload_mb=float(e("MAX_UPLOAD_MB", str(cls.max_upload_mb))),
            auth_mode=e("AUTH_MODE", cls.auth_mode).lower(),
            entra_tenant_id=e("ENTRA_TENANT_ID", ""),
            entra_api_audience=e("ENTRA_API_AUDIENCE", ""),
            required_scope=e("ENTRA_REQUIRED_SCOPE", cls.required_scope),
            storage_mode=e("STORAGE_MODE", cls.storage_mode).lower(),
            local_storage_dir=e("LOCAL_STORAGE_DIR", cls.local_storage_dir),
            azure_storage_account=e("AZURE_STORAGE_ACCOUNT", ""),
            blob_container=e("BLOB_CONTAINER", cls.blob_container),
            cors_origins=_list(e("CORS_ORIGINS", "http://localhost:5173")),
            legacy_api_key=e("API_KEY", ""),
        )

    def validate(self) -> None:
        if self.auth_mode not in {"demo", "entra"}:
            raise ValueError("AUTH_MODE must be 'demo' or 'entra'")
        if self.auth_mode == "entra" and not (self.entra_tenant_id and self.entra_api_audience):
            raise ValueError("AUTH_MODE=entra requires ENTRA_TENANT_ID and ENTRA_API_AUDIENCE")
        if self.storage_mode not in {"local", "azure"}:
            raise ValueError("STORAGE_MODE must be 'local' or 'azure'")
        if self.storage_mode == "azure" and not self.azure_storage_account:
            raise ValueError("STORAGE_MODE=azure requires AZURE_STORAGE_ACCOUNT")


@lru_cache
def get_settings() -> Settings:
    s = Settings.from_env()
    s.validate()
    return s
