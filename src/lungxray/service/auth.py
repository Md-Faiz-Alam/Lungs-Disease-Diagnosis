"""Authentication.

* ``entra``: validates Microsoft Entra ID (Azure AD) v2.0 access tokens issued to the SPA for
  this API: signature via the tenant's JWKS, issuer, audience, expiry and the delegated scope.
* ``demo``: every request is a single shared demo clinician (public portfolio demo).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, Request, status

from lungxray.service.settings import Settings, get_settings


@dataclass(frozen=True)
class User:
    id: str
    name: str
    email: str = ""
    is_demo: bool = False


DEMO_USER = User(id="demo-clinician", name="Demo Clinician", email="", is_demo=True)


@lru_cache(maxsize=4)
def _jwks_client(tenant_id: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(
        f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys", cache_keys=True
    )


def decode_entra_token(token: str, settings: Settings) -> dict:
    key = _jwks_client(settings.entra_tenant_id).get_signing_key_from_jwt(token).key
    audiences = [settings.entra_api_audience]
    if settings.entra_api_audience.startswith("api://"):  # v2 tokens may carry the bare client id
        audiences.append(settings.entra_api_audience.removeprefix("api://"))
    return jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        audience=audiences,
        issuer=f"https://login.microsoftonline.com/{settings.entra_tenant_id}/v2.0",
        options={"require": ["exp", "iat", "iss", "aud"]},
    )


def user_from_claims(claims: dict, required_scope: str) -> User:
    scopes = set(str(claims.get("scp", "")).split())
    if required_scope and required_scope not in scopes:
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"Token lacks the '{required_scope}' scope")
    uid = claims.get("oid") or claims.get("sub")
    if not uid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token has no subject")
    return User(
        id=str(uid),
        name=str(claims.get("name") or claims.get("preferred_username") or "Clinician"),
        email=str(claims.get("preferred_username") or ""),
    )


def get_current_user(request: Request, settings: Settings = Depends(get_settings)) -> User:
    if settings.auth_mode == "demo":
        return DEMO_USER
    header = request.headers.get("Authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    try:
        claims = decode_entra_token(header.split(" ", 1)[1], settings)
    except jwt.PyJWTError as e:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, f"Invalid token: {e}", headers={"WWW-Authenticate": "Bearer"}
        ) from e
    return user_from_claims(claims, settings.required_scope)
