# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through
GitHub: **Security → Report a vulnerability** on this repository. You should get a response
within 7 days.

Include the affected component (API, web app, infrastructure), steps to reproduce, and impact.

## Scope and design notes

- **No patient data should ever be uploaded to the public demo.** Demo workspaces are anonymous,
  isolated per browser, and images are deleted automatically after 30 days.
- DICOM uploads are de-identified on receipt; only pixel data and technical tags are kept.
- Production uses Microsoft Entra ID sign-in, managed identities and OIDC deployment. No
  credentials are stored in the repository or in CI. See [docs/azure-deployment.md](docs/azure-deployment.md).
- Automated checks on every change: CodeQL static analysis, dependency audits (pip-audit,
  npm audit) and container image scanning (Trivy). Dependencies are pinned in
  `requirements-lock.txt` and `web/package-lock.json` and updated deliberately.

## Supported versions

Only the latest commit on `main` (what is deployed) receives fixes.
