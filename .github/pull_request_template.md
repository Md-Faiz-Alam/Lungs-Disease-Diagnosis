## What and why

<!-- One or two sentences: what changes and the reason. -->

## How it was tested

- [ ] `pytest` passes (coverage gate 88%)
- [ ] `npm test`, `npm run typecheck` and `npm run lint` pass in `web/`
- [ ] Tried in the browser (for UI changes)

## Risk checklist

- [ ] No patient data, credentials or secrets added
- [ ] Model or preprocessing change? Re-evaluated on the clean test set and README results updated
- [ ] Infrastructure change? `az bicep build --file infra/main.bicep` succeeds
