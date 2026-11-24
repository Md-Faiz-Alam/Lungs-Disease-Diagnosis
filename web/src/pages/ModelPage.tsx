import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { FINDING_LABEL, formatDateTime, pct } from '../lib/format'
import type { ClassLabel } from '../api/types'

const LIMITATIONS = [
  'Trained on a public dataset assembled from several sources. File metadata alone predicts the class with 87% balanced accuracy, so part of the performance may come from source differences rather than disease.',
  'Near-perfect COVID-19 detection is likely partly a source shortcut; performance on other hospitals and scanners has not been established.',
  'The most common error is pneumonia called normal with high confidence, often on paediatric films. Calibration cannot flag these cases.',
  'Grad-CAM heatmaps are a qualitative aid and do not guarantee correct localisation.',
  'Not validated for lateral views, paediatric-specific use, or diseases outside the three trained classes.',
]

export function ModelPage() {
  const { data: m, isLoading, error } = useQuery({ queryKey: ['model'], queryFn: api.model, staleTime: Infinity })
  if (isLoading) return <p className="text-sm text-slate-500">Loading model information…</p>
  if (error || !m) return <p className="text-sm text-urgent">Model information unavailable.</p>
  const k = m.metrics

  return (
    <div className="mx-auto max-w-5xl">
      <h1 className="text-2xl font-semibold tracking-tight">Model &amp; validation</h1>
      <p className="mt-1 text-sm text-slate-500">
        {m.arch} · version <span className="font-mono">{m.run_id}</span> · built {formatDateTime(m.created_at)}
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric label="Balanced accuracy" value={k.balanced_accuracy} />
        <Metric label="Disease sensitivity" value={k.abnormal_sensitivity} />
        <Metric label="Macro AUC" value={k.macro_roc_auc} raw />
        <Metric label="Calibration error (ECE)" value={k.ece} raw />
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <section className="card p-5">
          <h2 className="font-semibold">Sensitivity by finding</h2>
          <ul className="mt-3 space-y-2 text-sm">
            {Object.entries(k.per_class_recall ?? {}).map(([c, r]) => (
              <li key={c} className="flex justify-between">
                <span>{FINDING_LABEL[c as ClassLabel] ?? c}</span>
                <span className="font-mono tabular-nums">{pct(r)}</span>
              </li>
            ))}
          </ul>
          <h2 className="mt-6 font-semibold">How confidence is produced</h2>
          <p className="mt-2 text-sm text-slate-600">
            Raw scores are temperature-calibrated (T = {m.temperature.toFixed(2)}) on a validation set, so a 90%
            confidence is right about 90% of the time. Findings below {pct(m.referral_threshold, 0)} are flagged
            <em> Needs review</em>.
          </p>
        </section>
        <section className="card p-5">
          <h2 className="font-semibold">Validation method</h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-slate-600">
            <li>1,243 held-out test images; 45 near-duplicates of training images removed.</li>
            <li>69 images with conflicting duplicate labels excluded.</li>
            <li>95% bootstrap confidence intervals reported for every headline metric.</li>
            <li>Quality gate: a new model is deployed only if it beats the current one.</li>
          </ul>
          <h2 className="mt-6 font-semibold">Known limitations</h2>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-slate-600">
            {LIMITATIONS.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  )
}

function Metric({ label, value, raw = false }: { label: string; value?: number; raw?: boolean }) {
  return (
    <div className="card p-4">
      <p className="label">{label}</p>
      <p className="mt-2 text-2xl font-semibold tabular-nums">
        {value === undefined ? '—' : raw ? value.toFixed(3) : pct(value)}
      </p>
    </div>
  )
}
