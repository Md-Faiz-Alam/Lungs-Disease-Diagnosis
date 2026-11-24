import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, ArrowLeft, Download, Info, Loader2, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { ClassLabel, Study } from '../api/types'
import { ReviewBadge, TriageBadge } from '../components/Badges'
import { ProbabilityBars } from '../components/ProbabilityBars'
import { Viewer } from '../components/Viewer'
import { FINDING_LABEL, formatDateTime, pct } from '../lib/format'

export function StudyPage() {
  const { id = '' } = useParams()
  const { data: study, isLoading, error } = useQuery({ queryKey: ['study', id], queryFn: () => api.study(id) })

  if (isLoading) return <p className="text-sm text-slate-500">Loading study…</p>
  if (error || !study)
    return (
      <div className="mx-auto max-w-lg py-16 text-center">
        <h1 className="text-lg font-semibold">Study not found</h1>
        <p className="mt-1 text-sm text-slate-500">It may have been deleted, or it belongs to another clinician.</p>
        <Link to="/worklist" className="btn-secondary mt-5">
          Back to worklist
        </Link>
      </div>
    )

  return (
    <div className="mx-auto max-w-[1500px]">
      <div className="flex flex-wrap items-center gap-3">
        <Link to="/worklist" className="btn-ghost -ml-2 px-2" aria-label="Back to worklist">
          <ArrowLeft aria-hidden className="size-4" />
        </Link>
        <h1 className="font-mono text-lg font-semibold">{study.id}</h1>
        <TriageBadge triage={study.triage} size="lg" />
        <ReviewBadge status={study.review_status} />
        <span className="ml-auto text-xs text-slate-500">Received {formatDateTime(study.created_at)}</span>
      </div>

      <div className="mt-4 grid gap-5 xl:grid-cols-[minmax(0,1fr)_400px]">
        <div className="h-[72vh] min-h-[480px]">
          <Viewer studyId={study.id} alt={`Chest radiograph for study ${study.id}`} />
        </div>
        <aside className="space-y-4">
          <FindingCard study={study} />
          <ReviewCard study={study} />
          <DetailsCard study={study} />
          <Actions study={study} />
        </aside>
      </div>
    </div>
  )
}

function FindingCard({ study }: { study: Study }) {
  return (
    <section className="card p-5" aria-labelledby="finding-h">
      <p className="label" id="finding-h">
        AI finding
      </p>
      <p className="mt-1 text-xl font-semibold">{FINDING_LABEL[study.prediction]}</p>
      <p className="mt-0.5 text-sm text-slate-500">
        Calibrated confidence <span className="font-mono font-medium text-slate-800">{pct(study.confidence)}</span>
      </p>
      {study.needs_review && (
        <p className="mt-3 flex gap-2 rounded-lg bg-review-bg p-3 text-sm text-review">
          <AlertTriangle aria-hidden className="mt-0.5 size-4 shrink-0" />
          Low model confidence — refer to a radiologist before acting on this finding.
        </p>
      )}
      <div className="mt-4">
        <ProbabilityBars probabilities={study.probabilities} predicted={study.prediction} />
      </div>
      {study.warnings.length > 0 && (
        <ul className="mt-4 space-y-1.5 border-t border-slate-100 pt-3 text-sm text-review">
          {study.warnings.map((w) => (
            <li key={w} className="flex gap-2">
              <Info aria-hidden className="mt-0.5 size-4 shrink-0" />
              {w}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

function ReviewCard({ study }: { study: Study }) {
  const qc = useQueryClient()
  const [mode, setMode] = useState<'idle' | 'override'>('idle')
  const [label, setLabel] = useState<ClassLabel>(study.prediction === 'NORMAL' ? 'PNEUMONIA' : 'NORMAL')
  const [note, setNote] = useState('')
  const mutation = useMutation({
    mutationFn: (v: { status: 'agreed' | 'disagreed'; label: string }) => api.review(study.id, v.status, v.label, note),
    onSuccess: (updated) => {
      qc.setQueryData(['study', study.id], updated)
      void qc.invalidateQueries({ queryKey: ['studies'] })
      setMode('idle')
    },
  })

  return (
    <section className="card p-5" aria-labelledby="review-h">
      <p className="label" id="review-h">
        Clinician review
      </p>
      {study.review_status !== 'pending' && (
        <div className="mt-2 rounded-lg bg-slate-50 p-3 text-sm">
          <p>
            <span className="font-medium">{study.review_status === 'agreed' ? 'Confirmed' : 'Overridden'}</span> as{' '}
            <span className="font-medium">{FINDING_LABEL[study.review_label as ClassLabel] ?? study.review_label}</span>
          </p>
          <p className="mt-0.5 text-xs text-slate-500">
            {study.reviewer} · {formatDateTime(study.reviewed_at)}
          </p>
          {study.review_note && <p className="mt-2 text-slate-700">“{study.review_note}”</p>}
        </div>
      )}
      <label className="mt-3 block text-sm">
        <span className="text-slate-600">Note (optional — do not enter patient identifiers)</span>
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          maxLength={1000}
          rows={2}
          className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
        />
      </label>
      {mode === 'override' && (
        <fieldset className="mt-3">
          <legend className="text-sm text-slate-600">Correct finding</legend>
          <div className="mt-1.5 flex flex-wrap gap-2">
            {(Object.keys(FINDING_LABEL) as ClassLabel[]).map((c) => (
              <label key={c} className="flex items-center gap-1.5 rounded-lg border border-slate-300 px-2.5 py-1.5 text-sm has-checked:border-brand-500 has-checked:bg-brand-50">
                <input type="radio" name="label" value={c} checked={label === c} onChange={() => setLabel(c)} />
                {FINDING_LABEL[c]}
              </label>
            ))}
          </div>
        </fieldset>
      )}
      <div className="mt-4 flex gap-2">
        {mode === 'idle' ? (
          <>
            <button
              type="button"
              className="btn-primary flex-1"
              disabled={mutation.isPending}
              onClick={() => mutation.mutate({ status: 'agreed', label: study.prediction })}
            >
              Confirm finding
            </button>
            <button type="button" className="btn-secondary flex-1" onClick={() => setMode('override')}>
              Override
            </button>
          </>
        ) : (
          <>
            <button
              type="button"
              className="btn-primary flex-1"
              disabled={mutation.isPending}
              onClick={() => mutation.mutate({ status: 'disagreed', label })}
            >
              Save override
            </button>
            <button type="button" className="btn-ghost" onClick={() => setMode('idle')}>
              Cancel
            </button>
          </>
        )}
      </div>
      {mutation.error && <p className="mt-2 text-sm text-urgent">{(mutation.error as Error).message}</p>}
    </section>
  )
}

function DetailsCard({ study }: { study: Study }) {
  const rows: [string, string][] = [
    ['Source', study.source === 'dicom' ? 'DICOM (de-identified)' : `Image · ${study.filename}`],
    ['Stored resolution', `${study.image_width} × ${study.image_height}`],
    ...Object.entries(study.dicom_metadata).map(([k, v]) => [k.replace(/([a-z])([A-Z])/g, '$1 $2'), v] as [string, string]),
    ['Model', `${study.model_arch} · ${study.model_run_id}`],
  ]
  return (
    <section className="card p-5" aria-labelledby="details-h">
      <p className="label" id="details-h">
        Study details
      </p>
      <dl className="mt-2 divide-y divide-slate-100 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 py-1.5">
            <dt className="text-slate-500">{k}</dt>
            <dd className="truncate text-right font-medium" title={v}>
              {v || '—'}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

function Actions({ study }: { study: Study }) {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [downloading, setDownloading] = useState(false)
  const del = useMutation({
    mutationFn: () => api.remove(study.id),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['studies'] })
      navigate('/worklist')
    },
  })
  return (
    <div className="flex gap-2">
      <button
        type="button"
        className="btn-secondary flex-1"
        disabled={downloading}
        onClick={async () => {
          setDownloading(true)
          try {
            await api.downloadReport(study.id)
          } finally {
            setDownloading(false)
          }
        }}
      >
        {downloading ? <Loader2 aria-hidden className="size-4 animate-spin" /> : <Download aria-hidden className="size-4" />}
        PDF report
      </button>
      <button
        type="button"
        className="btn-ghost text-urgent hover:bg-urgent-bg"
        disabled={del.isPending}
        onClick={() => {
          if (window.confirm(`Delete study ${study.id}? This permanently removes the images and result.`)) del.mutate()
        }}
      >
        <Trash2 aria-hidden className="size-4" /> Delete
      </button>
    </div>
  )
}
