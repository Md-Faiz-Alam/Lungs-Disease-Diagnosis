import { useMutation, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { CheckCircle2, Loader2, UploadCloud, XCircle } from 'lucide-react'
import { useCallback, useRef, useState, type DragEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { Study } from '../api/types'
import { TriageBadge } from '../components/Badges'
import { FINDING_LABEL, pct } from '../lib/format'

const ACCEPT = '.dcm,.dicom,image/png,image/jpeg,application/dicom'
const MAX_MB = 25
const SAMPLES = [
  { file: 'COVID19_1.jpg', label: 'Sample A' },
  { file: 'NORMAL_1.jpg', label: 'Sample B' },
  { file: 'PNEUMONIA_1.jpg', label: 'Sample C' },
]

interface Item {
  key: string
  name: string
  state: 'uploading' | 'done' | 'error'
  study?: Study
  error?: string
}

export function NewStudyPage() {
  const [items, setItems] = useState<Item[]>([])
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const qc = useQueryClient()
  const navigate = useNavigate()

  const upload = useMutation({ mutationFn: api.createStudy })

  const process = useCallback(
    async (files: File[]) => {
      const batch = files.map((f, i) => ({ key: `${Date.now()}-${i}`, name: f.name, state: 'uploading' as const }))
      setItems((prev) => [...batch, ...prev])
      const results: Study[] = []
      for (const [i, file] of files.entries()) {
        const key = batch[i].key
        const tooBig = file.size > MAX_MB * 1024 * 1024
        try {
          if (tooBig) throw new Error(`File is larger than ${MAX_MB} MB`)
          const study = await upload.mutateAsync(file)
          results.push(study)
          setItems((prev) => prev.map((it) => (it.key === key ? { ...it, state: 'done', study } : it)))
        } catch (e) {
          const error = e instanceof Error ? e.message : 'Upload failed'
          setItems((prev) => prev.map((it) => (it.key === key ? { ...it, state: 'error', error } : it)))
        }
      }
      await qc.invalidateQueries({ queryKey: ['studies'] })
      if (files.length === 1 && results.length === 1) navigate(`/studies/${results[0].id}`)
    },
    [navigate, qc, upload],
  )

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length) void process(files)
  }

  const loadSample = async (file: string) => {
    const blob = await (await fetch(`/samples/${file}`)).blob()
    void process([new File([blob], `sample-${file}`, { type: 'image/jpeg' })])
  }

  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="text-2xl font-semibold tracking-tight">New study</h1>
      <p className="mt-1 text-sm text-slate-500">
        Upload frontal (PA/AP) chest radiographs. DICOM files are de-identified on receipt — patient name, IDs and dates
        are never stored.
      </p>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={clsx(
          'mt-6 flex flex-col items-center rounded-2xl border-2 border-dashed px-6 py-14 text-center transition-colors',
          dragging ? 'border-brand-500 bg-brand-50' : 'border-slate-300 bg-white',
        )}
      >
        <UploadCloud aria-hidden className="size-10 text-brand-600" />
        <p className="mt-3 font-medium">Drag & drop radiographs here</p>
        <p className="mt-1 text-sm text-slate-500">DICOM (.dcm), PNG or JPEG · up to {MAX_MB} MB each · multiple files allowed</p>
        <button type="button" className="btn-primary mt-5" onClick={() => input.current?.click()}>
          Browse files
        </button>
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          multiple
          className="sr-only"
          aria-label="Choose radiograph files"
          onChange={(e) => {
            const files = Array.from(e.target.files ?? [])
            e.target.value = ''
            if (files.length) void process(files)
          }}
        />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="text-slate-500">No X-ray at hand? Try a de-identified sample:</span>
        {SAMPLES.map((s) => (
          <button key={s.file} type="button" className="btn-secondary py-1" onClick={() => void loadSample(s.file)}>
            {s.label}
          </button>
        ))}
      </div>

      {items.length > 0 && (
        <ul className="card mt-6 divide-y divide-slate-100">
          {items.map((it) => (
            <li key={it.key} className="flex items-center gap-3 px-4 py-3 text-sm">
              {it.state === 'uploading' && <Loader2 aria-label="Analysing" className="size-4 animate-spin text-brand-600" />}
              {it.state === 'done' && <CheckCircle2 aria-label="Done" className="size-4 text-routine" />}
              {it.state === 'error' && <XCircle aria-label="Failed" className="size-4 text-urgent" />}
              <span className="min-w-0 flex-1 truncate">{it.name}</span>
              {it.state === 'uploading' && <span className="text-slate-500">Analysing…</span>}
              {it.state === 'error' && <span className="text-urgent">{it.error}</span>}
              {it.study && (
                <>
                  <TriageBadge triage={it.study.triage} />
                  <span className="hidden text-slate-600 sm:inline">
                    {FINDING_LABEL[it.study.prediction]} · {pct(it.study.confidence)}
                  </span>
                  <Link to={`/studies/${it.study.id}`} className="font-medium text-brand-600 hover:underline">
                    Open
                  </Link>
                </>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
