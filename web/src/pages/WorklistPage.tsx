import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { FileImage, FileScan, Inbox, PlusCircle, Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { Study, Triage } from '../api/types'
import { ReviewBadge, TriageBadge } from '../components/Badges'
import { FINDING_LABEL, pct, sortWorklist, timeAgo } from '../lib/format'

type Filter = 'all' | Triage | 'reviewed'
const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'urgent', label: 'Urgent' },
  { key: 'review', label: 'Needs review' },
  { key: 'routine', label: 'Routine' },
  { key: 'reviewed', label: 'Signed off' },
]

export function WorklistPage() {
  const { data, isLoading, error } = useQuery({ queryKey: ['studies'], queryFn: api.studies })
  const [filter, setFilter] = useState<Filter>('all')
  const [query, setQuery] = useState('')
  const navigate = useNavigate()

  const studies = useMemo(() => sortWorklist(data ?? []), [data])
  const pending = studies.filter((s) => s.review_status === 'pending')
  const visible = studies.filter((s) => {
    if (filter === 'reviewed' && s.review_status === 'pending') return false
    if (filter !== 'all' && filter !== 'reviewed' && (s.triage !== filter || s.review_status !== 'pending')) return false
    return !query || s.id.toLowerCase().includes(query.toLowerCase()) || s.filename.toLowerCase().includes(query.toLowerCase())
  })

  return (
    <div className="mx-auto max-w-7xl">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Worklist</h1>
          <p className="mt-1 text-sm text-slate-500">Unreviewed studies first, ordered by AI triage priority.</p>
        </div>
        <Link to="/studies/new" className="btn-primary">
          <PlusCircle aria-hidden className="size-4" /> New study
        </Link>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi label="Urgent, unreviewed" value={pending.filter((s) => s.triage === 'urgent').length} tone="urgent" />
        <Kpi label="Needs review" value={pending.filter((s) => s.triage === 'review').length} tone="review" />
        <Kpi label="Routine, unreviewed" value={pending.filter((s) => s.triage === 'routine').length} tone="routine" />
        <Kpi label="Signed off" value={studies.length - pending.length} />
      </div>

      <div className="card mt-6 overflow-hidden">
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 px-4 py-3">
          <div className="flex flex-wrap gap-1" role="tablist" aria-label="Filter studies">
            {FILTERS.map((f) => (
              <button
                key={f.key}
                type="button"
                role="tab"
                aria-selected={filter === f.key}
                onClick={() => setFilter(f.key)}
                className={clsx(
                  'rounded-md px-2.5 py-1 text-sm font-medium',
                  filter === f.key ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100',
                )}
              >
                {f.label}
              </button>
            ))}
          </div>
          <label className="relative ml-auto">
            <span className="sr-only">Search studies</span>
            <Search aria-hidden className="absolute left-2.5 top-2.5 size-4 text-slate-400" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search study ID or file"
              className="w-60 rounded-lg border border-slate-300 py-1.5 pl-8 pr-3 text-sm placeholder:text-slate-400"
            />
          </label>
        </div>

        {isLoading && <p className="p-8 text-center text-sm text-slate-500">Loading worklist…</p>}
        {error && <p className="p-8 text-center text-sm text-urgent">Could not load studies: {(error as Error).message}</p>}
        {!isLoading && !error && studies.length === 0 && <EmptyState />}
        {!isLoading && !error && studies.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Priority</th>
                  <th className="px-4 py-2.5 font-medium">Study</th>
                  <th className="px-4 py-2.5 font-medium">AI finding</th>
                  <th className="px-4 py-2.5 font-medium">Confidence</th>
                  <th className="px-4 py-2.5 font-medium">Status</th>
                  <th className="px-4 py-2.5 font-medium">Received</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {visible.map((s) => (
                  <Row key={s.id} study={s} onOpen={() => navigate(`/studies/${s.id}`)} />
                ))}
                {visible.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-slate-500">
                      No studies match this filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function Row({ study: s, onOpen }: { study: Study; onOpen: () => void }) {
  const Icon = s.source === 'dicom' ? FileScan : FileImage
  return (
    <tr
      className="cursor-pointer hover:bg-slate-50"
      onClick={onOpen}
      onKeyDown={(e) => e.key === 'Enter' && onOpen()}
      tabIndex={0}
      aria-label={`Open study ${s.id}`}
    >
      <td className="px-4 py-3">
        <TriageBadge triage={s.triage} />
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <Icon aria-hidden className="size-4 text-slate-400" />
          <div>
            <p className="font-mono text-xs font-medium text-slate-900">{s.id}</p>
            <p className="max-w-56 truncate text-xs text-slate-500">{s.filename}</p>
          </div>
        </div>
      </td>
      <td className="px-4 py-3 font-medium">{FINDING_LABEL[s.prediction]}</td>
      <td className="px-4 py-3 font-mono text-xs tabular-nums">
        {pct(s.confidence)}
        {s.needs_review && <span className="ml-1.5 text-review">low</span>}
      </td>
      <td className="px-4 py-3">
        <ReviewBadge status={s.review_status} />
      </td>
      <td className="px-4 py-3 text-xs text-slate-500" title={s.created_at}>
        {timeAgo(s.created_at)}
      </td>
    </tr>
  )
}

function Kpi({ label, value, tone }: { label: string; value: number; tone?: Triage }) {
  return (
    <div className="card p-4">
      <p className="label">{label}</p>
      <p
        className={clsx(
          'mt-2 text-3xl font-semibold tabular-nums',
          tone === 'urgent' && 'text-urgent',
          tone === 'review' && 'text-review',
          tone === 'routine' && 'text-routine',
        )}
      >
        {value}
      </p>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center px-6 py-14 text-center">
      <Inbox aria-hidden className="size-10 text-slate-300" />
      <h2 className="mt-3 font-semibold">No studies yet</h2>
      <p className="mt-1 max-w-sm text-sm text-slate-500">
        Upload a chest radiograph (DICOM, PNG or JPEG) to get an AI triage finding with an explanation heatmap.
      </p>
      <Link to="/studies/new" className="btn-primary mt-5">
        <PlusCircle aria-hidden className="size-4" /> Upload first study
      </Link>
    </div>
  )
}
