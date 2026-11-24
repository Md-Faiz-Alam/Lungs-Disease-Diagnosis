import clsx from 'clsx'
import { AlertTriangle, CheckCircle2, CircleDot, Clock, ShieldAlert, XCircle } from 'lucide-react'
import type { ReviewStatus, Triage } from '../api/types'
import { REVIEW_LABEL, TRIAGE_LABEL } from '../lib/format'

const TRIAGE_STYLE: Record<Triage, string> = {
  urgent: 'bg-urgent-bg text-urgent ring-urgent/20',
  review: 'bg-review-bg text-review ring-review/20',
  routine: 'bg-routine-bg text-routine ring-routine/20',
}
const TRIAGE_ICON = { urgent: ShieldAlert, review: AlertTriangle, routine: CircleDot }

export function TriageBadge({ triage, size = 'sm' }: { triage: Triage; size?: 'sm' | 'lg' }) {
  const Icon = TRIAGE_ICON[triage]
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-full font-semibold ring-1 ring-inset',
        TRIAGE_STYLE[triage],
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm',
      )}
    >
      <Icon aria-hidden className={size === 'sm' ? 'size-3.5' : 'size-4'} />
      {TRIAGE_LABEL[triage]}
    </span>
  )
}

const REVIEW_STYLE: Record<ReviewStatus, string> = {
  pending: 'text-slate-500',
  agreed: 'text-routine',
  disagreed: 'text-review',
}
const REVIEW_ICON = { pending: Clock, agreed: CheckCircle2, disagreed: XCircle }

export function ReviewBadge({ status }: { status: ReviewStatus }) {
  const Icon = REVIEW_ICON[status]
  return (
    <span className={clsx('inline-flex items-center gap-1 text-xs font-medium', REVIEW_STYLE[status])}>
      <Icon aria-hidden className="size-3.5" />
      {REVIEW_LABEL[status]}
    </span>
  )
}
