import type { ClassLabel, ReviewStatus, Study, Triage } from '../api/types'

export const FINDING_LABEL: Record<ClassLabel, string> = {
  COVID19: 'COVID-19 pattern',
  NORMAL: 'No acute finding',
  PNEUMONIA: 'Pneumonia pattern',
}

export const TRIAGE_LABEL: Record<Triage, string> = {
  urgent: 'Urgent',
  review: 'Needs review',
  routine: 'Routine',
}

export const REVIEW_LABEL: Record<ReviewStatus, string> = {
  pending: 'Awaiting review',
  agreed: 'Confirmed',
  disagreed: 'Overridden',
}

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`

export function formatDateTime(iso: string): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function timeAgo(iso: string, now: Date = new Date()): string {
  const s = Math.max(0, Math.round((now.getTime() - new Date(iso).getTime()) / 1000))
  if (s < 60) return 'just now'
  const m = Math.round(s / 60)
  if (m < 60) return `${m} min ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h} h ago`
  return `${Math.round(h / 24)} d ago`
}

const TRIAGE_ORDER: Record<Triage, number> = { urgent: 0, review: 1, routine: 2 }

/** Worklist order: unreviewed first, then by urgency, newest first (mirrors the API). */
export function sortWorklist(studies: Study[]): Study[] {
  return [...studies].sort((a, b) => {
    const pa = a.review_status === 'pending' ? 0 : 1
    const pb = b.review_status === 'pending' ? 0 : 1
    if (pa !== pb) return pa - pb
    if (a.triage !== b.triage) return TRIAGE_ORDER[a.triage] - TRIAGE_ORDER[b.triage]
    return b.created_at.localeCompare(a.created_at)
  })
}
