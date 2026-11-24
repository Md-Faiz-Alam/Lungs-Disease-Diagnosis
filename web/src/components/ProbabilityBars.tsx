import clsx from 'clsx'
import type { ClassLabel } from '../api/types'
import { FINDING_LABEL, pct } from '../lib/format'

const BAR: Record<ClassLabel, string> = {
  COVID19: 'bg-urgent',
  PNEUMONIA: 'bg-review',
  NORMAL: 'bg-routine',
}

export function ProbabilityBars({
  probabilities,
  predicted,
}: {
  probabilities: Record<ClassLabel, number>
  predicted: ClassLabel
}) {
  const rows = (Object.entries(probabilities) as [ClassLabel, number][]).sort((a, b) => b[1] - a[1])
  return (
    <ul className="space-y-2.5" aria-label="Class probabilities">
      {rows.map(([label, p]) => (
        <li key={label}>
          <div className="mb-1 flex items-baseline justify-between text-sm">
            <span className={clsx(label === predicted ? 'font-semibold text-slate-900' : 'text-slate-600')}>
              {FINDING_LABEL[label]}
            </span>
            <span className="font-mono text-xs tabular-nums text-slate-600">{pct(p)}</span>
          </div>
          <div
            className="h-2 overflow-hidden rounded-full bg-slate-100"
            role="meter"
            aria-label={FINDING_LABEL[label]}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(p * 100)}
          >
            <div className={clsx('h-full rounded-full', BAR[label])} style={{ width: `${Math.max(p * 100, 1)}%` }} />
          </div>
        </li>
      ))}
    </ul>
  )
}
