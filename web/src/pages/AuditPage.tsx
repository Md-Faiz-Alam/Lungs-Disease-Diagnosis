import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { formatDateTime } from '../lib/format'

const ACTION_LABEL: Record<string, string> = {
  'study.create': 'Uploaded study',
  'study.view': 'Viewed study',
  'study.review.agreed': 'Confirmed AI finding',
  'study.review.disagreed': 'Overrode AI finding',
  'study.delete': 'Deleted study',
  'study.rejected': 'Upload rejected',
  'report.download': 'Downloaded report',
}

export function AuditPage() {
  const { data, isLoading, error } = useQuery({ queryKey: ['audit'], queryFn: api.audit })
  return (
    <div className="mx-auto max-w-5xl">
      <h1 className="text-2xl font-semibold tracking-tight">Audit log</h1>
      <p className="mt-1 text-sm text-slate-500">Every access and decision on your studies, newest first.</p>
      <div className="card mt-6 overflow-x-auto">
        {isLoading && <p className="p-6 text-sm text-slate-500">Loading…</p>}
        {error && <p className="p-6 text-sm text-urgent">Could not load the audit log.</p>}
        {data && data.length === 0 && <p className="p-6 text-sm text-slate-500">No activity yet.</p>}
        {data && data.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-2.5 font-medium">Time</th>
                <th className="px-4 py-2.5 font-medium">Action</th>
                <th className="px-4 py-2.5 font-medium">Study</th>
                <th className="px-4 py-2.5 font-medium">Detail</th>
                <th className="px-4 py-2.5 font-medium">User</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.map((e, i) => (
                <tr key={`${e.ts}-${i}`}>
                  <td className="whitespace-nowrap px-4 py-2.5 text-slate-600">{formatDateTime(e.ts)}</td>
                  <td className="px-4 py-2.5 font-medium">{ACTION_LABEL[e.action] ?? e.action}</td>
                  <td className="px-4 py-2.5 font-mono text-xs">
                    {e.study_id ? <Link to={`/studies/${e.study_id}`} className="text-brand-600 hover:underline">{e.study_id}</Link> : '—'}
                  </td>
                  <td className="max-w-64 truncate px-4 py-2.5 text-slate-600" title={e.detail}>
                    {e.detail || '—'}
                  </td>
                  <td className="px-4 py-2.5 text-slate-600">{e.user_name}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
