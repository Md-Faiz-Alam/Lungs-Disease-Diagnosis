import { config } from '../config'
import type { AuditEvent, Me, ModelInfo, ReviewStatus, Study } from './types'

type TokenProvider = () => Promise<string | null>
let tokenProvider: TokenProvider = async () => null

/** Registered by AuthProvider so every request carries the current Entra access token. */
export function setTokenProvider(provider: TokenProvider) {
  tokenProvider = provider
}

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const token = await tokenProvider()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const res = await fetch(`${config.apiBaseUrl}${path}`, { ...init, headers })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail || `Request failed (${res.status})`)
  }
  return res
}

const json = async <T>(path: string, init?: RequestInit) => (await request(path, init)).json() as Promise<T>

export const api = {
  me: () => json<Me>('/api/me'),
  model: () => json<ModelInfo>('/api/model'),
  studies: () => json<Study[]>('/api/studies'),
  study: (id: string) => json<Study>(`/api/studies/${encodeURIComponent(id)}`),
  audit: () => json<AuditEvent[]>('/api/audit'),

  createStudy(file: File): Promise<Study> {
    const body = new FormData()
    body.append('file', file, file.name)
    return json<Study>('/api/studies', { method: 'POST', body })
  },

  review(id: string, status: Exclude<ReviewStatus, 'pending'>, label: string, note: string): Promise<Study> {
    return json<Study>(`/api/studies/${encodeURIComponent(id)}/review`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, label, note }),
    })
  },

  async remove(id: string): Promise<void> {
    await request(`/api/studies/${encodeURIComponent(id)}`, { method: 'DELETE' })
  },

  /** Authenticated binary fetch, returned as an object URL (caller must revoke). */
  async blobUrl(path: string): Promise<string> {
    const res = await request(path)
    return URL.createObjectURL(await res.blob())
  },

  async downloadReport(id: string): Promise<void> {
    const url = await api.blobUrl(`/api/studies/${encodeURIComponent(id)}/report.pdf`)
    const a = document.createElement('a')
    a.href = url
    a.download = `${id}.pdf`
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 10_000)
  },
}
