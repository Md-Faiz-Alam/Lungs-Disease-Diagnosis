/**
 * Anonymous workspace ID for the public demo. Each browser gets a random UUID, sent as the
 * X-Demo-Session header, so visitors only ever see their own uploads. Kept in localStorage so the
 * worklist survives reloads; "Sign out" discards it (fresh, empty workspace).
 */
const KEY = 'lungxray.demo-workspace'
let memoryFallback: string | null = null

function newId(): string {
  return crypto.randomUUID()
}

export function getDemoSessionId(): string {
  try {
    let id = localStorage.getItem(KEY)
    if (!id) {
      id = newId()
      localStorage.setItem(KEY, id)
    }
    return id
  } catch {
    // storage blocked (private mode): keep a per-tab ID in memory
    memoryFallback ??= newId()
    return memoryFallback
  }
}

export function resetDemoSession(): void {
  memoryFallback = null
  try {
    localStorage.removeItem(KEY)
  } catch {
    /* ignore */
  }
}
