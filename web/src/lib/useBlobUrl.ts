import { useEffect, useState } from 'react'
import { api } from '../api/client'

/** Fetch an authenticated binary resource and expose it as an object URL (revoked on change/unmount). */
export function useBlobUrl(path: string | null): { url: string | null; error: string | null } {
  const [state, setState] = useState<{ url: string | null; error: string | null }>({ url: null, error: null })

  useEffect(() => {
    if (!path) return
    let revoked = false
    let created: string | null = null
    api
      .blobUrl(path)
      .then((url) => {
        if (revoked) {
          URL.revokeObjectURL(url)
          return
        }
        created = url
        setState({ url, error: null })
      })
      .catch((e: unknown) => !revoked && setState({ url: null, error: e instanceof Error ? e.message : 'Failed' }))
    return () => {
      revoked = true
      if (created) URL.revokeObjectURL(created)
    }
  }, [path])

  return state
}
