import { InteractionRequiredAuthError, type AccountInfo } from '@azure/msal-browser'
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { setTokenProvider } from '../api/client'
import { config, type AuthMode } from '../config'
import { loginScopes, msalInstance } from './msal'

interface AuthUser {
  name: string
  email: string
}

interface AuthState {
  mode: AuthMode
  ready: boolean
  isAuthenticated: boolean
  user: AuthUser | null
  signIn: () => Promise<void>
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)
const DEMO_KEY = 'lungxray.demo-session'

function readDemoSession(): boolean {
  try {
    return sessionStorage.getItem(DEMO_KEY) === '1'
  } catch {
    return false
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(config.authMode === 'demo')
  const [account, setAccount] = useState<AccountInfo | null>(null)
  const [demoActive, setDemoActive] = useState(readDemoSession)

  useEffect(() => {
    if (!msalInstance) return
    const msal = msalInstance
    let cancelled = false
    ;(async () => {
      await msal.initialize()
      const result = await msal.handleRedirectPromise()
      const active = result?.account ?? msal.getActiveAccount() ?? msal.getAllAccounts()[0] ?? null
      if (active) msal.setActiveAccount(active)
      if (!cancelled) {
        setAccount(active)
        setReady(true)
      }
    })().catch((err) => {
      console.error('MSAL initialisation failed', err)
      if (!cancelled) setReady(true)
    })
    return () => {
      cancelled = true
    }
  }, [])

  const getToken = useCallback(async (): Promise<string | null> => {
    if (!msalInstance) return null
    const active = msalInstance.getActiveAccount()
    if (!active) return null
    const request = { scopes: [config.entra.apiScope], account: active }
    try {
      return (await msalInstance.acquireTokenSilent(request)).accessToken
    } catch (err) {
      if (err instanceof InteractionRequiredAuthError) {
        await msalInstance.acquireTokenRedirect(request)
        return null
      }
      throw err
    }
  }, [])

  useEffect(() => setTokenProvider(getToken), [getToken])

  const signIn = useCallback(async () => {
    if (msalInstance) {
      await msalInstance.loginRedirect({ scopes: loginScopes })
      return
    }
    try {
      sessionStorage.setItem(DEMO_KEY, '1')
    } catch {
      /* private mode: the demo session simply won't survive a reload */
    }
    setDemoActive(true)
  }, [])

  const signOut = useCallback(async () => {
    if (msalInstance) {
      await msalInstance.logoutRedirect({ account: msalInstance.getActiveAccount() ?? undefined })
      return
    }
    try {
      sessionStorage.removeItem(DEMO_KEY)
    } catch {
      /* ignore */
    }
    setDemoActive(false)
  }, [])

  const value = useMemo<AuthState>(() => {
    const isAuthenticated = config.authMode === 'entra' ? account !== null : demoActive
    const user = !isAuthenticated
      ? null
      : config.authMode === 'entra'
        ? { name: account?.name ?? 'Clinician', email: account?.username ?? '' }
        : { name: 'Demo Clinician', email: 'demo mode' }
    return { mode: config.authMode, ready, isAuthenticated, user, signIn, signOut }
  }, [account, demoActive, ready, signIn, signOut])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// oxlint-disable-next-line react/only-export-components
export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
