/** Build-time configuration (Vite env vars, see .env.example). */
export type AuthMode = 'demo' | 'entra'

const env = import.meta.env

export const config = {
  /** Base URL of the FastAPI backend; empty = same origin (dev proxy). */
  apiBaseUrl: (env.VITE_API_BASE_URL ?? '').replace(/\/$/, ''),
  authMode: (env.VITE_AUTH_MODE === 'entra' ? 'entra' : 'demo') as AuthMode,
  entra: {
    clientId: env.VITE_ENTRA_CLIENT_ID ?? '',
    tenantId: env.VITE_ENTRA_TENANT_ID ?? '',
    /** Delegated scope exposed by the API app registration, e.g. api://<api-id>/Studies.ReadWrite */
    apiScope: env.VITE_API_SCOPE ?? '',
  },
} as const
