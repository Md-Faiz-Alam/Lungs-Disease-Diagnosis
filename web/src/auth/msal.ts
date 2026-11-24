import { PublicClientApplication, type Configuration } from '@azure/msal-browser'
import { config } from '../config'

export const loginScopes = ['openid', 'profile', config.entra.apiScope].filter(Boolean)

const msalConfig: Configuration = {
  auth: {
    clientId: config.entra.clientId,
    authority: `https://login.microsoftonline.com/${config.entra.tenantId}`,
    redirectUri: window.location.origin,
    postLogoutRedirectUri: window.location.origin,
  },
  // sessionStorage: tokens are cleared when the tab closes (shared clinical workstations)
  cache: { cacheLocation: 'sessionStorage' },
}

export const msalInstance: PublicClientApplication | null =
  config.authMode === 'entra' ? new PublicClientApplication(msalConfig) : null
