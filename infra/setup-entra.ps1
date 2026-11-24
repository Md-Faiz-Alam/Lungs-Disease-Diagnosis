<#
.SYNOPSIS
  Optional: enable Microsoft Entra ID sign-in (run after the first successful deployment).

.DESCRIPTION
  Creates (or updates) two app registrations in your tenant:
    * "LungXray API" - exposes the delegated scope Studies.ReadWrite, issues v2.0 access tokens
    * "LungXray Web" - single-page application (PKCE, no secret) allowed to call that scope
  then switches the deployment to AUTH_MODE=entra via GitHub repository variables.

  Some university tenants block app registration for students. If this script fails with
  "Insufficient privileges", keep AUTH_MODE=demo or use a tenant you control.
#>
[CmdletBinding()]
param(
  [string]$Repo = "",
  [string]$ResourceGroup = "rg-lungxray-prod",
  [switch]$RestrictToAssignedUsers
)
$ErrorActionPreference = "Stop"
function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Invoke-Az { $out = & az @args; if ($LASTEXITCODE -ne 0) { throw "az $($args -join ' ') failed" }; return $out }
function Invoke-Graph($method, $url, $body) {
  $file = New-TemporaryFile
  $body | ConvertTo-Json -Depth 10 | Set-Content -Path $file -Encoding utf8
  Invoke-Az rest --method $method --url $url --headers "Content-Type=application/json" --body "@$file" | Out-Null
  Remove-Item $file
}
function Get-OrCreateApp($name) {
  $id = az ad app list --display-name $name --query "[0].appId" -o tsv
  if (-not $id) { $id = Invoke-Az ad app create --display-name $name --sign-in-audience AzureADMyOrg --query appId -o tsv }
  return $id
}

if (-not $Repo) { $Repo = (gh repo view --json nameWithOwner -q .nameWithOwner) }
$tenantId = Invoke-Az account show --query tenantId -o tsv
$webHost = Invoke-Az staticwebapp list -g $ResourceGroup --query "[0].defaultHostname" -o tsv
if (-not $webHost) { throw "No Static Web App found in $ResourceGroup. Run the Deploy workflow once first." }
$webUrl = "https://$webHost"

Step "API app registration"
$apiAppId = Get-OrCreateApp "LungXray API"
$apiObjectId = Invoke-Az ad app show --id $apiAppId --query id -o tsv
$audience = "api://$apiAppId"
$scopeId = Invoke-Az ad app show --id $apiAppId --query "api.oauth2PermissionScopes[?value=='Studies.ReadWrite'].id | [0]" -o tsv
if (-not $scopeId) { $scopeId = [guid]::NewGuid().ToString() }
Invoke-Graph PATCH "https://graph.microsoft.com/v1.0/applications/$apiObjectId" @{
  identifierUris = @($audience)
  api = @{
    requestedAccessTokenVersion = 2   # the API validates v2.0 issuer/audience
    oauth2PermissionScopes = @(@{
      id = $scopeId; value = "Studies.ReadWrite"; type = "User"; isEnabled = $true
      adminConsentDisplayName = "Read and write chest X-ray studies"
      adminConsentDescription = "Upload, view and review chest X-ray studies on behalf of the signed-in clinician."
      userConsentDisplayName = "Read and write your chest X-ray studies"
      userConsentDescription = "Upload, view and review your chest X-ray studies."
    })
  }
}
az ad sp show --id $apiAppId 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Invoke-Az ad sp create --id $apiAppId -o none }
Write-Host "  audience $audience"

Step "Web (SPA) app registration"
$spaAppId = Get-OrCreateApp "LungXray Web"
$spaObjectId = Invoke-Az ad app show --id $spaAppId --query id -o tsv
Invoke-Graph PATCH "https://graph.microsoft.com/v1.0/applications/$spaObjectId" @{
  spa = @{ redirectUris = @($webUrl, "http://localhost:5173") }
  requiredResourceAccess = @(@{ resourceAppId = $apiAppId; resourceAccess = @(@{ id = $scopeId; type = "Scope" }) })
}
# Pre-authorise the SPA so clinicians are not shown a consent prompt for our own API
Invoke-Graph PATCH "https://graph.microsoft.com/v1.0/applications/$apiObjectId" @{
  api = @{ preAuthorizedApplications = @(@{ appId = $spaAppId; delegatedPermissionIds = @($scopeId) }) }
}
az ad sp show --id $spaAppId 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Invoke-Az ad sp create --id $spaAppId -o none }
if ($RestrictToAssignedUsers) {
  $spId = Invoke-Az ad sp show --id $spaAppId --query id -o tsv
  Invoke-Graph PATCH "https://graph.microsoft.com/v1.0/servicePrincipals/$spId" @{ appRoleAssignmentRequired = $true }
  Write-Host "  sign-in restricted to users assigned in Enterprise applications > LungXray Web"
}
Write-Host "  redirect URIs: $webUrl, http://localhost:5173"

Step "Switching the deployment to Entra ID"
$vars = [ordered]@{
  AUTH_MODE           = "entra"
  ENTRA_TENANT_ID     = $tenantId
  ENTRA_API_AUDIENCE  = $audience
  ENTRA_SPA_CLIENT_ID = $spaAppId
  ENTRA_API_SCOPE     = "$audience/Studies.ReadWrite"
}
foreach ($k in $vars.Keys) { gh variable set $k --repo $Repo --body $vars[$k]; Write-Host "  $k" }
Write-Host "`nDone. Re-run the Deploy workflow (or push to main) to apply." -ForegroundColor Green
