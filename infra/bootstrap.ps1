<#
.SYNOPSIS
  One-time setup of the LungXray Azure environment and GitHub deployment (run by the subscription owner).

.DESCRIPTION
  Idempotent: safe to re-run. It
    1. registers the Azure resource providers the app needs
    2. creates the resource group and a container registry (Basic, admin user disabled)
    3. creates a GitHub deploy identity federated via OIDC (no client secret) restricted to this
       repository's "production" environment, with least-privilege roles on the resource group
    4. uploads the production model to a GitHub Release and records its SHA-256
    5. stores the non-secret IDs as GitHub repository variables and creates the environment
  After this, every push to main deploys automatically (.github/workflows/deploy.yml).

.EXAMPLE
  ./infra/bootstrap.ps1 -BudgetEmail you@example.com
#>
[CmdletBinding()]
param(
  [string]$AppName = "lungxray",
  [string]$Location = "centralindia",
  [ValidateSet("westus2", "centralus", "eastus2", "westeurope", "eastasia")]
  [string]$StaticWebAppLocation = "eastasia",
  [string]$ResourceGroup = "rg-lungxray-prod",
  [string]$Repo = "",
  [string]$ModelPath = "models/production/model.pt",
  [string]$ModelTag = "model-v1.0.0",
  [string]$BudgetEmail = "",
  [int]$MonthlyBudget = 20
)
$ErrorActionPreference = "Stop"

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Invoke-Az { $out = & az @args; if ($LASTEXITCODE -ne 0) { throw "az $($args -join ' ') failed" }; return $out }

foreach ($tool in "az", "gh") {
  if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool CLI not found. Install it first (see docs/azure-deployment.md)." }
}
if (-not $Repo) { $Repo = (gh repo view --json nameWithOwner -q .nameWithOwner) }

$account = Invoke-Az account show -o json | ConvertFrom-Json
$subscriptionId = $account.id
$tenantId = $account.tenantId
Write-Host "Subscription: $($account.name) ($subscriptionId)"
Write-Host "Repository:   $Repo"

Step "Registering resource providers"
foreach ($ns in "Microsoft.App", "Microsoft.OperationalInsights", "Microsoft.Insights", "Microsoft.Web",
                "Microsoft.ContainerRegistry", "Microsoft.Storage", "Microsoft.ManagedIdentity", "Microsoft.Consumption") {
  Invoke-Az provider register --namespace $ns --wait | Out-Null
  Write-Host "  $ns"
}

Step "Resource group $ResourceGroup ($Location)"
Invoke-Az group create -n $ResourceGroup -l $Location --tags app=$AppName environment=production managedBy=bootstrap -o none

Step "Container registry"
$rgId = Invoke-Az group show -n $ResourceGroup --query id -o tsv
$hash = [System.BitConverter]::ToString(
  [System.Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($rgId))
).Replace("-", "").Substring(0, 8).ToLower()
$acrName = "acr$AppName$hash"
$existing = az acr show -n $acrName -g $ResourceGroup --query name -o tsv 2>$null
if (-not $existing) {
  Invoke-Az acr create -n $acrName -g $ResourceGroup --sku Basic --admin-enabled false -o none
}
$acrId = Invoke-Az acr show -n $acrName -g $ResourceGroup --query id -o tsv
Write-Host "  $acrName"

Step "GitHub deploy identity (OIDC federation, no secrets)"
$idName = "id-$AppName-github-deploy"
Invoke-Az identity create -n $idName -g $ResourceGroup -o none
$deployId = Invoke-Az identity show -n $idName -g $ResourceGroup -o json | ConvertFrom-Json
$subject = "repo:${Repo}:environment:production"
$fc = az identity federated-credential show --identity-name $idName -g $ResourceGroup -n github-production 2>$null
if (-not $fc) {
  Invoke-Az identity federated-credential create --identity-name $idName -g $ResourceGroup -n github-production `
    --issuer "https://token.actions.githubusercontent.com" --subject $subject `
    --audiences "api://AzureADTokenExchange" -o none
}
Write-Host "  federated subject: $subject"

Step "Least-privilege role assignments for the deploy identity"
function Grant($role, $scope, $condition) {
  $has = az role assignment list --assignee $deployId.principalId --role $role --scope $scope --query "[0].id" -o tsv 2>$null
  if ($has) { Write-Host "  $role (exists)"; return }
  $cmd = @("role", "assignment", "create", "--assignee-object-id", $deployId.principalId,
            "--assignee-principal-type", "ServicePrincipal", "--role", $role, "--scope", $scope, "-o", "none")
  if ($condition) { $cmd += @("--condition", $condition, "--condition-version", "2.0") }
  # New identities can take a few seconds to replicate in Entra ID
  for ($i = 0; $i -lt 6; $i++) {
    & az @cmd 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "  $role"; return }
    Start-Sleep -Seconds 10
  }
  throw "Could not assign $role"
}
Grant "Contributor" $rgId $null
Grant "AcrPush" $acrId $null
# May only create/delete assignments of the three data-plane roles the app needs (ABAC condition).
$allowed = "7f951dda-4ed3-4680-a7ca-43fe172d538d, ba92f5b4-2d11-453d-a403-e96b0029c9fe, 0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3"
$condition = "((!(ActionMatches{'Microsoft.Authorization/roleAssignments/write'})) OR " +
  "(@Request[Microsoft.Authorization/roleAssignments:RoleDefinitionId] ForAnyOfAnyValues:GuidEquals {$allowed})) AND " +
  "((!(ActionMatches{'Microsoft.Authorization/roleAssignments/delete'})) OR " +
  "(@Resource[Microsoft.Authorization/roleAssignments:RoleDefinitionId] ForAnyOfAnyValues:GuidEquals {$allowed}))"
Grant "Role Based Access Control Administrator" $rgId $condition

Step "Production model (GitHub Release $ModelTag)"
$asset = "lungxray-model.pt"
$modelUrl = "https://github.com/$Repo/releases/download/$ModelTag/$asset"
$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "lungxray-model"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$tmp = Join-Path $tmpDir $asset
gh release view $ModelTag --repo $Repo 2>$null | Out-Null
$releaseExists = ($LASTEXITCODE -eq 0)
if (Test-Path $ModelPath) {
  # Local model present (e.g. your PC): publish it
  $sha = (Get-FileHash $ModelPath -Algorithm SHA256).Hash.ToLower()
  Copy-Item $ModelPath $tmp -Force
  if (-not $releaseExists) {
    gh release create $ModelTag $tmp --repo $Repo --title "Production model $ModelTag" `
      --notes "ResNet-18 bundle (weights + preprocessing + calibration). SHA-256: $sha"
  } else {
    gh release upload $ModelTag $tmp --repo $Repo --clobber
  }
} elseif ($releaseExists) {
  # No local model (e.g. Azure Cloud Shell): use the already-published release and checksum it
  gh release download $ModelTag --repo $Repo --pattern $asset --dir $tmpDir --clobber
  $sha = (Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower()
} else {
  throw "No model at $ModelPath and no release $ModelTag on $Repo. Publish the model first."
}
Remove-Item -Recurse -Force $tmpDir
Write-Host "  $modelUrl"
Write-Host "  sha256 $sha"

Step "GitHub environment and repository variables"
gh api -X PUT "repos/$Repo/environments/production" --silent
$vars = [ordered]@{
  AZURE_CLIENT_ID        = $deployId.clientId
  AZURE_TENANT_ID        = $tenantId
  AZURE_SUBSCRIPTION_ID  = $subscriptionId
  AZURE_RESOURCE_GROUP   = $ResourceGroup
  AZURE_LOCATION         = $Location
  SWA_LOCATION           = $StaticWebAppLocation
  ACR_NAME               = $acrName
  MODEL_URL              = $modelUrl
  MODEL_SHA256           = $sha
  MONTHLY_BUDGET         = "$MonthlyBudget"
}
if ($BudgetEmail) { $vars["BUDGET_EMAIL"] = $BudgetEmail }
foreach ($k in $vars.Keys) {
  gh variable set $k --repo $Repo --body $vars[$k]
  Write-Host "  $k"
}

Step "Starting the first deployment"
# From now on every push to main deploys automatically; this run just deploys the current code once.
gh workflow run deploy.yml --repo $Repo --ref main
Write-Host "  Follow it with: gh run watch --repo $Repo"
Write-Host "`nDone. Every push to main now tests, builds and deploys automatically." -ForegroundColor Green
Write-Host "Optional: ./infra/setup-entra.ps1 to enable Microsoft Entra ID sign-in." -ForegroundColor Green
