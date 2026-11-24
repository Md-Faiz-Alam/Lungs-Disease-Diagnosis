<#
.SYNOPSIS
  One-time setup of the LungXray Azure environment and GitHub deployment (run by the subscription owner).

.DESCRIPTION
  Idempotent: safe to re-run. It
    1. registers the Azure resource providers the app needs
    2. creates the resource group (the API image lives in GitHub Container Registry, free)
    3. creates a GitHub deploy identity federated via OIDC (no client secret) restricted to this
       repository's "production" environment, with least-privilege roles on the resource group
    4. uploads the production model to a GitHub Release and records its SHA-256
    5. stores the non-secret IDs as GitHub repository variables and creates the environment
    6. creates a monthly cost alert (your e-mail stays in Azure; it is never sent to GitHub)
  After this, every push to main deploys automatically (.github/workflows/deploy.yml).

.EXAMPLE
  ./infra/bootstrap.ps1 -BudgetEmail you@example.com
#>
[CmdletBinding()]
param(
  [string]$AppName = "lungxray",
  [string]$Location = "uaenorth",
  [ValidateSet("westus2", "centralus", "eastus2", "westeurope", "eastasia")]
  [string]$StaticWebAppLocation = "eastasia",
  [string]$ResourceGroup = "rg-lungxray-prod",
  [string]$Repo = "",
  [string]$ModelPath = "models/production/model.pt",
  [string]$ModelTag = "model-v1.0.0",
  [string]$BudgetEmail = "",
  [int]$MonthlyBudget = 2  # USD, this resource group only
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
                "Microsoft.Storage", "Microsoft.ManagedIdentity", "Microsoft.Consumption") {
  Invoke-Az provider register --namespace $ns --wait | Out-Null
  Write-Host "  $ns"
}

Step "Resource group $ResourceGroup ($Location)"
Invoke-Az group create -n $ResourceGroup -l $Location --tags app=$AppName environment=production managedBy=bootstrap -o none
$rgId = Invoke-Az group show -n $ResourceGroup --query id -o tsv

if ($BudgetEmail) {
  Step "Monthly cost alert ($MonthlyBudget, billing currency)"
  Invoke-Az deployment group create -g $ResourceGroup -n budget --template-file "$PSScriptRoot/budget.bicep" `
    --parameters amount=$MonthlyBudget alertEmail=$BudgetEmail -o none
  Write-Host "  alerts at 80% actual and 100% forecast spend"
} else {
  Write-Host "  (no -BudgetEmail given: skipping the cost alert)" -ForegroundColor Yellow
}

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
# May only create/delete assignments of the two storage data roles the app needs (ABAC condition).
$allowed = "ba92f5b4-2d11-453d-a403-e96b0029c9fe, 0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3"
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
# Only the main branch may deploy: Azure trusts this environment (OIDC subject), so no other
# branch, fork or pull request can ever obtain Azure access.
'{"deployment_branch_policy":{"protected_branches":false,"custom_branch_policies":true}}' |
  gh api -X PUT "repos/$Repo/environments/production" --input - --silent
$policies = gh api "repos/$Repo/environments/production/deployment-branch-policies" --jq '.branch_policies[].name'
if (-not ($policies -contains "main")) {
  gh api -X POST "repos/$Repo/environments/production/deployment-branch-policies" -f name=main -f type=branch --silent
}
Write-Host "  environment 'production': deployments allowed from main only"
$vars = [ordered]@{
  AZURE_CLIENT_ID        = $deployId.clientId
  AZURE_TENANT_ID        = $tenantId
  AZURE_SUBSCRIPTION_ID  = $subscriptionId
  AZURE_RESOURCE_GROUP   = $ResourceGroup
  AZURE_LOCATION         = $Location
  SWA_LOCATION           = $StaticWebAppLocation
  MODEL_URL              = $modelUrl
  MODEL_SHA256           = $sha
}
foreach ($k in $vars.Keys) {
  gh variable set $k --repo $Repo --body $vars[$k]
  Write-Host "  $k"
}

Write-Host "`nIMPORTANT (one time): after the first deployment pushes the image, make the package public:" -ForegroundColor Yellow
Write-Host "  https://github.com/$Repo/pkgs/container/lungxray-api  ->  Package settings  ->  Change visibility  ->  Public" -ForegroundColor Yellow
Write-Host "  (the workflow stops with this same message until you do; then re-run it)" -ForegroundColor Yellow

Step "Starting the first deployment"
# From now on every push to main deploys automatically; this run just deploys the current code once.
gh workflow run deploy.yml --repo $Repo --ref main
Write-Host "  Follow it with: gh run watch --repo $Repo"
Write-Host "`nDone. Every push to main now tests, builds and deploys automatically." -ForegroundColor Green
Write-Host "Optional: ./infra/setup-entra.ps1 to enable Microsoft Entra ID sign-in." -ForegroundColor Green
