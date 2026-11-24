// LungXray production environment (resource-group scope).
//
// Secretless by design: the API reaches Storage through a user-assigned managed identity,
// GitHub Actions deploys via OIDC federation, and the API image is pulled anonymously from
// GitHub Container Registry (public package, free). No keys or client secrets exist anywhere.
//
//   Static Web App (React SPA)  ──HTTPS──►  Container App (FastAPI + model)
//                                              │ managed identity
//                        ┌─────────────────────┴──────────────────────┐
//                   Blob (images)                          Table (studies, audit)
//                                              │
//                          Log Analytics + Application Insights (logs, traces)

targetScope = 'resourceGroup'

@description('Short name used as a prefix for every resource.')
@minLength(3)
@maxLength(12)
param appName string = 'lungxray'

param location string = resourceGroup().location

@description('Static Web Apps is only offered in a few regions.')
@allowed(['westus2', 'centralus', 'eastus2', 'westeurope', 'eastasia'])
param staticWebAppLocation string = 'eastus2'

@description('Full image reference, e.g. ghcr.io/<owner>/lungxray-api:<git-sha> (public package).')
param apiImage string

@allowed(['demo', 'entra'])
param authMode string = 'demo'
param entraTenantId string = ''
@description('Audience of access tokens for the API, e.g. api://<api-app-client-id>.')
param entraApiAudience string = ''

@description('Upper bound on API replicas. 1 caps worst-case cost and keeps the per-replica rate limit exact; raise it for real production load.')
@minValue(1)
@maxValue(10)
param maxReplicas int = 1

@description('Days before study images are deleted automatically (data minimisation).')
@minValue(1)
param imageRetentionDays int = 30

var suffix = uniqueString(resourceGroup().id)
var tags = { app: appName, environment: 'production', managedBy: 'bicep' }

// Built-in role definition IDs
var roles = {
  blobDataContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  tableDataContributor: '0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3'
}

// ---------- Observability ----------
resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${appName}-${suffix}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    workspaceCapping: { dailyQuotaGb: 1 } // hard cap on ingestion cost
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${appName}-${suffix}'
  location: location
  kind: 'web'
  tags: tags
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
    DisableLocalAuth: false
  }
}

// ---------- Identity ----------
resource apiIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${appName}-api'
  location: location
  tags: tags
}

// ---------- Storage (studies, images, audit) ----------
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: take('st${appName}${suffix}', 24)
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false // Entra ID (managed identity) only, no account keys
    defaultToOAuthAuthentication: true
    publicNetworkAccess: 'Enabled'
    encryption: {
      keySource: 'Microsoft.Storage'
      services: { blob: { enabled: true }, table: { enabled: true } }
      requireInfrastructureEncryption: true // double encryption at rest
    }
  }

  resource blobService 'blobServices' = {
    name: 'default'
    properties: {
      deleteRetentionPolicy: { enabled: true, days: 7 } // soft delete
    }
    resource studies 'containers' = {
      name: 'studies'
      properties: { publicAccess: 'None' }
    }
  }

  resource tableService 'tableServices' = {
    name: 'default'
    resource studiesTable 'tables' = { name: 'studies' }
    resource auditTable 'tables' = { name: 'audit' }
  }

  resource lifecycle 'managementPolicies' = {
    name: 'default'
    properties: {
      policy: {
        rules: [
          {
            name: 'delete-old-study-images'
            enabled: true
            type: 'Lifecycle'
            definition: {
              filters: { blobTypes: ['blockBlob'], prefixMatch: ['studies/'] }
              actions: { baseBlob: { delete: { daysAfterModificationGreaterThan: imageRetentionDays } } }
            }
          }
        ]
      }
    }
  }
}

resource blobRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, apiIdentity.id, roles.blobDataContributor)
  scope: storage
  properties: {
    principalId: apiIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.blobDataContributor)
  }
}

resource tableRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, apiIdentity.id, roles.tableDataContributor)
  scope: storage
  properties: {
    principalId: apiIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.tableDataContributor)
  }
}

// ---------- Front end ----------
resource web 'Microsoft.Web/staticSites@2023-12-01' = {
  name: 'stapp-${appName}-${suffix}'
  location: staticWebAppLocation
  tags: tags
  sku: { name: 'Free', tier: 'Free' }
  properties: {
    stagingEnvironmentPolicy: 'Disabled'
    allowConfigFileUpdates: true
  }
}

// ---------- API ----------
resource containerEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${appName}-${suffix}'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
    workloadProfiles: [{ name: 'Consumption', workloadProfileType: 'Consumption' }]
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-${appName}-api'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${apiIdentity.id}': {} }
  }
  properties: {
    managedEnvironmentId: containerEnv.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'AUTH_MODE', value: authMode }
            { name: 'ENTRA_TENANT_ID', value: entraTenantId }
            { name: 'ENTRA_API_AUDIENCE', value: entraApiAudience }
            { name: 'STORAGE_MODE', value: 'azure' }
            { name: 'AZURE_STORAGE_ACCOUNT', value: storage.name }
            { name: 'AZURE_CLIENT_ID', value: apiIdentity.properties.clientId } // tells DefaultAzureCredential which identity
            { name: 'CORS_ORIGINS', value: 'https://${web.properties.defaultHostname}' }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
            { name: 'PREDICTION_LOG', value: '/tmp/predictions.jsonl' }
          ]
          probes: [
            {
              type: 'Startup'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 5
              periodSeconds: 5
              failureThreshold: 24 // up to 2 minutes for a cold start (model load)
            }
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8000 }
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0 // scale to zero when idle: pay only while in use
        maxReplicas: maxReplicas
        rules: [{ name: 'http', http: { metadata: { concurrentRequests: '10' } } }]
      }
    }
  }
  dependsOn: [blobRole, tableRole]
}

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output webUrl string = 'https://${web.properties.defaultHostname}'
output staticWebAppName string = web.name
output storageAccount string = storage.name
output appInsightsName string = appInsights.name
