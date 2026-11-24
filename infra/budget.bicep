// Monthly cost alert for the LungXray resource group.
// Deployed once by infra/bootstrap.ps1 (not by CI), so the alert e-mail address never
// reaches GitHub variables or the public Actions logs.

targetScope = 'resourceGroup'

@description('Monthly amount in the billing currency (USD for Azure for Students).')
@minValue(1)
param amount int = 2

@description('Where alerts are sent.')
param alertEmail string

param startDate string = utcNow('yyyy-MM-01')

resource budget 'Microsoft.Consumption/budgets@2023-11-01' = {
  name: 'budget-lungxray'
  properties: {
    category: 'Cost'
    amount: amount
    timeGrain: 'Monthly'
    timePeriod: { startDate: startDate }
    notifications: {
      actual80: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 80
        thresholdType: 'Actual'
        contactEmails: [alertEmail]
      }
      forecast100: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 100
        thresholdType: 'Forecasted'
        contactEmails: [alertEmail]
      }
    }
  }
}
