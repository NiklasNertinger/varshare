# sync_data_down.ps1
# This script uses native Windows Secure Copy (scp) to pull heavy evaluation logs and plots back to your local PC.

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Agent OS: VarShare Data Sync (Download)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

$remoteUser = "nertinger"
$remoteHost = "pegasus.dfki.de"
$remotePath = "/netscratch/$remoteUser/varshare/analysis/"
$localPath = ".\analysis"

# Ensure local analysis directory exists
if (-not (Test-Path -Path $localPath)) {
    New-Item -ItemType Directory -Path $localPath | Out-Null
}

Write-Host "`n[1/1] Downloading massive data blocks via SCP..." -ForegroundColor Yellow
Write-Host "      (If you do not have an SSH key set up, it will ask for your password now)" -ForegroundColor Gray
Write-Host "      Remote Path: $remoteHost`:$remotePath"
Write-Host "      Local Path:  $localPath`n"

# Run scp recursively. 
# Note: scp -r downloads everything. If this becomes too slow in the future, we can filter it.
scp -r "$remoteUser`@$remoteHost`:$remotePath*" "$localPath\"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: SCP transfer failed or was interrupted." -ForegroundColor Red
    exit 1
}

Write-Host "`n=============================================" -ForegroundColor Green
Write-Host "   SUCCESS: All cluster data synced locally! " -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
