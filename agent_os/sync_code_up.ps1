# sync_code_up.ps1
# This script synchronizes your local code with the DFKI cluster by pushing to Git and telling the cluster to pull.

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   Agent OS: VarShare Cluster Code Sync" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Step 1: Push local code to remote repository
Write-Host "`n[1/2] Pushing local commits to GitHub..." -ForegroundColor Yellow
git push origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to push to GitHub. Please check your git status." -ForegroundColor Red
    exit 1
}

# Step 2: SSH into cluster and pull
Write-Host "`n[2/2] Triggering 'git pull' on pegasus.dfki.de..." -ForegroundColor Yellow
Write-Host "      (If you do not have an SSH key set up, it will ask for your password now)" -ForegroundColor Gray

ssh pegasus.dfki.de "cd ~/varshare && git pull origin main"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to pull code on the cluster." -ForegroundColor Red
    exit 1
}

Write-Host "`n=============================================" -ForegroundColor Green
Write-Host "   SUCCESS: Cluster is now perfectly synced! " -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
