# test_mega_hpo_local.ps1
# Runs a short smoke test for all 11 HPO studies locally.

# Environment Overrides for Short Run
$env:HPO_TIME_STEPS = "30000"
$env:HPO_N_STEPS = "512"

# Ensure venv
if (Test-Path venv\Scripts\Activate.ps1) {
    . venv\Scripts\Activate.ps1
}

$studies = @(
    "mt10_varshare_base",
    "mt10_varshare_emb_onehot",
    "mt10_varshare_emb_learned",
    "mt10_varshare_lora",
    "mt10_varshare_partial",
    "mt10_varshare_reptile",
    "mt10_varshare_scaled_down",
    "mt10_varshare_fixed_prior",
    "mt10_varshare_annealing",
    "mt10_varshare_trigger",
    "mt10_varshare_emp_bayes"
)

# Create Logs Dir
New-Item -ItemType Directory -Force -Path logs/test_hpo | Out-Null

# Clean up previous local test DBs to force fresh run
Remove-Item -Path "analysis/test_hpo_*.db" -ErrorAction SilentlyContinue
Remove-Item -Path "analysis/test_hpo_*.log" -ErrorAction SilentlyContinue

foreach ($study in $studies) {
    Write-Host ">>> Testing Study: $study" -ForegroundColor Cyan
    
    # Run Python Script
    # use sqlite:/// for Windows-safe local testing (avoids Journal symlink error)
    python scripts/optimize_${study}.py --n-trials 2 --storage-path "sqlite:///analysis/test_hpo_${study}.db"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!!! Error running $study !!!" -ForegroundColor Red
        # Optional: exit or continue? Let's continue to check others.
    }
}

Write-Host "All tests completed." -ForegroundColor Green
