
# Usage: .\scripts\verify_metrics_50k.ps1
# Runs a 50k step verification of VarShare with advanced metrics enabled.

$ExpName = "verify_metrics_50k"
$Steps = 50000

Write-Host ">>> Starting 50k Step Metrics Verification ($ExpName)..."
Write-Host "This will take a few minutes."

python scripts/train_varshare_ppo.py `
    --env-type metaworld `
    --mt-setting MT10 `
    --num-envs 8 `
    --n-steps 128 `
    --total-timesteps $Steps `
    --eval-freq 10000 `
    --exp-name $ExpName `
    --wandb-project "varshare-debug"

Write-Host ">>> Run Complete."
Write-Host "Check results in: analysis/$ExpName/seed_1/heartbeat.csv"
