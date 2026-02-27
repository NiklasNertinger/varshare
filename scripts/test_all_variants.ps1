$ErrorActionPreference = "Continue"

$venv_python = ".\.venv\Scripts\python.exe"

$envs = @(
    @{"env_type"="metaworld"; "mt_setting"="MT4"; "num_tasks"=4},
    @{"env_type"="MultiTaskLunarLander"; "mt_setting"="None"; "num_tasks"=2}
)

$det_variants = @("base", "lora", "gated", "l1", "pcgrad", "decay", "film", "hyperprior", "ara")

$baselines = @(
    @{"name"="shared_embedding"; "script"="scripts/train_baseline_ppo.py"; "args"="--algo shared"},
    @{"name"="shared_embedding_pcgrad"; "script"="scripts/train_baseline_ppo.py"; "args"="--algo pcgrad"},
    @{"name"="paco"; "script"="scripts/train_baseline_ppo.py"; "args"="--algo paco"},
    @{"name"="soft_mod"; "script"="scripts/train_baseline_ppo.py"; "args"="--algo soft_mod"},
    @{"name"="varshare_prior_opt"; "script"="scripts/train_varshare_ppo.py"; "args"="--variant standard --kl-beta 0.01 --prior-scale 0.5"}
)

$total_timesteps = 512
$n_steps = 256
$eval_freq = 500

Write-Host "=================================="
Write-Host "Starting Mega Smoke Test"
Write-Host "=================================="

foreach ($env_dict in $envs) {
    $env_type = $env_dict["env_type"]
    $mt_setting = $env_dict["mt_setting"]
    $num_envs = 1 # Keep serial for testing, evaluates will run across all tasks though

    Write-Host "`n>>> Testing Environment: $env_type (MT: $mt_setting)"

    # Test Deterministic Variants
    foreach ($var in $det_variants) {
        Write-Host "  -> Running DetVarShare Variant: $var"
        $cmd = "$venv_python deterministic/scripts/train_det_ppo.py --env-type $env_type --mt-setting $mt_setting --variant $var --total-timesteps $total_timesteps --n-steps $n_steps --num-envs $num_envs --eval-freq $eval_freq --exp-name test_${env_type}_det_${var}"
        
        $output = Invoke-Expression $cmd 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "     [ERROR] Failed!" -ForegroundColor Red
            Write-Host "     $output"
        } else {
            # Extract final average reward to ensure it logged properly
            $reward_line = $output | Where-Object { $_ -match "Final Average Reward:" } | Select-Object -Last 1
            Write-Host "     [SUCCESS] $reward_line" -ForegroundColor Green
        }
    }

    # Test Baselines
    foreach ($base in $baselines) {
        $b_name = $base["name"]
        $b_script = $base["script"]
        $b_args = $base["args"]
        
        Write-Host "  -> Running Baseline: $b_name"
        $cmd = "$venv_python $b_script --env-type $env_type --mt-setting $mt_setting $b_args --total-timesteps $total_timesteps --n-steps $n_steps --num-envs $num_envs --eval-freq $eval_freq --exp-name test_${env_type}_base_${b_name}"
        
        $output = Invoke-Expression $cmd 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "     [ERROR] Failed!" -ForegroundColor Red
            Write-Host "     $output"
        } else {
            $reward_line = $output | Where-Object { $_ -match "Final Average Reward:" } | Select-Object -Last 1
            Write-Host "     [SUCCESS] $reward_line" -ForegroundColor Green
        }
    }
}

Write-Host "`n=================================="
Write-Host "Smoke Test Complete."
Write-Host "=================================="
