import subprocess
import sys
import os

def run_benchmark(script, method_arg_name, method_name, device, exp_name):
    print(f"\n{'='*60}")
    print(f"Running Benchmark: {exp_name} on {device.upper()}")
    print(f"{'='*60}")
    
    cuda_val = "True" if device == "gpu" else "False"
    
    cmd = [
        sys.executable, script,
        method_arg_name, method_name,
        "--total-timesteps", "100000",
        "--env-type", "metaworld",
        "--mt-setting", "MT10",
        "--wandb-project", "varshare-hardware-bench",
        "--exp-name", exp_name,
        "--cuda", cuda_val,
        "--eval-mode", "False" # Disable eval to isolate pure training compute speed
    ]
    
    # We use a single seed to keep things fast and comparable
    cmd.extend(["--seed", "42"])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command and wait for completion
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"[FAIL] {exp_name} failed on {device.upper()}.")
    else:
        print(f"[OK] {exp_name} finished on {device.upper()}.")

def main():
    # Define the methods to benchmark
    # Format: (script, argument_flag, method_name, readable_name)
    methods = [
        # Main VarShare Methods
        ("train_routing.py", "--variant", "base", "varshare_base"),
        ("train_routing.py", "--variant", "lora", "varshare_lora"),
        ("train_routing.py", "--variant", "gated", "varshare_gated"),
        ("train_routing.py", "--variant", "film", "varshare_film"),
        ("train_routing.py", "--variant", "hyperprior", "varshare_hyperprior"),
        ("train_routing.py", "--variant", "pcgrad", "varshare_pcgrad"),
        ("train_routing.py", "--variant", "cagrad", "varshare_cagrad"),
        
        # Baselines
        ("train_baselines.py", "--algo", "shared", "baseline_shared"),
        ("train_baselines.py", "--algo", "independent", "baseline_independent"),
        ("train_baselines.py", "--algo", "paco", "baseline_paco"),
        ("train_baselines.py", "--algo", "soft_mod", "baseline_softmod"),
        ("train_baselines.py", "--algo", "care", "baseline_care"),
        ("train_baselines.py", "--algo", "moore", "baseline_moore")
    ]
    
    # Sequential execution to ensure no resource contention ruins the timing
    for script, arg_name, method_name, readable_name in methods:
        for device in ["cpu", "gpu"]:
            exp_name = f"bench_{readable_name}_{device}"
            run_benchmark(script, arg_name, method_name, device, exp_name)

if __name__ == "__main__":
    main()
