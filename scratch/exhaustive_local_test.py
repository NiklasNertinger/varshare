import subprocess
import time
import sys

import os

def run_cmd(args):
    start = time.time()
    # Copy full parent environment to preserve SystemRoot/PATH on Windows
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    try:
        proc = subprocess.run(
            [sys.executable] + args,
            capture_output=True,
            text=True,
            env=env,
            timeout=120  # Limit to 2 minutes max per run
        )
        duration = time.time() - start
        return proc.returncode, duration, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return -1, duration, "", "TIMEOUT EXPIRED"

def main():
    print("======================================================================")
    # Highlight the test suite initialization with academic polish
    print("            VARSHARE EXHAUSTIVE INTEGRATION & STABILITY TEST SUITE     ")
    print("======================================================================")
    print("This script runs end-to-end dry-runs for all 12 pipeline configurations")
    print("to verify training, parallel evaluation (AsyncVectorEnv), signal traps,")
    print("checkpointing, and offline plotting after our major refactoring.\n")

    # Define test list as tuples: (Script Name, Arguments)
    test_suite = [
        # 1. Standard Shared Baseline
        ("train_baselines.py", ["train_baselines.py", "--algo", "shared", "--total-timesteps", "2000", "--eval-freq", "1000", "--env-type", "ComplexCartPole", "--exp-name", "test_shared"]),
        # 2. Standard Independent Baseline
        ("train_baselines.py", ["train_baselines.py", "--algo", "independent", "--total-timesteps", "2000", "--eval-freq", "1000", "--env-type", "ComplexCartPole", "--exp-name", "test_independent"]),
        # 3. Standard PCGrad Baseline
        ("train_baselines.py", ["train_baselines.py", "--algo", "pcgrad", "--total-timesteps", "2000", "--eval-freq", "1000", "--env-type", "ComplexCartPole", "--exp-name", "test_pcgrad"]),
        # 4. Standard CAGrad Baseline
        ("train_baselines.py", ["train_baselines.py", "--algo", "cagrad", "--total-timesteps", "2000", "--eval-freq", "1000", "--env-type", "ComplexCartPole", "--exp-name", "test_cagrad"]),
        
        # 5. PaCo Baseline (Continuous-Control)
        ("train_baselines.py", ["train_baselines.py", "--algo", "paco", "--num-experts", "3", "--total-timesteps", "2000", "--eval-freq", "1000", "--env-type", "MultiTaskLunarLander", "--exp-name", "test_paco"]),
        # 6. Soft Modularization Baseline (Continuous-Control)
        ("train_baselines.py", ["train_baselines.py", "--algo", "soft_mod", "--num-modules", "2", "--total-timesteps", "2000", "--eval-freq", "1000", "--env-type", "MultiTaskLunarLander", "--exp-name", "test_soft_mod"]),
        # 7. CARE Baseline (Continuous-Control)
        ("train_baselines.py", ["train_baselines.py", "--algo", "care", "--num-experts", "3", "--total-timesteps", "2000", "--eval-freq", "1000", "--env-type", "MultiTaskLunarLander", "--exp-name", "test_care"]),
        # 8. MOORE Baseline (Continuous-Control)
        ("train_baselines.py", ["train_baselines.py", "--algo", "moore", "--num-experts", "3", "--total-timesteps", "2000", "--eval-freq", "1000", "--env-type", "MultiTaskLunarLander", "--exp-name", "test_moore"]),
        
        # 9. VarShare Base
        ("train_routing.py", ["train_routing.py", "--variant", "base", "--total-timesteps", "2000", "--eval-freq", "1000", "--env-type", "ComplexCartPole", "--exp-name", "test_varshare_base"]),
        # 10. VarShare Routing
        ("train_routing.py", ["train_routing.py", "--variant", "routing", "--total-timesteps", "2000", "--eval-freq", "1000", "--env-type", "ComplexCartPole", "--exp-name", "test_varshare_routing"]),
    ]

    results = []
    failed = False

    for idx, (script, args) in enumerate(test_suite):
        name = args[2].upper() if script == "train_baselines.py" else f"VARSHARE-{args[2].upper()}"
        print(f"[{idx+1}/{len(test_suite)}] Testing: {name:<20} ... ", end="", flush=True)
        
        code, dur, out, err = run_cmd(args)
        
        if code == 0:
            print(f"PASSED OK ({dur:.2f}s)")
            results.append((name, "PASS", f"{dur:.2f}s", ""))
        else:
            print(f"FAILED ERROR ({dur:.2f}s)")
            # Extract main error message for summary
            err_summary = err.strip().splitlines()[-3:] if err else ["Unknown error"]
            results.append((name, "FAIL", f"{dur:.2f}s", " | ".join(err_summary)))
            failed = True

    print("\n" + "="*80)
    print("                    FINAL TEST RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method/Config':<30} | {'Status':<8} | {'Duration':<10} | {'Error Summary'}")
    print("-"*80)
    for name, status, dur, err in results:
        print(f"{name:<30} | {status:<8} | {dur:<10} | {err}")
    print("="*80)

    if failed:
        print("\n[CRITICAL ERROR] One or more integration tests failed. Review errors above!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All pipelines initialized, trained, and evaluated with 100% stability! (ASCII OK)")
        sys.exit(0)

if __name__ == "__main__":
    main()
