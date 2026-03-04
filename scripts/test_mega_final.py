import os
import subprocess
import sys

# Define all the variants to test
PHASE_1 = [
    ("soft_mod", "scripts/train_baseline_ppo.py", ["--algo", "soft_mod", "--num-modules", "4"]),
    ("shared_embedding", "scripts/train_baseline_ppo.py", ["--algo", "shared"]),
    ("paco", "scripts/train_baseline_ppo.py", ["--algo", "paco", "--num-experts", "4"]),
    ("shared_embedding_pcgrad", "scripts/train_baseline_ppo.py", ["--algo", "pcgrad"]),
    ("varshare_standard_consistent", "scripts/train_varshare_ppo.py", ["--variant", "standard", "--consistent-noise", "true"]),
    ("det_base", "deterministic/scripts/train_det_ppo.py", ["--variant", "base"])
]

PHASE_2 = [
    ("varshare_standard", "scripts/train_varshare_ppo.py", ["--variant", "standard", "--consistent-noise", "false"]),
    ("varshare_reptile_consistent", "scripts/train_varshare_ppo.py", ["--variant", "reptile", "--consistent-noise", "true"]),
    ("varshare_partial_consistent", "scripts/train_varshare_ppo.py", ["--variant", "partial", "--consistent-noise", "true"]),
    ("varshare_lora_consistent", "scripts/train_varshare_ppo.py", ["--variant", "lora", "--consistent-noise", "true"]),
    # Legacy Prob Base (with specific hyperparams applied via script)
    ("legacy_prob_base_consistent", "scripts/train_varshare_ppo.py", ["--variant", "standard", "--consistent-noise", "true", "--kl-beta", "0.00043", "--lr-actor", "0.000117", "--lr-critic", "0.000259", "--hidden-dim", "64"]),
    ("legacy_prob_base_inconsistent", "scripts/train_varshare_ppo.py", ["--variant", "standard", "--consistent-noise", "false", "--kl-beta", "0.00043", "--lr-actor", "0.000117", "--lr-critic", "0.000259", "--hidden-dim", "64"]),
    ("det_lora", "deterministic/scripts/train_det_ppo.py", ["--variant", "lora"]),
    ("det_ara", "deterministic/scripts/train_det_ppo.py", ["--variant", "ara"]),
    ("det_l1", "deterministic/scripts/train_det_ppo.py", ["--variant", "l1"]),
    ("det_gated", "deterministic/scripts/train_det_ppo.py", ["--variant", "gated"]),
    ("det_pcgrad", "deterministic/scripts/train_det_ppo.py", ["--variant", "pcgrad"])
]

ALL_VARIANTS = PHASE_1 + PHASE_2

def run_test():
    total_timesteps = 1000
    eval_freq = 500
    env_type = "IdenticalCartPole" # fast CP environment for basic crash check
    
    print("==================================================")
    print(f"Starting Local Mega-Test (Zero-Crash Guarantee)")
    print(f"Total Variants: {len(ALL_VARIANTS)}")
    print("==================================================")
    
    successes = 0
    failures = []
    
    for name, script, extra_args in ALL_VARIANTS:
        print(f"\n>>> Testing: {name}")
        cmd = [
            sys.executable, script,
            "--exp-name", f"TEST_{name}",
            "--seed", "1",
            "--total-timesteps", str(total_timesteps),
            "--n-steps", "128",
            "--batch-size", "64",
            "--num-envs", "4",
            "--eval-mode", "True",
            "--eval-freq", str(eval_freq),
            "--env-type", env_type,
            "--analysis-dir", "analysis/TEST_MEGA"
        ] + extra_args
        
        try:
            # We enforce a timeout just in case it hangs
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            if result.returncode == 0:
                print(f"[OK] {name}")
                successes += 1
            else:
                print(f"[FAILED] {name}\nERROR:")
                print(result.stderr[-1000:])
                failures.append(name)
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {name}")
            failures.append(name)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failures.append(name)
            
    print("\n==================================================")
    print(f"TEST SUMMARY: {successes}/{len(ALL_VARIANTS)} Succeeded")
    if failures:
        print("FAILURES DETECTED IN:", failures)
        sys.exit(1)
    else:
        print("ALL TESTS PASSED. ZERO-CRASH GUARANTEE SECURED.")
        sys.exit(0)

if __name__ == "__main__":
    run_test()
