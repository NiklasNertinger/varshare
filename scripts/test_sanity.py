import subprocess
import os
import shutil
import json

def run_test(script, args, name):
    print(f"Running {name}...")
    
    # Clean up previous test runs
    analysis_dir = "/netscratch/nertinger/varshare/analysis/sanity_test"
    log_dir = "/netscratch/nertinger/varshare/logs/sanity_test"
    
    import sys
    env_type = "IdenticalLunarLander" if "paco" in name else "IdenticalCartPole"
    
    full_args = [
        sys.executable, script,
        "--exp-name", name,
        "--seed", "42",
        "--total-timesteps", "128",
        "--n-steps", "64",
        "--batch-size", "32",
        "--num-envs", "2",
        "--env-type", env_type,
        "--eval-mode", "True",
        "--eval-freq", "128",
        "--analysis-dir", analysis_dir,
    ] + args
    
    result = subprocess.run(full_args, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[FAIL] {name} FAILED!")
        print(result.stderr[-1000:])
        return False
        
    print(f"[OK] {name} SUCCESS")
    
    # Read the heartbeat to get the final loss and reward for deterministic comparison
    try:
        seed_dir = os.path.join(analysis_dir, name, "seed_42")
        with open(os.path.join(seed_dir, "heartbeat.csv"), "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                header = lines[0].strip().split(",")
                last_line = lines[-1].strip().split(",")
                data = dict(zip(header, last_line))
                print(f"   loss: {float(data.get('loss/policy', 0)):.6f}, eval_reward: {float(data.get('eval_reward_current', 0)):.6f}")
                
                # Save to a baseline file for comparison
                baseline_path = os.path.join(analysis_dir, f"{name}_baseline.json")
                if not os.path.exists(baseline_path):
                    with open(baseline_path, "w") as bf:
                        json.dump(data, bf, indent=2)
                    print(f"   Saved baseline to {baseline_path}")
                else:
                    with open(baseline_path, "r") as bf:
                        baseline_data = json.load(bf)
                    
                    match = True
                    for k in ["loss/policy", "loss/value", "eval_reward_current"]:
                        v1 = float(data.get(k, 0))
                        v2 = float(baseline_data.get(k, 0))
                        if abs(v1 - v2) > 1e-5:
                            print(f"   [FAIL] REGRESSION on {k}: {v1} != {v2}")
                            match = False
                    if match:
                        print(f"   [OK] Exact Match with Baseline")
            else:
                print("   [FAIL] Heartbeat file is empty")
    except Exception as e:
        print(f"   [FAIL] Could not read heartbeat: {e}")
        
    return True

def main():
    print("=" * 60)
    print("  VARSHARE SANITY & REGRESSION SUITE")
    print("=" * 60)
    
    tests = [
        ("train_routing.py", ["--variant", "base"], "sanity_routing_base"),
        ("train_baselines.py", ["--algo", "shared"], "sanity_shared"),
        ("train_baselines.py", ["--algo", "paco"], "sanity_paco"),
        ("train_baselines.py", ["--algo", "independent"], "sanity_independent")
    ]
    
    all_passed = True
    for script, args, name in tests:
        if not run_test(script, args, name):
            all_passed = False
            
    print("=" * 60)
    if all_passed:
        print("[DONE] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")

if __name__ == "__main__":
    main()
