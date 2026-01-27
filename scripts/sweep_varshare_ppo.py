import subprocess
import sys
import re
import os

# Grid
lr_actors = [0.001, 0.005]
lr_critics = [0.0001, 0.0005]
n_steps_list = [256, 1024]

results = []

print(f"{'LR_Actor':<10} | {'LR_Critic':<10} | {'N_Steps':<8} | {'Final_Reward':<12}")
print("-" * 50)

for n_steps in n_steps_list:
    for lr_a in lr_actors:
        for lr_c in lr_critics:
            exp_name = f"sweep_N{n_steps}_lrA{lr_a}_lrC{lr_c}"
            
            # Using 10000 for eval freq so we get roughly ~4 evals in 41000 steps
            # Since total timesteps is 41000, we expect evals at roughly 10k, 20k, 30k, 40k
            cmd = [
                sys.executable, "scripts/train_varshare_golden.py",
                "--exp-name", exp_name,
                "--total-timesteps", "41000",
                "--seed", "1",
                "--lr-actor", str(lr_a),
                "--lr-critic", str(lr_c),
                "--n-steps", str(n_steps),
                "--eval-freq", "10000", 
                "--eval-mode", "True",
                "--wandb-project", "varshare-sweep" # Separate project just in case
            ]
            
            print(f"Running: N={n_steps}, lrA={lr_a}, lrC={lr_c} ...", end="", flush=True)
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                
                # Check for errors
                if result.returncode != 0:
                    print(f" FAILED (Return Code {result.returncode})")
                    # Print last 500 chars of stderr to see what went wrong
                    print("STDERR TAIL:")
                    print(result.stderr[-500:]) 
                    continue

                # Parse output
                # Output format: "Eval Reward: 123.45"
                matches = re.findall(r"Eval Reward: ([\d.-]+)", result.stdout)
                if matches:
                    final_reward = float(matches[-1])
                    print(f" Done. Reward: {final_reward}")
                    results.append({
                        "lr_actor": lr_a, 
                        "lr_critic": lr_c, 
                        "n_steps": n_steps, 
                        "reward": final_reward
                    })
                else:
                    print(f" Done. No eval found. Output length: {len(result.stdout)}")
                    # Maybe print a bit of stdout to debug if needed
                    # print(result.stdout[-200:])
                    results.append({
                        "lr_actor": lr_a, 
                        "lr_critic": lr_c, 
                        "n_steps": n_steps, 
                        "reward": -9999.0
                    })
                    
            except Exception as e:
                print(f" EXCEPTION: {e}")

print("\n\n=== RESULTS SORTED BY REWARD ===")
results.sort(key=lambda x: x["reward"], reverse=True)
for r in results:
    print(f"Reward: {r['reward']:.2f} | N_Steps: {r['n_steps']} | LR_A: {r['lr_actor']} | LR_C: {r['lr_critic']}")
