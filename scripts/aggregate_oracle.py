import os
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Aggregate independent Oracle runs into unified MT baseline format.")
    parser.add_argument("--dir", type=str, required=True, help="Path to oracle analysis dir (e.g., .../MT10/oracle)")
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks expected")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Directory {args.dir} does not exist.")
        return

    # Find all unique seeds
    seeds = set()
    for item in os.listdir(args.dir):
        if item.startswith("seed_") and "_task_" in item:
            seed_part = item.split("_task_")[0]
            try:
                seed_num = int(seed_part.split("_")[1])
                seeds.add(seed_num)
            except:
                pass
                
    if not seeds:
        print(f"No oracle task directories found in {args.dir}")
        return
        
    print(f"Found seeds: {seeds}")
    
    for seed in seeds:
        print(f"\nProcessing Seed {seed}...")
        task_dfs = []
        missing_tasks = []
        
        for task_id in range(args.num_tasks):
            task_dir = os.path.join(args.dir, f"seed_{seed}_task_{task_id}")
            csv_path = os.path.join(task_dir, "heartbeat.csv")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                task_dfs.append(df)
            else:
                missing_tasks.append(task_id)
                
        if missing_tasks:
            print(f"WARNING: Seed {seed} is missing tasks: {missing_tasks}")
            if not task_dfs:
                continue
                
        # To average properly, we should group by step
        # Assuming steps are perfectly aligned because eval_freq is fixed
        print(f"Loaded {len(task_dfs)} task logs. Computing macro-average...")
        
        # Concatenate and group by Step
        combined_df = pd.concat(task_dfs)
        avg_df = combined_df.groupby("Step").mean().reset_index()
        
        # Create output directory mimicking standard structure
        out_dir = os.path.join(args.dir, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "heartbeat.csv")
        
        avg_df.to_csv(out_path, index=False)
        print(f"Saved aggregated oracle baseline to: {out_path}")

if __name__ == "__main__":
    main()
