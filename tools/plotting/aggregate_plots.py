import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=str, default="analysis", help="Root analysis directory")
    args = parser.parse_args()
    return args

def smooth(values, weight=0.6):
    last = values[0]
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def align_and_aggregate(histories):
    """
    Aggregation logic:
    We need to align multiple runs on the 'step' axis.
    Since steps might slightly differ, we'll interpolate to a common step grid.
    """
    if not histories:
        return None, None, None
        
    # Find common step range
    max_steps = max([h[-1]['step'] for h in histories])
    min_steps = min([h[-1]['step'] for h in histories])
    
    # Create common grid
    # We'll stick to a resolution of ~200 points for plotting
    grid_size = 200
    common_steps = np.linspace(0, min_steps, grid_size)
    
    # Collect all metrics
    metrics = histories[0][0].keys()
    aggregated = {k: {'mean': [], 'std': []} for k in metrics if k != 'step'}
    
    # Process each metric
    for key in metrics:
        if key == 'step': continue
        
        # Check if key is numeric (some might be None or strings? Eval reward can be None)
        # We only interpolate numeric values.
        
        all_runs_interp = []
        valid_key = True
        
        for h in histories:
            steps = np.array([d['step'] for d in h])
            values = [d.get(key) for d in h]
            
            # Filter Nones (e.g. eval_reward)
            # Interpolation requires valid X and Y
            valid_indices = [i for i, v in enumerate(values) if v is not None and isinstance(v, (int, float))]
            
            if len(valid_indices) < 2:
                valid_key = False
                break
                
            x_valid = steps[valid_indices].astype(np.float64)
            y_valid = np.array([float(values[i]) for i in valid_indices], dtype=np.float64)
            
            # Interpolate
            y_interp = np.interp(common_steps, x_valid, y_valid)
            all_runs_interp.append(y_interp)
            
        if valid_key and all_runs_interp:
            arr = np.array(all_runs_interp)
            aggregated[key]['mean'] = np.mean(arr, axis=0)
            aggregated[key]['std'] = np.std(arr, axis=0)
            
    return common_steps, aggregated

def process_experiment(exp_name, exp_path):
    # Only process target experiments
    target_exps = ["exp3_multitask_serial", "exp4_multitask_parallel", "exp7_identical_serial", "exp8_identical_parallel"]
    if exp_name not in target_exps:
        return

    print(f"Processing Experiment: {exp_name}")
    
    # Find all seed folders
    seed_folders = glob.glob(os.path.join(exp_path, "*seed_*"))
    histories = []
    
    for seed_dir in seed_folders:
        json_path = os.path.join(seed_dir, "history.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    histories.append(json.load(f))
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                
    if not histories:
        print(f"No valid history.json files found for {exp_name}")
        return

    common_steps, aggregated = align_and_aggregate(histories)
    if common_steps is None:
        return
        
    # Plotting
    plot_dir = exp_path
    
    for key, data in aggregated.items():
        if len(data['mean']) == 0: continue
        
        plt.figure(figsize=(10, 6))
        
        mean = data['mean']
        std = data['std']
        
        plt.plot(common_steps, mean, label=f"Mean (n={len(histories)})")
        plt.fill_between(common_steps, mean - std, mean + std, alpha=0.2, label="Std Dev")
        
        plt.xlabel("Total Environment Steps")
        plt.ylabel(key)
        plt.title(f"{key} (Aggregated) - {exp_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        safe_key = key.replace("/", "_")
        plt.savefig(os.path.join(plot_dir, f"aggregated_{safe_key}.png"))
        plt.close()
        
    print(f"Generated aggregated plots in {plot_dir}")

def main():
    args = parse_args()
    
    if not os.path.exists(args.analysis_dir):
        print(f"Analysis directory {args.analysis_dir} not found.")
        return
        
    # Iterate over experiment folders
    for exp_name in os.listdir(args.analysis_dir):
        exp_path = os.path.join(args.analysis_dir, exp_name)
        if os.path.isdir(exp_path):
            process_experiment(exp_name, exp_path)

if __name__ == "__main__":
    main()
