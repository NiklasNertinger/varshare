import os
import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def smooth_data(df, window=50):
    """Applies a smooth rolling average over the last X measurements, as requested."""
    # Only smooth training curves to preserve exact evaluation data resolution
    train_cols = [c for c in df.columns if 'train' in c]
    df_smoothed = df.copy()
    if train_cols:
        df_smoothed[train_cols] = df[train_cols].rolling(window=window, min_periods=1).mean()
    return df_smoothed

def generate_seed_plots(df, seed_dir):
    """Generates standard tracking plots and task-specific breakdowns for a single seed."""
    os.makedirs(seed_dir, exist_ok=True)
    metrics_to_plot = {
        "reward": ["performance/train_reward_50", "eval/mean_reward"],
        "success": ["performance/train_success_50", "eval/mean_success"],
        "loss": ["loss/policy", "loss/value"],
        "diagnostics": ["diagnostics/grad_norm", "loss/kl_penalty"]
    }
    
    # 1. Main Plots
    for name, cols in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        for col in cols:
            if col in df.columns:
                sns.lineplot(data=df, x="TOTAL_ENV_STEPS", y=col, label=col.split('/')[-1])
        plt.title(f"{name.capitalize()} over Step")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(seed_dir, f"{name}.png"))
        plt.close()

    # 2. Task-Specific Plots
    tasks_dir = os.path.join(seed_dir, "tasks")
    os.makedirs(tasks_dir, exist_ok=True)
    task_reward_cols = [c for c in df.columns if c.startswith("eval/reward_task_")]
    if task_reward_cols:
        plt.figure(figsize=(12, 8))
        for col in task_reward_cols:
            sns.lineplot(data=df, x="TOTAL_ENV_STEPS", y=col, label=col.split('_')[-1])
        plt.title("Task-Specific Rewards")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(tasks_dir, "all_tasks_reward.png"))
        plt.close()

def plot_final_campaign(base_dir_str=None):
    if base_dir_str:
        base_dir = pathlib.Path(base_dir_str)
    else:
        base_dir = pathlib.Path("/netscratch") / os.environ.get("USER", "default_user") / "varshare" / "analysis" / "final_eval"
    
    if not base_dir.exists():
        print(f"Directory {base_dir} not found. Ensure experiments have run.")
        return

    all_data = []

    # 1. Parse and plot individual seeds, collect for aggregation
    for csv_file in base_dir.rglob("heartbeat.csv"):
        try:
            df = pd.read_csv(csv_file)
            if df.empty: continue
            
            # Smooth by 50 for training curves as requested
            df = smooth_data(df, window=50)
            
            # Aggressively downsample array to max 600 points to prevent OOM & Seaborn hangs
            max_points = 600
            if len(df) > max_points:
                chunk_size = len(df) // max_points
                df = df.groupby(df.index // chunk_size).mean(numeric_only=True)
            
            # Identify hierarchy based on relative path to base_dir
            try:
                rel_parts = csv_file.relative_to(base_dir).parts
                if len(rel_parts) >= 6:
                    phase = rel_parts[-6]
                    env = rel_parts[-5]
                    variant_name = rel_parts[-4]
                    seed = rel_parts[-2]
                else:
                    # Fallback for local testing arrays
                    phase = "test_phase"
                    env = "TEST_ENV"
                    variant_name = rel_parts[-3] if len(rel_parts) >= 3 else "unknown"
                    seed = rel_parts[-2] if len(rel_parts) >= 2 else "seed_1"
            except ValueError:
                continue
            
            # Plot individual seed details
            seed_dir = csv_file.parent
            generate_seed_plots(df, seed_dir)
            
            df["seed"] = seed
            df["algo"] = variant_name
            df["env"] = env
            df["phase"] = phase
            df["algo_dir"] = str(seed_dir.parent) 
            all_data.append(df)
        except Exception as e:
            print(f"Failed to process {csv_file}: {e}")

    if not all_data:
        print("No valid CSV logs found to plot.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # 2. Plot Algorithm Aggregations (across 3 seeds)
    for (env, algo), group in full_df.groupby(["env", "algo"]):
        algo_out_dir = group["algo_dir"].iloc[0] # The directory above seed_x
        os.makedirs(algo_out_dir, exist_ok=True)
        
        # Main aggregated metrics
        metrics = {"reward": ["performance/train_reward_50", "eval/mean_reward"],
                   "success": ["performance/train_success_50", "eval/mean_success"]}
                   
        for name, cols in metrics.items():
            plt.figure(figsize=(10, 6))
            for col in cols:
                if col in group.columns:
                    # Drop NaNs before plotting to prevent seaborn hanging
                    clean_group = group.dropna(subset=[col, "TOTAL_ENV_STEPS"])
                    if clean_group.empty: continue
                    
                    # Seaborn auto-aggregates across 'seed' with shaded standard deviation
                    sns.lineplot(data=clean_group, x="TOTAL_ENV_STEPS", y=col, label=col.split('/')[-1], errorbar="sd")
            plt.title(f"Aggregated {name.capitalize()} - {algo} ({env})")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(algo_out_dir, f"aggregated_{name}.png"))
            plt.close()
            
        # Aggregated Task Breakdown
        task_out_dir = os.path.join(algo_out_dir, "tasks")
        os.makedirs(task_out_dir, exist_ok=True)
        task_cols = [c for c in group.columns if c.startswith("eval/reward_task_")]
        if task_cols:
            plt.figure(figsize=(12, 8))
            for col in task_cols:
                clean_task_group = group.dropna(subset=[col, "TOTAL_ENV_STEPS"])
                if clean_task_group.empty: continue
                sns.lineplot(data=clean_task_group, x="TOTAL_ENV_STEPS", y=col, label=col.split('_')[-1], errorbar="sd")
            plt.title(f"Aggregated Task-Specific Rewards - {algo}")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(task_out_dir, "aggregated_tasks_reward.png"))
            plt.close()

    # 3. Stacked Comparison Plots (Phase 1 Only & Phase 1+2)
    for env, env_group in full_df.groupby("env"):
        phase1_group = env_group[env_group["phase"] == "phase1"]
        
        def plot_stacked(data_group, title_prefix, out_folder):
            out_path = base_dir / out_folder / env
            os.makedirs(out_path, exist_ok=True)
            
            targets = [
                ("eval_reward", "eval/mean_reward"),
                ("train_reward", "performance/train_reward_50"),
                ("eval_success_rate", "eval/mean_success"),
                ("train_success_rate", "performance/train_success_50")
            ]
            
            for plot_name, col in targets:
                if col not in data_group.columns: continue
                # Skip success plots if completely 0.0 (e.g., CP/LL don't use it)
                if data_group[col].max() == 0.0: continue
                
                plt.figure(figsize=(14, 8))
                
                clean_data = data_group.dropna(subset=[col, "TOTAL_ENV_STEPS"])
                if clean_data.empty: continue
                
                sns.lineplot(data=clean_data, x="TOTAL_ENV_STEPS", y=col, hue="algo", errorbar="sd", palette="tab20")
                plt.title(f"{title_prefix} - {plot_name} ({env})")
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(os.path.join(out_path, f"{plot_name}.png"))
                plt.close()

        # Phase 1 plots
        if not phase1_group.empty:
            plot_stacked(phase1_group, "Phase 1 Comparison", "phase1_comparison_plots")
        
        # Phase 1+2 (All algos) plots
        if not env_group.empty:
            plot_stacked(env_group, "Final Comparison (All Algorithms)", "final_comparison_plots")

    print(f"Plotting complete. Outputs located in {base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for VarShare evaluation campaigns.")
    parser.add_argument("--dir", type=str, default=None, help="Base directory to scan for heartbeat.csv files.")
    args = parser.parse_args()
    
    plot_final_campaign(args.dir)
