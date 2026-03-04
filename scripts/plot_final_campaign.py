import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def smooth_data(df, window=10):
    """Applies a smooth rolling average over the last 10 measurements, as requested."""
    numeric_cols = df.select_dtypes(include='number').columns
    df_smoothed = df.copy()
    df_smoothed[numeric_cols] = df[numeric_cols].rolling(window=window, min_periods=1).mean()
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

def plot_final_campaign():
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
            
            # Smooth by 10 as requested
            df = smooth_data(df, window=10)
            
            parts = csv_file.parts
            seed = parts[-2]
            algo_run_name = parts[-3]
            algo_base = parts[-4]
            env = parts[-5]
            phase = parts[-6] # "phase1" or "phase2"
            
            # Identify variant name clean
            variant_name = algo_base
            
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
                    # Seaborn auto-aggregates across 'seed' with shaded standard deviation
                    sns.lineplot(data=group, x="TOTAL_ENV_STEPS", y=col, label=col.split('/')[-1], errorbar="sd")
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
                sns.lineplot(data=group, x="TOTAL_ENV_STEPS", y=col, label=col.split('_')[-1], errorbar="sd")
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
                sns.lineplot(data=data_group, x="TOTAL_ENV_STEPS", y=col, hue="algo", errorbar="sd", palette="tab20")
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
    plot_final_campaign()
