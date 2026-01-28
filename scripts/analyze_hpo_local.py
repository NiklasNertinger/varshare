import optuna
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from optuna.visualization import plot_contour, plot_parallel_coordinate
# Handle Optuna versions (Local v4 vs Cluster v3)
try:
    from optuna.storages import JournalStorage, JournalFileBackend as JournalFileStorage
except ImportError:
    from optuna.storages import JournalStorage, JournalFileStorage

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def load_study(storage_path, study_name):
    try:
        storage = JournalStorage(JournalFileStorage(storage_path))
        study = optuna.load_study(study_name=study_name, storage=storage)
        df = study.trials_dataframe()
        # Filter completed trials only
        df = df[df.state == "COMPLETE"]
        # Clean column names (params_lr -> lr)
        df.columns = [c.replace("params_", "") for c in df.columns]
        return df, study
    except KeyError:
        # Study does not exist
        return None, None
    except Exception as e:
        print(f"Skipping {study_name}: {e}")
        return None, None

def is_log_param(col_name):
    """Heuristic to decide if a param should be plotted on log scale"""
    log_keywords = ['lr', 'learning_rate', 'beta', 'alpha', 'sigma', 'lambda', 'weight_decay']
    return any(k in col_name.lower() for k in log_keywords)

def plot_distributions(df, algo_name, out_dir):
    """Plots All vs Top-10% distributions for each param"""
    # Identify hyperparam columns (all numeric columns that aren't metadata)
    param_cols = [c for c in df.columns if c not in ['number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state', 'user_attrs_eval_success']]
    param_cols = [c for c in param_cols if df[c].dtype in [np.float64, np.int64]]

    # Get Top 10%
    threshold = df['value'].quantile(0.90)
    top_df = df[df['value'] >= threshold]

    for col in param_cols:
        plt.figure(figsize=(8, 5))
        
        # Check for log scale
        use_log = is_log_param(col)
        
        sns.kdeplot(df[col], fill=True, color="grey", label="All Trials", alpha=0.3, log_scale=use_log)
        sns.kdeplot(top_df[col], fill=True, color="green", label="Top 10%", alpha=0.5, log_scale=use_log)
        
        plt.title(f"{algo_name}: {col} Distribution (Impact Analysis)")
        plt.xlabel(col)
        plt.ylabel("Density")
        if use_log:
            plt.xscale('log')
            
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{algo_name}_dist_{col}.png"))
        plt.close()

def plot_correlations(df, algo_name, out_dir):
    """
    1. Correlation Matrix of Params for Top 20 Trials (User Request)
    2. Contour Plots for top Interaction pairs
    """
    # Filter Top 20
    top_df = df.nlargest(20, 'value')
    
    param_cols = [c for c in df.columns if c not in ['number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state']]
    # Filter only numeric
    numeric_cols = top_df[param_cols].select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return

    # 1. Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr = top_df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"{algo_name}: Hyperparam Correlation (Top 20 Trials)")
    plt.savefig(os.path.join(out_dir, f"{algo_name}_corr_matrix.png"))
    plt.close()

    # 2. Parallel Coordinates (Global structure)
    # Scales values to seeing 'bundles' of success lines
    # Using pandas/matplotlib instead of plotly static for simplicity in png
    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(
        top_df[list(numeric_cols) + ['value']], 
        'value', 
        colormap='viridis',
        alpha=0.8
    )
    plt.title(f"{algo_name}: Parallel Coordinates (Top 20)")
    plt.gca().legend_.remove() # Legend is just score values, too messy
    plt.savefig(os.path.join(out_dir, f"{algo_name}_parallel_coords.png"))
    plt.close()

def analyze_algorithms(storage_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Define known studies (Check your journal for exact names!)
    # Added possible variations for SoftMod
    studies = [
        # Old HPO
        "mt10_varshare", "mt10_paco", "mt10_soft_mod", "mt10_softmod", "mt10_shared", "mt10_pcgrad", "mt10_oracle",
        # Mega-Batch HPO (11 Studies)
        "mt10_varshare_base",
        "mt10_varshare_emb_onehot",
        "mt10_varshare_emb_learned",
        "mt10_varshare_lora",
        "mt10_varshare_partial",
        "mt10_varshare_reptile",
        "mt10_varshare_scaled_down",
        "mt10_varshare_fixed_prior",
        "mt10_varshare_annealing",
        "mt10_varshare_trigger",
        "mt10_varshare_emp_bayes"
    ]
    
    leaderboard = []

    for study_name in studies:
        print(f"Analyzing {study_name}...")
        df, study_obj = load_study(storage_path, study_name)
        
        if df is None or len(df) == 0:
            # print(f"  -> No data found.") 
            # Silence this since we iterate over variants
            continue
            
        # 1. Basic Stats
        best_value = df['value'].max()
        top_5_mean = df.nlargest(5, 'value')['value'].mean()
        top_5_std = df.nlargest(5, 'value')['value'].std()
        
        leaderboard.append({
            "Algorithm": study_name,
            "Best Score": best_value,
            "Top-5 Mean": top_5_mean,
            "Robustness (Std)": top_5_std,
            "Trials": len(df)
        })
        
        # 2. Detailed Plots
        algo_dir = os.path.join(output_dir, study_name)
        os.makedirs(algo_dir, exist_ok=True)
        
        plot_distributions(df, study_name, algo_dir)
        plot_correlations(df, study_name, algo_dir)
        
        # 3. Slice Plots (Scatter) - Requested
        for col in [c for c in df.columns if c not in ['number', 'value', 'state', 'datetime_start', 'datetime_complete', 'duration']]:
             if df[col].dtype in [np.float64, np.int64]:
                 plt.figure(figsize=(8, 5))
                 use_log = is_log_param(col)
                 
                 sns.scatterplot(data=df, x=col, y='value', alpha=0.6)
                 
                 if use_log:
                     plt.xscale('log')
                     
                 plt.title(f"{study_name}: {col} vs Score")
                 plt.savefig(os.path.join(algo_dir, f"{study_name}_slice_{col}.png"))
                 plt.close()
        
        # 4. Save & Print Top 10
        top_df = df.nlargest(10, 'value')
        top_df.to_csv(os.path.join(algo_dir, "top_10_trials.csv"), index=False)
        
        print(f"\n--- Top 5 Trials for {study_name} ---")
        print(top_df.head(5).to_string(index=False))

    # Print Leaderboard
    leader_df = pd.DataFrame(leaderboard).sort_values("Top-5 Mean", ascending=False)
    print("\n\n==========================================")
    print("🏆 ALGORITHM LEADERBOARD (Top-5 Mean)")
    print("==========================================")
    print(leader_df.to_string(index=False))
    
    leader_df.to_csv(os.path.join(output_dir, "leaderboard.csv"), index=False)
    print(f"\nAnalysis saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=str, default="analysis/optuna_journal.log", help="Path to log file")
    parser.add_argument("--out", type=str, default="analysis/hpo_report", help="Output directory")
    args = parser.parse_args()
    
    analyze_algorithms(args.journal, args.out)
