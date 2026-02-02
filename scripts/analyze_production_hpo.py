import optuna
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Handle Optuna versions
try:
    from optuna.storages import JournalStorage, JournalFileBackend as JournalFileStorage
except ImportError:
    from optuna.storages import JournalStorage, JournalFileStorage

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def load_study_data(storage, study_name):
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        df = study.trials_dataframe()
        # Filter completed trials only
        df = df[df.state == "COMPLETE"]
        if len(df) == 0:
            return None, study
            
        # Clean column names (params_lr -> lr)
        df.columns = [c.replace("params_", "") for c in df.columns]
        return df, study
    except Exception as e:
        print(f"Error loading {study_name}: {e}")
        return None, None

def is_log_param(col_name):
    log_keywords = ['lr', 'learning_rate', 'beta', 'alpha', 'sigma', 'lambda', 'weight_decay']
    return any(k in col_name.lower() for k in log_keywords)

def plot_distributions(df, algo_name, out_dir):
    param_cols = [c for c in df.columns if c not in ['number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state', 'user_attrs_eval_success'] and not c.startswith("user_attrs")]
    param_cols = [c for c in param_cols if df[c].dtype in [np.float64, np.int64]]

    if not param_cols: return

    threshold = df['value'].quantile(0.90)
    top_df = df[df['value'] >= threshold]

    for col in param_cols:
        try:
            plt.figure(figsize=(8, 5))
            use_log = is_log_param(col)
            
            sns.kdeplot(df[col], fill=True, color="grey", label="All Trials", alpha=0.3, log_scale=use_log, warn_singular=False)
            if len(top_df) > 1:
                sns.kdeplot(top_df[col], fill=True, color="green", label="Top 10%", alpha=0.5, log_scale=use_log, warn_singular=False)
            
            plt.title(f"{algo_name}: {col} Distribution")
            plt.xlabel(col)
            plt.legend()
            plt.savefig(os.path.join(out_dir, f"dist_{col}.png"))
            plt.close()
        except Exception as e:
            print(f"Skipping plot for {col}: {e}")
            plt.close()

def plot_correlations(df, algo_name, out_dir):
    if len(df) < 5: return
    
    top_df = df.nlargest(min(20, len(df)), 'value')
    param_cols = [c for c in df.columns if c not in ['number', 'value', 'datetime_start', 'datetime_complete', 'duration', 'state'] and not c.startswith("user_attrs")]
    numeric_cols = top_df[param_cols].select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2: return

    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr = top_df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"{algo_name}: Correlation (Top {len(top_df)})")
    plt.savefig(os.path.join(out_dir, "corr_matrix.png"))
    plt.close()

def analyze_journal(journal_path, base_out_dir):
    if not os.path.exists(journal_path):
        print(f"Journal not found: {journal_path}")
        return

    print(f"\n>>> Analyzing Journal: {journal_path}")
    
    # Initialize Storage
    try:
        if "JournalFileBackend" in globals():
             storage = JournalStorage(JournalFileBackend(journal_path))
        else:
             storage = JournalStorage(JournalFileStorage(journal_path))
    except Exception as e:
        print(f"Failed to open storage: {e}")
        return

    # Dynamic Study Discovery
    summaries = optuna.get_all_study_summaries(storage=storage)
    studies = [s.study_name for s in summaries]
    print(f"Found {len(studies)} studies: {studies}")

    leaderboard = []

    for study_name in studies:
        print(f"  Processing {study_name}...")
        df, study_obj = load_study_data(storage, study_name)
        
        if df is None:
            continue
            
        # Stats
        best_value = df['value'].max()
        top_5 = df.nlargest(5, 'value')
        
        leaderboard.append({
            "Algorithm": study_name,
            "Best Score": best_value,
            "Top-5 Mean": top_5['value'].mean(),
            "Trials (Completed)": len(df)
        })
        
        # Output Dir
        algo_dir = os.path.join(base_out_dir, study_name)
        os.makedirs(algo_dir, exist_ok=True)
        
        # Plots
        plot_distributions(df, study_name, algo_dir)
        plot_correlations(df, study_name, algo_dir)
        
        # Slice Plots
        param_cols = [c for c in df.columns if c not in ['number', 'value', 'state', 'datetime_start', 'datetime_complete', 'duration'] and not c.startswith("user_attrs")]
        for col in param_cols:
             if df[col].dtype in [np.float64, np.int64]:
                 plt.figure(figsize=(8, 5))
                 use_log = is_log_param(col)
                 try:
                     sns.scatterplot(data=df, x=col, y='value', alpha=0.6)
                     if use_log: plt.xscale('log')
                     plt.title(f"{study_name}: {col} vs Score")
                     plt.savefig(os.path.join(algo_dir, f"slice_{col}.png"))
                 except: pass
                 plt.close()

        # Save Top 10
        df.nlargest(10, 'value').to_csv(os.path.join(algo_dir, "top_10_trials.csv"), index=False)

    # Save Leaderboard
    if leaderboard:
        leader_df = pd.DataFrame(leaderboard).sort_values("Top-5 Mean", ascending=False)
        leader_df.to_csv(os.path.join(base_out_dir, "leaderboard.csv"), index=False)
        print("\nLeaderboard:")
        print(leader_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=str, default="analysis", help="Base directory containing logs")
    args = parser.parse_args()
    
    # 1. Scaled Baselines
    analyze_journal(
        os.path.join(args.analysis_dir, "optuna_journal_scaled.log"),
        os.path.join(args.analysis_dir, "plots", "scaled_baselines")
    )
    
    # 2. Mega HPO
    analyze_journal(
        os.path.join(args.analysis_dir, "optuna_journal_mega.log"),
        os.path.join(args.analysis_dir, "plots", "mega_variants")
    )

if __name__ == "__main__":
    main()
