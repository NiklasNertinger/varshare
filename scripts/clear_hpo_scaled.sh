#!/bin/bash
# Scaled HPO Cleanup Script

# 1. Stop any currently running scaled HPO jobs
echo "Stopping all hpo jobs..."
scancel --name hpo_scaled
scancel --name hpo_paco_scaled
scancel --name hpo_softmod_scaled
scancel --name hpo_shared_scaled
scancel --name hpo_pcgrad_scaled

# 2. Delete scaled HPO data (This preserves the OLD hpo_journal.log)
echo "Deleting scaled HPO results and journal..."
rm -fv /netscratch/$USER/varshare/analysis/optuna_journal_scaled.log
rm -rfv /netscratch/$USER/varshare/analysis/optuna_scaled/

# 3. Clean up the logs directory
echo "Cleaning slurm logs for scaled HPO..."
rm -fv /netscratch/$USER/varshare/logs/hpo_*_scaled_*.out
rm -fv /netscratch/$USER/varshare/logs/hpo_*_scaled_*.err

echo "----------------------------------------"
echo "Done. Scaled HPO track is now empty."
echo "You can re-launch with your sbatch commands."
