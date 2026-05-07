import os
import sys
import time
import subprocess

def run_cmd(cmd, shell=True):
    try:
        out = subprocess.check_output(cmd, shell=shell, stderr=subprocess.STDOUT)
        return out.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}\nOutput: {e.output.decode('utf-8')}")
        raise e

def main():
    print("==========================================================================")
    # Master Pipeline Orchestrator Daemon
    print("STARTING MASTER OVERNIGHT HPO & BENCHMARKING PIPELINE")
    print("==========================================================================")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # 1. Submit all HPO Sweeps
    print("\n[Step 1] Triggering HPO submissions via submit_all_hpos.sh...")
    sub_output = run_cmd("bash scripts/submit_all_hpos.sh")
    print(sub_output)
    
    # Extract Job IDs from the ALL_JOB_IDS line
    job_ids = []
    for line in sub_output.split("\n"):
        if line.startswith("ALL_JOB_IDS:"):
            job_ids = line.replace("ALL_JOB_IDS:", "").strip().split()
            break
            
    if not job_ids:
        print("Error: Failed to capture SLURM Job IDs from submission script.")
        sys.exit(1)
        
    print(f"\nSuccessfully tracking {len(job_ids)} SLURM Job IDs: {', '.join(job_ids)}")
    
    # 2. Polling Loop
    print("\n[Step 2] Entering active polling loop. Will check squeue every 5 minutes...")
    comma_ids = ",".join(job_ids)
    
    while True:
        # Check active jobs in squeue matching our HPO array Job IDs
        # -h removes header, -j filters by ID list, -o %i prints the Job ID
        squeue_out = run_cmd(f"squeue -j {comma_ids} -h -o '%i'")
        
        # If output is empty, all jobs are completed (no longer active/pending)
        active_lines = [line.strip() for line in squeue_out.split("\n") if line.strip()]
        
        if len(active_lines) == 0:
            print("\n>>> All HPO Array Jobs have completed!")
            break
            
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Active/Pending HPO jobs remaining: {len(active_lines)}")
        
        # Sleep for 5 minutes (300s) before checking again
        time.sleep(300)
        
    # 3. Extract Best Hyperparameters
    print("\n[Step 3] Executing champion selection extraction...")
    extract_out = run_cmd("python scripts/extract_hpo_results.py")
    print(extract_out)
    
    # 4. Trigger Final Benchmarking Campaign
    print("\n[Step 4] Starting automated benchmarking campaign for MT10...")
    submit_out = run_cmd("python scripts/submit_final_campaign.py --envs MT10")
    print(submit_out)
    
    print("\n==========================================================================")
    print("MASTER OVERNIGHT PIPELINE COMPLETE - ALL CHAMPIONS SUBMITTED!")
    print("==========================================================================")

if __name__ == "__main__":
    main()
