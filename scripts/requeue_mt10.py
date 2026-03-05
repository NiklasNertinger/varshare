import subprocess
import time
import os

def main():
    user = os.environ.get("USER", "nertinger")
    print(f"Fetching current MT10 jobs from squeue for user: {user}...")
    
    try:
        result = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i %j"], capture_output=True, text=True, check=True)
    except Exception as e:
        print(f"Failed to run squeue: {e}")
        return
        
    mt10_jobs = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        if "MT10" in line:
            job_id = line.split()[0]
            mt10_jobs.append(job_id)
            
    if not mt10_jobs:
        print("No MT10 jobs found currently in the queue.")
    else:
        print(f"Found {len(mt10_jobs)} MT10 array jobs. Cancelling...")
        for jid in mt10_jobs:
            subprocess.run(["scancel", jid])
            print(f"Cancelled {jid}")
            
    print("Waiting a few seconds for SLURM to legally clear them from the queue...")
    time.sleep(3)
    
    print("Pulling latest git changes to ensure we have the 72h fix...")
    subprocess.run(["git", "pull"])
    
    print("\nResubmitting MT10 arrays with the updated 72h configuration limit...")
    subprocess.run(["python", "scripts/submit_final_campaign.py", "--envs", "MT10"])
    
    print("\nDone! MT10 jobs have been safely requeued.")

if __name__ == "__main__":
    main()
