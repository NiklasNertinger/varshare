"""
Logging Utilities

This module isolates Weights & Biases (W&B) initialization,
experiment configuration logging, and standard CSV heartbeat tracking.
"""

import os
import csv
import wandb
from typing import Any, Optional, Dict, Tuple


def init_logging(
    args: Any,
    seed_dir: str,
    run_name: str,
    resume_wandb_id: Optional[str] = None
) -> Tuple[bool, Any, Optional[str]]:
    """
    Initializes W&B and sets up the local CSV heartbeat logger.
    
    Returns:
        use_wandb (bool): Whether W&B successfully initialized.
        heartbeat_writer (csv.writer): Configured CSV writer (or None if resuming).
        wandb_run_id (str): The active W&B run ID.
    """
    use_wandb = False
    wandb_run_id = None
    
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity="niklas-nertinger-university-of-oxford",
            id=resume_wandb_id,
            resume="must" if resume_wandb_id else "never",
            name=run_name if not resume_wandb_id else None,
            config=vars(args),
            group=args.variant if hasattr(args, 'variant') else args.algo,
            tags=[args.variant if hasattr(args, 'variant') else args.algo, args.env_type]
        )
        use_wandb = True
        wandb_run_id = run.id
    except Exception as e:
        print(f"W&B Initialization Failed: {e}. Running without W&B.")

    heartbeat_path = os.path.join(seed_dir, "heartbeat.csv")
    is_resuming = resume_wandb_id is not None
    heartbeat_mode = "a" if is_resuming else "w"
    
    # We must keep the file open during training, so we return the path and handle it in the main loop
    # or we can let the caller open it. It's safer to let the caller open it so it can be closed properly.
    return use_wandb, heartbeat_path, wandb_run_id


def log_metrics(
    global_step: int,
    metrics: Dict[str, float],
    use_wandb: bool,
    heartbeat_file: Any,
    heartbeat_writer: Any,
    heartbeat_path: str,
    is_resuming: bool
) -> Any:
    """
    Logs metrics to both W&B and the local CSV heartbeat.
    Dynamically creates the CSV header if this is the first write.
    """
    if use_wandb:
        wandb.log(metrics, step=global_step)
        
    if heartbeat_file is not None:
        if heartbeat_writer is None:
            heartbeat_writer = csv.DictWriter(heartbeat_file, fieldnames=["global_step"] + list(metrics.keys()))
            if not is_resuming or os.path.getsize(heartbeat_path) == 0:
                heartbeat_writer.writeheader()
        
        row = {"global_step": global_step}
        row.update(metrics)
        heartbeat_writer.writerow(row)
        heartbeat_file.flush()
        
    return heartbeat_writer
