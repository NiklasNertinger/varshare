
import optuna
import argparse
import sys
import os
from optuna_dashboard import run_server
from optuna.storages import JournalStorage, JournalFileStorage

def main():
    parser = argparse.ArgumentParser(description="Launch Optuna Dashboard for Journal Storage")
    parser.add_argument("--journal-path", type=str, required=True, help="Path to the journal log file")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the dashboard on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    args = parser.parse_args()

    print(f"Loading Journal Storage from: {args.journal_path}")
    
    # Initialize Storage
    try:
        from optuna.storages.journal import JournalFileBackend
        # Optuna 4.0 style if available
        storage = JournalStorage(JournalFileBackend(args.journal_path))
    except (ImportError, AttributeError):
        # Fallback
        storage = JournalStorage(JournalFileStorage(args.journal_path))

    print(f"Starting Dashboard on {args.host}:{args.port}...")
    run_server(storage, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
