import pathlib
import pandas as pd

def check_grads():
    base_dir = pathlib.Path("analysis/TEST_MEGA")
    results = {}
    
    for csv_file in base_dir.rglob("heartbeat.csv"):
        try:
            df = pd.read_csv(csv_file)
            if "diagnostics/grad_norm" in df.columns:
                avg_grad = df["diagnostics/grad_norm"].mean()
                variant = csv_file.parent.parent.name.replace("TEST_", "")
                results[variant] = avg_grad
        except Exception as e:
            pass
            
    # Print sorted
    print("Average Gradient Norms by Variant (Local 1,000 Step Test):")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    check_grads()
