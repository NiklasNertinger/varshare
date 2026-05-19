import subprocess
import time

def run(variant):
    cmd = ["C:\\Users\\nertinger\\work\\projects\\varshare\\.venv\\Scripts\\python.exe", "train_routing.py", "--variant", variant, "--total-timesteps", "3000", "--cuda", "False", "--eval-mode", "False", "--num-envs", "10", "--n-steps", "128", "--batch-size", "128"]
    start = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.time()
    return end - start

print(f"PCGrad: {run('pcgrad'):.2f}s")
print(f"Routing: {run('routing'):.2f}s")
print(f"Base: {run('base'):.2f}s")
