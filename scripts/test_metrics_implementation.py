
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import VarShareNetwork

def test_metrics():
    print("--- Testing VarShare Metrics Implementation ---")
    
    # 1. Instantiate Network
    # input_dim=10, output_dim=5, hidden=[32, 32]
    # variant="standard" means all layers deal with metrics? 
    # Actually VarShareNetwork checks variant internally.
    # If variant="standard", all layers are VarShareLayer.
    net = VarShareNetwork(
        input_dim=10, 
        output_dims=[5], 
        hidden_dims=[32, 32], 
        num_tasks=2, 
        variant="standard",
        prior_scale=1.0,
        learned_prior=False
    )
    
    print("Network created.")
    
    # 2. Add some fake data to params to ensure non-zero metrics
    # Layer 0
    with torch.no_grad():
        # Make theta large
        net.layers[0].theta.add_(1.0)
        # Make mu small (to test sparsity)
        net.layers[0].mus["0"].data.fill_(0.0001) 
        # Make mu large in layer 1
        net.layers[1].add_task(0)
        net.layers[1].mus["0"].data.fill_(0.5)
        
    # 3. Get Metrics for Task 0
    metrics = net.get_architectural_metrics(0)
    
    print("\nCalculated Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    # 4. Assertions
    expected_keys = [
        "mean_norm_theta", "mean_norm_mu", "sharing_ratio", "avg_sigma",
        "sparsity", "residual_snr", "posterior_collapse", "sigma_variance",
        "profile_layer_0", "profile_layer_1"
    ]
    
    missing = [k for k in expected_keys if k not in metrics]
    if missing:
        print(f"\nFAILED: Missing keys: {missing}")
        sys.exit(1)
    else:
        print("\nSUCCESS: All keys present.")
        
    # Check Sparsity Logic
    # Layer 0 mu is 0.0001, Theta is ~1.0. Threshold is 0.01. So Layer 0 sparsity should be 1.0.
    # Layer 1 mu is 0.5. Sparsity should be 0.0.
    # Average sparsity should be 0.5.
    print(f"Sparsity: {metrics['sparsity']} (Expected ~0.5)")
    
if __name__ == "__main__":
    test_metrics()
