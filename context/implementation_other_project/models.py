import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class VarShareLayer(nn.Module):
    def __init__(self, in_features, out_features, prior_scale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_scale = prior_scale
        
        # Shared parameters (theta)
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Task-specific parameters will be stored in a dictionary mapping task_id -> parameters
        # We need a way to register these dynamically or fix the number of tasks.
        # For simplicity in this implementation, we will assume a fixed max number of tasks 
        # or require the user to pass task_id to forward().
        # Actually, PyTorch modules usually hold all parameters. 
        # We will use valid nn.ParameterDict or nn.ModuleDict if we knew tasks upfront.
        # But let's stick to a simpler approach: 
        # The Network will hold the task-specific params. 
        # OR: This layer holds a list of mu/sigma for M tasks.
        self.mus = nn.ParameterDict()
        self.rhos = nn.ParameterDict()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def add_task(self, task_id):
        # Initialize mu to 0 (start at shared base)
        self.mus[str(task_id)] = nn.Parameter(torch.zeros_like(self.theta))
        # Initialize rho such that sigma is small but non-zero. 
        # softplus(rho) = sigma. softplus(-5) ~= 0.006.
        self.rhos[str(task_id)] = nn.Parameter(torch.ones_like(self.theta) * -5.0)

        # Storage for frozen noise/weights (for per_episode sampling)
        if not hasattr(self, 'frozen_noise'):
            self.register_buffer('frozen_noise', torch.zeros_like(self.theta))
            self.sampling_mode = "per_forward" # "per_forward" or "fixed"

    def set_sampling_mode(self, mode):
        assert mode in ["per_forward", "fixed"]
        self.sampling_mode = mode

    def sample_phi(self, task_id):
        """Pre-samples noise/weights for the given task and stores them."""
        task_key = str(task_id)
        if task_key not in self.mus:
            # Fallback for inference if strictly needed, but better to error or warn in real usage
            return
            
        # We sample epsilon here and store it. 
        # Actually, for efficiency, let's just sample the noise epsilon.
        # But wait, forward pass needs to know WHICH task's noise to use if we store it globally?
        # If we run multiple tasks in parallel (vector env), 'fixed' mode is tricky if layer is shared.
        # But VarShare usually runs 1 task context at a time or batch of same task?
        # Re-reading spec: "sample phi ~ q_m using reparam... effective weights w..."
        # If we are in "fixed" mode, we assume we are running ONE task (or batch of that task).
        # We will renew 'self.frozen_noise' (epsilon).
        
        self.frozen_noise.normal_()

    def forward(self, x, task_id, sample_mode="sample", num_samples=1):
        """
        sample_mode: 'mean' (deterministic) or 'sample' (stochastic)
        num_samples: Number of samples to average over (McMonte Carlo)
        """
        task_key = str(task_id)
        if task_key not in self.mus:
            raise ValueError(f"Task {task_id} not registered. Call add_task first.")

        theta = self.theta
        mu = self.mus[task_key]
        rho = self.rhos[task_key]
        sigma = F.softplus(rho)
        
        # Mean output (Deterministic path)
        weight_mean = theta + mu
        out_mean = F.linear(x, weight_mean, self.bias)
        
        if sample_mode == "mean":
            return out_mean

        # Sampling Mode logic
        # Check if we are doing "fixed" sampling (episode-wise) or standard training (per-forward)
        use_fixed_sampling = (hasattr(self, 'sampling_mode') and self.sampling_mode == "fixed")

        if self.training or use_fixed_sampling:
            if use_fixed_sampling:
                # Use stored frozen noise (weight-space)
                # w_sample = theta + mu + sigma * self.frozen_noise
                weight_sample = weight_mean + sigma * self.frozen_noise
                return F.linear(x, weight_sample, self.bias)
            else:
                 # Standard Training: Local Reparameterization / Flipout
                 # Var[delta_y] = x^2 @ sigma^2 
                 delta_y_var = F.linear(x.pow(2), sigma.pow(2))
                 delta_y_std = torch.sqrt(delta_y_var + 1e-8)

                 if num_samples == 1:
                     epsilon = torch.randn_like(out_mean)
                     return out_mean + delta_y_std * epsilon
                 else:
                     # Multi-sample
                     outputs = []
                     for _ in range(num_samples):
                         epsilon = torch.randn_like(out_mean)
                         outputs.append(out_mean + delta_y_std * epsilon)
                     return torch.stack(outputs).mean(dim=0)
        else:
            # Eval mode
            return out_mean

    def kl_divergence(self, task_id):
        task_key = str(task_id)
        mu = self.mus[task_key]
        rho = self.rhos[task_key]
        sigma = F.softplus(rho)
        
        # KL(q(phi) || p(phi))
        # q = N(mu, sigma^2)
        # p = N(0, prior_scale^2)
        # KL = log(sig_p / sig_q) + (sig_q^2 + mu^2)/(2 sig_p^2) - 0.5
        
        # We perform element-wise KL and sum
        var_q = sigma.pow(2)
        var_p = self.prior_scale ** 2
        
        kl = 0.5 * (math.log(var_p) - torch.log(var_q) + (var_q + mu.pow(2)) / var_p - 1)
        return kl.sum()

    def prune_task(self, task_id, threshold=1e-3):
        task_key = str(task_id)
        if task_key in self.mus:
            mu = self.mus[task_key]
            # Create mask where abs(mu) < threshold
            mask = torch.abs(mu) < threshold
            # Set mu to 0
            # Since mu is a Parameter, we use data
            mu.data[mask] = 0.0
            # Optionally remove from optimization if we were doing further training?
            # For post-training compression, just zeroing is enough.
            return mask.sum().item(), mu.numel()
        return 0, 0

import math

class VarShareNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], num_tasks=1, prior_scale=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_tasks = num_tasks
        
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layer = VarShareLayer(prev_dim, h_dim, prior_scale=prior_scale)
            for t in range(num_tasks):
                layer.add_task(t)
            self.layers.append(layer)
            prev_dim = h_dim
            
        self.output_layer = VarShareLayer(prev_dim, output_dim, prior_scale=prior_scale)
        for t in range(num_tasks):
            self.output_layer.add_task(t)

    def set_sampling_mode(self, mode):
        for layer in self.layers:
            layer.set_sampling_mode(mode)
        self.output_layer.set_sampling_mode(mode)

    def sample_phi(self, task_id):
        for layer in self.layers:
            layer.sample_phi(task_id)
        self.output_layer.sample_phi(task_id)
            
    def forward(self, x, task_id, **kwargs):
        # kwargs to swallow extra args from wrappers if any
        for layer in self.layers:
            x = F.relu(layer(x, task_id))
        return self.output_layer(x, task_id)
        
    def get_kl(self, task_id):
        kl = 0
        for layer in self.layers:
            kl += layer.kl_divergence(task_id)
        kl += self.output_layer.kl_divergence(task_id)
        return kl

    def prune(self, task_id, threshold=1e-3):
        total_pruned = 0
        total_params = 0
        for layer in self.layers:
            p, t = layer.prune_task(task_id, threshold)
            total_pruned += p
            total_params += t
        p, t = self.output_layer.prune_task(task_id, threshold)
        total_pruned += p
        total_params += t
        return total_pruned, total_params

class IndependentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], num_tasks=1):
        super().__init__()
        self.networks = nn.ModuleList([
            self._build_net(input_dim, output_dim, hidden_dims) for _ in range(num_tasks)
        ])
        
    def _build_net(self, i, o, h):
        layers = []
        prev = i
        for hid in h:
            layers.append(nn.Linear(prev, hid))
            layers.append(nn.ReLU())
            prev = hid
        layers.append(nn.Linear(prev, o))
        return nn.Sequential(*layers)
        
    def forward(self, x, task_id):
        # Handling masking/batching if task_id is a tensor is tricky.
        # Assuming single task_id integer for the batch or handle splitting?
        # For simplicity: x is (Batch, Dim). task_id is int.
        # We assume all samples in batch are same task, or logic happens outside.
        return self.networks[task_id](x)

class SharedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], num_tasks=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dims[1], output_dim) for _ in range(num_tasks)
        ])
        
    def forward(self, x, task_id):
        features = self.encoder(x)
        return self.heads[task_id](features)

class ContextualNetwork(nn.Module):
    """
    Standard Multi-Task Baseline: Contextual Policy.
    Uses a shared backbone and a learnable task embedding concatenated to the input.
    All weights are shared (including the head).
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], num_tasks=4, embedding_dim=10):
        super().__init__()
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        
        # Task Embedding
        self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
        
        # Shared Backbone
        # Input is State + Embedding
        current_dim = input_dim + embedding_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # Shared Head (The embedding provides the context, so we can share the head)
        self.head = nn.Linear(current_dim, output_dim)
        
    def forward(self, x, task_idx):
        # task_idx might be int or tensor
        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=x.device)
        
        # Ensure task_idx matches batch size if x is a batch
        # Case A: x is (B, D), task_idx is (1) -> Expand to (B)
        # Case B: x is (B, D), task_idx is (B) -> Keep
        if task_idx.dim() == 0 or len(task_idx) == 1:
            if x.dim() > 1: # Batch > 1
                 task_idx = task_idx.expand(x.size(0))
            
        embeds = self.task_embedding(task_idx)
        
        # Concatenate state and embedding
        x_emb = torch.cat([x, embeds], dim=1)
        
        features = self.backbone(x_emb)
        return self.head(features)

class SoftSharingLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mus = nn.ParameterDict()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def add_task(self, task_id):
        self.mus[str(task_id)] = nn.Parameter(torch.zeros_like(self.theta))

    def forward(self, x, task_id):
        task_key = str(task_id)
        theta = self.theta
        mu = self.mus[task_key]
        weight = theta + mu
        return F.linear(x, weight, self.bias)

    def l2_penalty(self, task_id):
        mu = self.mus[str(task_id)]
        return mu.pow(2).sum()

class SoftSharingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], num_tasks=1):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layer = SoftSharingLayer(prev_dim, h_dim)
            for t in range(num_tasks):
                layer.add_task(t)
            self.layers.append(layer)
            prev_dim = h_dim
        
        self.output_layer = SoftSharingLayer(prev_dim, output_dim)
        for t in range(num_tasks):
            self.output_layer.add_task(t)

    def forward(self, x, task_id):
        for layer in self.layers:
            x = F.relu(layer(x, task_id))
        return self.output_layer(x, task_id)

    def get_l2_penalty(self, task_id):
        l2 = 0
        for layer in self.layers:
            l2 += layer.l2_penalty(task_id)
        l2 += self.output_layer.l2_penalty(task_id)
        return l2

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, num_tasks=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features) # Frozen pretrained? Or trained shared?
        # In this RL context, we learn 'linear' as the shared base from scratch.
        
        self.rank = rank
        self.num_tasks = num_tasks
        
        # LoRA A and B parameters per task
        # B is initialized to 0, A is random.
        # W = W0 + B @ A
        # Shapes: W0: (out, in). B: (out, r). A: (r, in).
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        
        for t in range(num_tasks):
            self.add_task(t)
            
    def add_task(self, task_id):
        # A: Gaussian initialization
        # B: Zero initialization (so starts as identity/base)
        t = str(task_id)
        self.lora_A[t] = nn.Parameter(torch.randn(self.rank, self.linear.in_features) * 0.01)
        self.lora_B[t] = nn.Parameter(torch.zeros(self.linear.out_features, self.rank))
        
    def forward(self, x, task_id):
        # Base forward
        base_out = self.linear(x)
        
        # LoRA forward
        # W_lora = B @ A
        # out = x @ W_lora.T = x @ (B @ A).T = x @ A.T @ B.T
        # Or simply: lora_out = (x @ A.T) @ B.T
        
        t = str(task_id)
        A = self.lora_A[t]
        B = self.lora_B[t]
        
        lora_out = (x @ A.T) @ B.T
        return base_out + lora_out

class LoRANetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], num_tasks=1, rank=4):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(LoRALayer(prev_dim, h_dim, rank=rank, num_tasks=num_tasks))
            prev_dim = h_dim
        self.output_layer = LoRALayer(prev_dim, output_dim, rank=rank, num_tasks=num_tasks)
        
    def forward(self, x, task_id):
        for layer in self.layers:
            x = F.relu(layer(x, task_id))
        return self.output_layer(x, task_id)

class LargeNetwork(nn.Module):
    """
    BRC-Lite: Just a much bigger Shared Network to test the Capacity Hypothesis.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256, 256], num_tasks=1):
        super().__init__()
        # Standard SharedNetwork logic but expecting bigger dims
        self.encoder = nn.Sequential()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.encoder = nn.Sequential(*layers)
        
        self.heads = nn.ModuleList([
            nn.Linear(prev, output_dim) for _ in range(num_tasks)
        ])
        
    def forward(self, x, task_id):
        features = self.encoder(x)
        return self.heads[task_id](features)
