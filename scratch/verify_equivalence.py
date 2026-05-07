import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

# Setup random seeds for absolute consistency
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. SOFT MODULARIZATION EQUIVALENCE TEST
# =====================================================================

class SoftModLinearExpert(nn.Module):
    def __init__(self, num_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.rand(self.num_experts, self.in_features, self.out_features))
        self.bias = nn.Parameter(torch.rand(self.num_experts, 1, self.out_features))

        for i in range(self.num_experts):
            nn.init.xavier_uniform_(self.weight[i])
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(self.bias, x, self.weight)

class SoftModRoutingNetwork(nn.Module):
    def __init__(self, hidden_features: int, num_experts_per_layer: int, num_layers: int):
        super().__init__()
        self.num_experts_per_layer = num_experts_per_layer
        self.W_d = nn.ModuleList([nn.Linear(hidden_features, num_experts_per_layer ** 2) for _ in range(num_layers)])
        self.W_u = nn.ModuleList([nn.Linear(num_experts_per_layer ** 2, hidden_features) for _ in range(num_layers - 1)])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        p = self.W_d[0](F.relu(inp))
        prob = [p]
        for W_u, W_d in zip(self.W_u, self.W_d[1:]):
            p = W_d(F.relu((W_u(prob[-1]) * inp)))
            prob.append(p)
            
        prob_tensor = torch.cat([
            F.softmax(logprob.reshape(logprob.shape[0], self.num_experts_per_layer, self.num_experts_per_layer), dim=2).unsqueeze(0)
            for logprob in prob
        ], dim=0)
        return prob_tensor

class OldSoftModularizedMLP(nn.Module):
    def __init__(self, num_experts: int, in_features: int, out_features: int, num_layers: int, hidden_features: int):
        super().__init__()
        self.layers = nn.ModuleList()
        current_in = hidden_features
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                SoftModLinearExpert(num_experts, current_in, hidden_features),
                nn.ReLU()
            ))
            current_in = hidden_features
        self.layers.append(SoftModLinearExpert(num_experts, current_in, out_features))
        self.routing_network = SoftModRoutingNetwork(hidden_features, num_experts, num_layers - 1)

    def forward(self, f_obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inp = f_obs * z
        probs = self.routing_network(inp)
        probs = probs.permute(0, 2, 3, 1)
        num_experts = probs.shape[1]
        x = f_obs.unsqueeze(0).expand(num_experts, -1, -1)
        for index, layer in enumerate(self.layers[:-1]):
            p = probs[index]
            x = layer(x)
            _out = p.unsqueeze(-1) * x.unsqueeze(0).repeat(num_experts, 1, 1, 1)
            x = _out.sum(dim=1)
        out = self.layers[-1](x).sum(dim=0)
        return out

class NewSoftModularizedMLP(nn.Module):
    def __init__(self, num_experts: int, in_features: int, out_features: int, num_layers: int, hidden_features: int):
        super().__init__()
        self.layers = nn.ModuleList()
        current_in = hidden_features
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                SoftModLinearExpert(num_experts, current_in, hidden_features),
                nn.ReLU()
            ))
            current_in = hidden_features
        self.layers.append(SoftModLinearExpert(num_experts, current_in, out_features))
        self.routing_network = SoftModRoutingNetwork(hidden_features, num_experts, num_layers - 1)

    def forward(self, f_obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inp = f_obs * z
        probs = self.routing_network(inp) # (num_layers-1, B, E, E)
        num_experts = probs.shape[2]
        
        x = f_obs.unsqueeze(0).expand(num_experts, -1, -1) # (E, B, D)
        for index, layer in enumerate(self.layers[:-1]):
            p = probs[index] # (B, E, E)
            x = layer(x) # (E, B, D)
            
            x_perm = x.permute(1, 0, 2) # (B, E, D)
            x_new = torch.bmm(p, x_perm) # (B, E, D)
            x = x_new.permute(1, 0, 2) # (E, B, D)
            
        out = self.layers[-1](x).sum(dim=0) # (B, out_dim)
        return out

def test_softmod():
    print("--- Running SoftMod Equivalence Test ---")
    B = 16
    hidden_features = 64
    out_features = 10
    num_layers = 3
    num_experts = 4
    
    old_mlp = OldSoftModularizedMLP(num_experts, hidden_features, out_features, num_layers, hidden_features)
    new_mlp = NewSoftModularizedMLP(num_experts, hidden_features, out_features, num_layers, hidden_features)
    
    # Load exact same weights
    new_mlp.load_state_dict(old_mlp.state_dict())
    
    # Random Inputs
    f_obs = torch.randn(B, hidden_features)
    z = torch.randn(B, hidden_features)
    
    out_old = old_mlp(f_obs, z)
    out_new = new_mlp(f_obs, z)
    
    # Test Output Equality
    diff = torch.max(torch.abs(out_old - out_new)).item()
    print(f"Max output difference: {diff:.12f}")
    assert torch.allclose(out_old, out_new, atol=1e-5), f"Assertion failed with diff: {diff}"
    print("[OK] SoftMod forward pass is mathematically identical!")
    
    # Test Gradients
    loss_old = out_old.sum()
    loss_new = out_new.sum()
    
    old_mlp.zero_grad()
    loss_old.backward()
    grads_old = {name: p.grad.clone() for name, p in old_mlp.named_parameters() if p.grad is not None}
    
    new_mlp.zero_grad()
    loss_new.backward()
    grads_new = {name: p.grad.clone() for name, p in new_mlp.named_parameters() if p.grad is not None}
    
    for name in grads_old:
        grad_diff = torch.max(torch.abs(grads_old[name] - grads_new[name])).item()
        assert torch.allclose(grads_old[name], grads_new[name], atol=1e-5), f"Grad assertion failed for {name} with diff {grad_diff}"
    print("[OK] SoftMod gradients are mathematically identical!")

# =====================================================================
# 2. INDEPENDENT BASELINE EQUIVALENCE TEST
# =====================================================================

class OldIndependentActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dims=[32, 32], num_tasks=4):
        super().__init__()
        self.num_tasks = num_tasks
        obs_shape = int(np.array(observation_space.shape).prod())
        self.action_dim = int(np.array(action_space.shape).prod()) if not hasattr(action_space, 'n') else int(action_space.n)
        self.is_continuous = not hasattr(action_space, 'n')
        
        def build_mlp(out_dim):
            layers = []
            curr_in = obs_shape
            for h in hidden_dims:
                layers.append(nn.Linear(curr_in, h))
                layers.append(nn.ReLU())
                curr_in = h
            layers.append(nn.Linear(curr_in, out_dim))
            return nn.Sequential(*layers)
            
        self.actors = nn.ModuleList([build_mlp(self.action_dim) for _ in range(num_tasks)])
        self.critics = nn.ModuleList([build_mlp(1) for _ in range(num_tasks)])
        if self.is_continuous:
            self.actor_logstds = nn.Parameter(torch.zeros(num_tasks, 1, self.action_dim))

    def _format_task_idx(self, x, task_idx):
        if task_idx is None:
            task_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        elif not isinstance(task_idx, torch.Tensor):
            task_idx = torch.tensor([task_idx], dtype=torch.long, device=x.device)
            if task_idx.shape[0] == 1:
                task_idx = task_idx.expand(x.shape[0])
        return task_idx.long()

    def get_value(self, x, task_idx=None):
        task_idx = self._format_task_idx(x, task_idx)
        out = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
        for t in torch.unique(task_idx):
            t_val = t.item()
            mask = (task_idx == t).flatten()
            out[mask] = self.critics[t_val](x[mask])
        return out

    def get_action_and_value(self, x, action=None, task_idx=None, sample=True):
        task_idx = self._format_task_idx(x, task_idx)
        action_mean = torch.empty((x.shape[0], self.action_dim), device=x.device, dtype=x.dtype)
        value = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
        if self.is_continuous:
            action_logstd = torch.empty((x.shape[0], self.action_dim), device=x.device, dtype=x.dtype)
        
        for t in torch.unique(task_idx):
            t_val = t.item()
            mask = (task_idx == t).flatten()
            action_mean[mask] = self.actors[t_val](x[mask])
            value[mask] = self.critics[t_val](x[mask])
            if self.is_continuous:
                action_logstd[mask] = self.actor_logstds[t_val].expand(mask.sum().item(), self.action_dim)
            
        if self.is_continuous:
            action_std = torch.exp(action_logstd)
            probs = torch.distributions.Normal(action_mean, action_std)
            if action is None:
                action = probs.sample() if sample else action_mean
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
        else:
            probs = torch.distributions.Categorical(logits=action_mean)
            if action is None:
                action = probs.sample() if sample else torch.argmax(action_mean, dim=1)
            return action, probs.log_prob(action), probs.entropy(), value

class NewIndependentActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dims=[32, 32], num_tasks=4):
        super().__init__()
        self.num_tasks = num_tasks
        obs_shape = int(np.array(observation_space.shape).prod())
        self.action_dim = int(np.array(action_space.shape).prod()) if not hasattr(action_space, 'n') else int(action_space.n)
        self.is_continuous = not hasattr(action_space, 'n')
        
        def build_mlp(out_dim):
            layers = []
            curr_in = obs_shape
            for h in hidden_dims:
                layers.append(nn.Linear(curr_in, h))
                layers.append(nn.ReLU())
                curr_in = h
            layers.append(nn.Linear(curr_in, out_dim))
            return nn.Sequential(*layers)
            
        self.actors = nn.ModuleList([build_mlp(self.action_dim) for _ in range(num_tasks)])
        self.critics = nn.ModuleList([build_mlp(1) for _ in range(num_tasks)])
        if self.is_continuous:
            self.actor_logstds = nn.Parameter(torch.zeros(num_tasks, 1, self.action_dim))

    def _format_task_idx(self, x, task_idx):
        if task_idx is None:
            task_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        elif not isinstance(task_idx, torch.Tensor):
            task_idx = torch.tensor([task_idx], dtype=torch.long, device=x.device)
            if task_idx.shape[0] == 1:
                task_idx = task_idx.expand(x.shape[0])
        return task_idx.long()

    def _forward_mlp(self, x, task_idx, mlps):
        first_mlp = mlps[0]
        linear_indices = [i for i, layer in enumerate(first_mlp) if isinstance(layer, nn.Linear)]
        
        g = x
        for step, l_idx in enumerate(linear_indices):
            weights = torch.stack([mlp[l_idx].weight for mlp in mlps])
            biases = torch.stack([mlp[l_idx].bias for mlp in mlps])
            
            w_b = weights[task_idx]
            b_b = biases[task_idx]
            
            g = torch.bmm(w_b, g.unsqueeze(-1)).squeeze(-1) + b_b
            if step < len(linear_indices) - 1:
                g = F.relu(g)
        return g

    def get_value(self, x, task_idx=None):
        task_idx = self._format_task_idx(x, task_idx)
        return self._forward_mlp(x, task_idx, self.critics)

    def get_action_and_value(self, x, action=None, task_idx=None, sample=True):
        task_idx = self._format_task_idx(x, task_idx)
        action_mean = self._forward_mlp(x, task_idx, self.actors)
        value = self._forward_mlp(x, task_idx, self.critics)
        
        if self.is_continuous:
            action_logstd = self.actor_logstds[task_idx].squeeze(1)
            action_std = torch.exp(action_logstd)
            probs = torch.distributions.Normal(action_mean, action_std)
            if action is None:
                action = probs.sample() if sample else action_mean
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
        else:
            probs = torch.distributions.Categorical(logits=action_mean)
            if action is None:
                action = probs.sample() if sample else torch.argmax(action_mean, dim=1)
            return action, probs.log_prob(action), probs.entropy(), value

def test_independent():
    print("--- Running Independent Equivalence Test ---")
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(8,))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
    
    B = 16
    num_tasks = 4
    
    old_net = OldIndependentActorCritic(obs_space, action_space, hidden_dims=[32, 32], num_tasks=num_tasks)
    new_net = NewIndependentActorCritic(obs_space, action_space, hidden_dims=[32, 32], num_tasks=num_tasks)
    
    # Load same parameters
    new_net.load_state_dict(old_net.state_dict())
    
    # Random input + tasks
    x = torch.randn(B, 8)
    task_idx = torch.randint(0, num_tasks, (B,))
    action = torch.randn(B, 3)
    
    # 1. Compare Values
    val_old = old_net.get_value(x, task_idx)
    val_new = new_net.get_value(x, task_idx)
    
    val_diff = torch.max(torch.abs(val_old - val_new)).item()
    print(f"Value diff: {val_diff:.12f}")
    assert torch.allclose(val_old, val_new, atol=1e-5)
    print("[OK] Value forward pass is mathematically identical!")
    
    # 2. Compare action distributions
    act_old, lp_old, ent_old, v_old = old_net.get_action_and_value(x, action, task_idx)
    act_new, lp_new, ent_new, v_new = new_net.get_action_and_value(x, action, task_idx)
    
    lp_diff = torch.max(torch.abs(lp_old - lp_new)).item()
    ent_diff = torch.max(torch.abs(ent_old - ent_new)).item()
    
    print(f"LogProb diff: {lp_diff:.12f}, Entropy diff: {ent_diff:.12f}")
    assert torch.allclose(lp_old, lp_new, atol=1e-5)
    assert torch.allclose(ent_old, ent_new, atol=1e-5)
    print("[OK] Action / distribution forward pass is mathematically identical!")
    
    # 3. Compare Gradients
    loss_old = lp_old.sum() + ent_old.sum() + v_old.sum()
    loss_new = lp_new.sum() + ent_new.sum() + v_new.sum()
    
    old_net.zero_grad()
    loss_old.backward()
    grads_old = {name: p.grad.clone() for name, p in old_net.named_parameters() if p.grad is not None}
    
    new_net.zero_grad()
    loss_new.backward()
    grads_new = {name: p.grad.clone() for name, p in new_net.named_parameters() if p.grad is not None}
    
    for name in grads_old:
        grad_diff = torch.max(torch.abs(grads_old[name] - grads_new[name])).item()
        assert torch.allclose(grads_old[name], grads_new[name], atol=1e-5), f"Grad failed for {name} with diff {grad_diff}"
    print("[OK] Independent baseline gradients are mathematically identical!")

if __name__ == "__main__":
    test_softmod()
    print("")
    test_independent()
    print("\nALL EQUIVALENCE TESTS PASSED FLAWLESSLY!")
