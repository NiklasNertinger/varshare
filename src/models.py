
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class VarShareLayer(nn.Module):
    def __init__(self, in_features, out_features, prior_scale=1.0, learned_prior=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_scale = prior_scale
        self.learned_prior = learned_prior
        
        # Shared parameters (theta)
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Task-specific parameters (Weights only)
        self.mus = nn.ParameterDict()
        self.rhos = nn.ParameterDict()
        
        # Learned Prior (Empirical Bayes)
        if self.learned_prior:
            # Initialize to log(0.1) -> sigma=0.1
            self.log_sigma_prior = nn.Parameter(torch.tensor(math.log(0.1)))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def add_task(self, task_id, mu_init=0.0, rho_init=-5.0, **kwargs):
        t_key = str(task_id)
        # Weights
        self.mus[t_key] = nn.Parameter(torch.zeros_like(self.theta) + mu_init)
        self.rhos[t_key] = nn.Parameter(torch.ones_like(self.theta) * rho_init)

    def forward(self, x, task_id, sample=True):
        if isinstance(task_id, torch.Tensor):
            t_id = task_id.flatten()[0].item()
        else:
            t_id = task_id

        task_key = str(t_id)
        
        # Fallback if task not registered (or implement strict check)
        if task_key not in self.mus:
            return F.linear(x, self.theta, self.bias)

        # Weights
        theta = self.theta
        mu = self.mus[task_key]
        rho = self.rhos[task_key]
        sigma = F.softplus(rho)
        weight_mean = theta + mu
        
        if not self.training or not sample:
            return F.linear(x, weight_mean, self.bias)

        # Reparameterization (Local Reparam)
        # Mean output
        out_mean = F.linear(x, weight_mean, self.bias)
        
        # Variance output
        # Var[y] = x^2 * sigma_w^2
        delta_y_var = F.linear(x.pow(2), sigma.pow(2))
        delta_y_std = torch.sqrt(delta_y_var + 1e-8)
        
        epsilon = torch.randn_like(out_mean)
        return out_mean + delta_y_std * epsilon

    def kl_divergence(self, task_id):
        if isinstance(task_id, torch.Tensor):
            t_id = task_id.flatten()[0].item()
        else:
            t_id = task_id
            
        task_key = str(t_id)
        if task_key not in self.mus: return torch.tensor(0.0, device=self.theta.device)
        
        # KL for Weights
        mu = self.mus[task_key]
        rho = self.rhos[task_key]
        sigma = F.softplus(rho)
        
        var_q = sigma.pow(2)
        # Determine Prior Variance
        if self.learned_prior:
             # Ensure sigma_prior is positive
             var_p = torch.exp(self.log_sigma_prior).pow(2)
        else:
             var_p = self.prior_scale ** 2
        
        kl_w = 0.5 * (math.log(var_p) - torch.log(var_q + 1e-8) + (var_q + mu.pow(2)) / var_p - 1)
        
        return kl_w.sum()

    def get_architectural_metrics(self, task_id):
        task_key = str(task_id)
        if task_key not in self.mus:
            return None
        
        theta = self.theta
        mu = self.mus[task_key]
        rho = self.rhos[task_key]
        sigma = F.softplus(rho)
        
        norm_theta = torch.norm(theta).item()
        norm_mu = torch.norm(mu).item()
        avg_sigma = torch.mean(sigma).item()
        
        return {
            "norm_theta": norm_theta,
            "norm_mu": norm_mu,
            "avg_sigma": avg_sigma,
            "sharing_ratio": norm_theta / (norm_theta + norm_mu + 1e-8)
        }

class VarShareLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, prior_scale=1.0, learned_prior=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.prior_scale = prior_scale
        self.learned_prior = learned_prior
        
        # Shared parameters (theta)
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Task-specific parameters (LoRA Factors)
        # B @ A
        # A: [rank, in_features]
        # B: [out_features, rank]
        
        self.mus_A = nn.ParameterDict()
        self.rhos_A = nn.ParameterDict()
        
        self.mus_B = nn.ParameterDict()
        self.rhos_B = nn.ParameterDict()
        
        # Learned Prior
        if self.learned_prior:
            self.log_sigma_prior = nn.Parameter(torch.tensor(math.log(0.1)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def add_task(self, task_id, mu_init=0.0, rho_init=-5.0, **kwargs):
        t_key = str(task_id)
        
        # Initialize A: small random to break symmetry? Or zero?
        # Usually LoRA inits A random, B zero.
        # Here we are variational.
        # mu_A ~ N(0, 0.01)
        self.mus_A[t_key] = nn.Parameter(torch.randn(self.rank, self.in_features) * 0.01)
        self.rhos_A[t_key] = nn.Parameter(torch.ones(self.rank, self.in_features) * rho_init)
        
        # mu_B = 0
        self.mus_B[t_key] = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.rhos_B[t_key] = nn.Parameter(torch.ones(self.out_features, self.rank) * rho_init)

    def forward(self, x, task_id, sample=True):
        if isinstance(task_id, torch.Tensor):
            t_id = task_id.flatten()[0].item()
        else:
            t_id = task_id

        task_key = str(t_id)
        if task_key not in self.mus_A:
            return F.linear(x, self.theta, self.bias)

        # Get Params
        mu_A = self.mus_A[task_key]
        rho_A = self.rhos_A[task_key]
        sigma_A = F.softplus(rho_A)
        
        mu_B = self.mus_B[task_key]
        rho_B = self.rhos_B[task_key]
        sigma_B = F.softplus(rho_B)
        
        # Sample or Mean
        if self.training and sample:
            A = Normal(mu_A, sigma_A).rsample()
            B = Normal(mu_B, sigma_B).rsample()
        else:
            A = mu_A
            B = mu_B
            
        # Effective Weight
        # residual = B @ A -> [out, rank] @ [rank, in] -> [out, in]
        residual = B @ A
        weight = self.theta + residual
        
        return F.linear(x, weight, self.bias)

    def kl_divergence(self, task_id):
        if isinstance(task_id, torch.Tensor):
            t_id = task_id.flatten()[0].item()
        else:
            t_id = task_id
            
        task_key = str(t_id)
        if task_key not in self.mus_A: return torch.tensor(0.0, device=self.theta.device)
        
        # KL = KL(A) + KL(B)
        # A
        mu_A = self.mus_A[task_key]
        rho_A = self.rhos_A[task_key]
        sigma_A = F.softplus(rho_A)
        var_q_A = sigma_A.pow(2)
        
        if self.learned_prior:
            var_p = torch.exp(self.log_sigma_prior).pow(2)
        else:
            var_p = self.prior_scale ** 2
            
        kl_A = 0.5 * (math.log(var_p) - torch.log(var_q_A + 1e-8) + (var_q_A + mu_A.pow(2)) / var_p - 1)
        
        # B
        mu_B = self.mus_B[task_key]
        rho_B = self.rhos_B[task_key]
        sigma_B = F.softplus(rho_B)
        var_q_B = sigma_B.pow(2)
        kl_B = 0.5 * (math.log(var_p) - torch.log(var_q_B + 1e-8) + (var_q_B + mu_B.pow(2)) / var_p - 1)
        
        return kl_A.sum() + kl_B.sum()

    def get_architectural_metrics(self, task_id):
        task_key = str(task_id)
        if task_key not in self.mus_A: return None
        
        with torch.no_grad():
            mu_A = self.mus_A[task_key]
            mu_B = self.mus_B[task_key]
            residual = mu_B @ mu_A
            
            norm_theta = torch.norm(self.theta).item()
            norm_res = torch.norm(residual).item()
            
            return {
                "norm_theta": norm_theta,
                "norm_mu": norm_res,
                "sharing_ratio": norm_theta / (norm_theta + norm_res + 1e-8),
                "avg_sigma": 0.0 # Simplify
            }

class VarShareNetwork(nn.Module):
    def __init__(self, input_dim, output_dims, hidden_dims=[64, 64], num_tasks=1, prior_scale=1.0, 
                 rho_init=-5.0, mu_init=0.0, variant="standard", rank=4, learned_prior=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_tasks = num_tasks
        
        prev_dim = input_dim
        total_layers = len(hidden_dims)
        
        for i, h_dim in enumerate(hidden_dims):
            # Check for "partial" variant
            # Last 2 layers = VarShare. 
            # Note: This loops over HIDDEN layers only. 
            # If hidden_dims=[64, 64], we have Layer 0 (Input->64) and Layer 1 (64->64).
            # Then usually there is a Head layer.
            # "Last 2 layers" usually means the output head + last hidden layer?
            # Or last 2 hidden?
            # User said "Last 2 layers". Let's assume (Last Hidden + Head).
            # So here: if we have 2 hidden layers, maybe only the second one is VarShare?
            # Let's interpret "Partial" as: Only use VarShare for the DEEPEST layers.
            # If variant="partial", we use Standard Linear for early layers.
            
            use_varshare_here = True
            if variant == "partial":
                # Let's say we only make the very last hidden layer VarShare
                # And the Head (handled later) will also be VarShare.
                # So indices < total_layers - 1 are Standard.
                if i < total_layers - 1:
                    use_varshare_here = False
            
            if use_varshare_here:
                if variant == "lora":
                    layer = VarShareLoRALayer(prev_dim, h_dim, rank=rank, prior_scale=prior_scale, learned_prior=learned_prior)
                else: 
                    # Standard or Reptile (same structure) or Scaled (handled by caller passing dims)
                    layer = VarShareLayer(prev_dim, h_dim, prior_scale=prior_scale, learned_prior=learned_prior)
                    
                for t in range(num_tasks):
                    layer.add_task(t, mu_init, rho_init)
            else:
                # Standard Linear Layer (Shared)
                layer = layer_init(nn.Linear(prev_dim, h_dim))
                
            self.layers.append(layer)
            prev_dim = h_dim
            
        # We assume output_dims is a list of [actor_dim, critic_dim=1] or similar?
        # Actually VarShareNetwork is usually the "Backbone".
        # But for full VarShare, even the head can be variational?
        # User prompt doesn't specify deep details, but reference implementation had separate output layer.
        # We will expose `forward_features` and let heads be separate or part of it?
        # Reference `VarShareNetwork` had an output layer.
        # Let's make this versatile.
        
        self.output_dim = prev_dim 
        
    def forward(self, x, task_id, sample=True):
        for layer in self.layers:
            # Check if layer supports task_id (VarShare/LoRA)
            if hasattr(layer, "add_task"):
                x = F.relu(layer(x, task_id, sample=sample))
            else:
                # Standard nn.Linear
                x = F.relu(layer(x))
        return x

    def get_kl(self, task_id):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, "kl_divergence"):
                 kl += layer.kl_divergence(task_id)
        return kl

    def get_architectural_metrics(self, task_id):
        metrics = []
        for layer in self.layers:
            if hasattr(layer, "get_architectural_metrics"):
                m = layer.get_architectural_metrics(task_id)
                if m: metrics.append(m)
            else:
                # Standard Layer Metrics (approximate for aggregation)
                # sharing_ratio = 1.0 (all shared)
                # norm_theta = actual norm
                # norm_mu = 0
                if hasattr(layer, "weight"):
                     metrics.append({
                         "norm_theta": torch.norm(layer.weight).item(),
                         "norm_mu": 0.0,
                         "avg_sigma": 0.0,
                         "sharing_ratio": 1.0
                     })
        
        if not metrics: return {}
        
        # Aggregate across layers
        avg_metrics = {
            "mean_norm_theta": np.mean([m["norm_theta"] for m in metrics]),
            "mean_norm_mu": np.mean([m["norm_mu"] for m in metrics]),
            "sharing_ratio": np.mean([m["sharing_ratio"] for m in metrics]),
            "avg_sigma": np.mean([m["avg_sigma"] for m in metrics])
        }
        return avg_metrics

    def get_task_similarity(self):
        all_similarities = []
        for layer in self.layers:
            if not hasattr(layer, "mus"): continue # Skip standard
            
            task_ids = sorted([int(k) for k in layer.mus.keys()])
            if len(task_ids) < 2: continue
            
            mus = [layer.mus[str(tid)].flatten() for tid in task_ids]
            mus_tensor = torch.stack(mus)
            mus_norm = F.normalize(mus_tensor, p=2, dim=1)
            sim_mat = torch.mm(mus_norm, mus_norm.t())
            
            triu_indices = torch.triu_indices(len(task_ids), len(task_ids), offset=1)
            layer_sims = sim_mat[triu_indices[0], triu_indices[1]]
            all_similarities.append(layer_sims.mean().item())
            
        return np.mean(all_similarities) if all_similarities else 1.0




class ActorCritic(nn.Module):
    """
    Standard Actor-Critic network for PPO.
    Supports optional Task Embeddings for Multi-Task Learning.
    """
    def __init__(self, observation_space, action_space, hidden_dim=64, 
                 use_task_embedding=False, embedding_dim=10, num_tasks=1,
                 use_varshare=False, varshare_args={}):
        super().__init__()
        self.use_task_embedding = use_task_embedding
        self.use_varshare = use_varshare
        self.num_tasks = num_tasks
        
        obs_shape = int(np.array(observation_space.shape).prod())
        
        # Check Action Space Type
        if hasattr(action_space, 'n'):
            self.is_continuous = False
            self.action_dim = int(action_space.n)
        else:
            self.is_continuous = True
            self.action_dim = int(np.array(action_space.shape).prod())
        
        input_dim = obs_shape
        if use_task_embedding and not use_varshare:
            self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
            input_dim += embedding_dim
            
        if self.use_varshare:
            # Separate Backbones (Golden Reference)
            self.actor_backbone = VarShareNetwork(
                input_dim=input_dim, 
                output_dims=None,
                hidden_dims=[hidden_dim, hidden_dim],
                num_tasks=num_tasks,
                **varshare_args
            )
            
            self.critic_backbone = VarShareNetwork(
                input_dim=input_dim, 
                output_dims=None,
                hidden_dims=[hidden_dim, hidden_dim],
                num_tasks=num_tasks,
                **varshare_args
            )
            
            feature_dim = hidden_dim 
            
            # Heads are also VarShareLayers (Reference: "every linear layer")
            # If variant=partial, head IS VarShare.
            # If variant=lora, head IS LoRA.
            
            variant = varshare_args.get("variant", "standard")
            rank = varshare_args.get("rank", 4)
            # Remove from kwargs to avoid dup args if passing separately
            # But we pass **varshare_args to VarShareNetwork init, so all good.
            
            
            if variant == "lora":
                 self.critic_head = VarShareLoRALayer(feature_dim, 1, rank=rank, prior_scale=1.0, learned_prior=varshare_args.get("learned_prior", False))
                 self.actor_head = VarShareLoRALayer(feature_dim, self.action_dim, rank=rank, prior_scale=1.0, learned_prior=varshare_args.get("learned_prior", False))
            else:
                 # Standard, Reptile, Partial (Head is always VS in Partial)
                 self.critic_head = VarShareLayer(feature_dim, 1, prior_scale=1.0, learned_prior=varshare_args.get("learned_prior", False))
                 self.actor_head = VarShareLayer(feature_dim, self.action_dim, prior_scale=1.0, learned_prior=varshare_args.get("learned_prior", False))
            
            for t in range(num_tasks):
                self.critic_head.add_task(t, **varshare_args) # Re-use args like rho_init
                self.actor_head.add_task(t, **varshare_args)
            
        else:
            # Standard MLP
            self.critic = nn.Sequential(
                layer_init(nn.Linear(input_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, 1), std=1.0),
            )
            
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(input_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, self.action_dim), std=0.01),
            )
        
        if self.is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def _get_input(self, x, task_idx):
        if self.use_task_embedding and not self.use_varshare:
            if task_idx is None:
                raise ValueError("task_idx required for task embedding model")
            
            # Ensure task_idx matches batch dimensions
            if isinstance(task_idx, int):
                task_idx = torch.tensor([task_idx], device=x.device).expand(x.shape[0])
            elif isinstance(task_idx, torch.Tensor):
                 if task_idx.dim() == 0:
                      task_idx = task_idx.expand(x.shape[0])
                 elif len(task_idx) != x.shape[0]:
                      if len(task_idx) == 1:
                           task_idx = task_idx.expand(x.shape[0])
            
            embeds = self.task_embedding(task_idx.long())
            return torch.cat([x, embeds], dim=1)
        return x

    def get_features(self, x, task_idx=None):
        if self.use_varshare:
            # Return actor features by default
            return self.actor_backbone(x, task_idx)
        else:
            # Not supported/needed for standard path yet
            return x

    def get_value(self, x, task_idx=None):
        x_in = self._get_input(x, task_idx)
        if self.use_varshare:
            # Critic uses its own backbone
            features = self.critic_backbone(x_in, task_idx)
            # Pass task_id to critic head
            return self.critic_head(features, task_idx)
        return self.critic(x_in)

    def get_action_and_value(self, x, action=None, task_idx=None, sample=True):
        x_in = self._get_input(x, task_idx)
        
        if self.use_varshare:
            # Actor Path
            actor_features = self.actor_backbone(x_in, task_idx, sample=sample)
            
            # Critic Path (Separate)
            critic_features = self.critic_backbone(x_in, task_idx, sample=sample)
            
            value = self.critic_head(critic_features, task_idx, sample=sample)
            
            if self.is_continuous:
                # Pass task_id to actor head
                action_mean = self.actor_head(actor_features, task_idx, sample=sample)
                action_logstd = self.actor_logstd.expand_as(action_mean)
                action_std = torch.exp(action_logstd)
                probs = torch.distributions.Normal(action_mean, action_std)
                
                if not sample: # Selection for deterministic eval
                     action = action_mean
                elif action is None: 
                     action = probs.sample()
                     
                return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
            else:
                # Pass task_id to actor head
                logits = self.actor_head(actor_features, task_idx, sample=sample)
                probs = torch.distributions.Categorical(logits=logits)
                
                if not sample:
                     action = torch.argmax(logits, dim=1)
                elif action is None: 
                     action = probs.sample()
                     
                return action, probs.log_prob(action), probs.entropy(), value

        else:
            if self.is_continuous:
                action_mean = self.actor_mean(x_in)
                action_logstd = self.actor_logstd.expand_as(action_mean)
                action_std = torch.exp(action_logstd)
                
                probs = torch.distributions.Normal(action_mean, action_std)
                if action is None:
                    action = probs.sample()
                return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x_in)
            else:
                logits = self.actor_mean(x_in)
                probs = torch.distributions.Categorical(logits=logits)
                if action is None:
                    action = probs.sample()
                return action, probs.log_prob(action), probs.entropy(), self.critic(x_in)
    
    def get_kl(self, task_idx):
        if self.use_varshare:
            kl = self.actor_backbone.get_kl(task_idx)
            kl += self.critic_backbone.get_kl(task_idx)
            kl += self.actor_head.kl_divergence(task_idx)
            kl += self.critic_head.kl_divergence(task_idx)
            return kl
        return 0.0

class PaCoActorCritic(nn.Module):
    """
    Parameter Composition (PaCo) Actor-Critic.
    Instead of task embeddings, it uses multiple shared 'experts' and learns
    task-specific composition weights.
    """
    def __init__(self, observation_space, action_space, hidden_dim=64, num_tasks=1, num_experts=4):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        
        obs_shape = int(np.array(observation_space.shape).prod())
        
        if hasattr(action_space, 'n'):
            self.is_continuous = False
            self.action_dim = int(action_space.n)
        else:
            self.is_continuous = True
            self.action_dim = int(np.array(action_space.shape).prod())

        # Experts (Shared)
        self.actor_experts = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(obs_shape, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, self.action_dim), std=0.01),
            ) for _ in range(num_experts)
        ])
        
        self.critic_experts = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(obs_shape, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, 1), std=1.0),
            ) for _ in range(num_experts)
        ])
        
        # Composition Weights (Task-Specific)
        # We use logits and softmax to ensure weights sum to 1
        self.actor_weights = nn.Parameter(torch.zeros(num_tasks, num_experts))
        self.critic_weights = nn.Parameter(torch.zeros(num_tasks, num_experts))

        if self.is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def get_value(self, x, task_idx=None):
        if task_idx is None:
            # Fallback to expert 0 if no task_idx (should not happen in MTL)
            return self.critic_experts[0](x)
            
        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=x.device).expand(x.shape[0])
        elif isinstance(task_idx, torch.Tensor):
            if task_idx.dim() == 0:
                task_idx = task_idx.expand(x.shape[0])
            elif len(task_idx) != x.shape[0]:
                 # Handle mismatch if only one task_idx provided for batch
                 if len(task_idx) == 1:
                    task_idx = task_idx.expand(x.shape[0])
        
        # Get expert outputs: [batch, num_experts, 1]
        expert_outputs = torch.stack([expert(x) for expert in self.critic_experts], dim=1)
        
        # Get softmax weights: [batch, num_experts, 1]
        weights = F.softmax(self.actor_weights[task_idx.long()], dim=1).unsqueeze(-1)
        
        # Weighted sum
        return (expert_outputs * weights).sum(dim=1)

    def get_action_and_value(self, x, action=None, task_idx=None, sample=True):
        if task_idx is None:
            raise ValueError("task_idx required for PaCoActorCritic")
            
        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=x.device).expand(x.shape[0])
        elif isinstance(task_idx, torch.Tensor):
            if task_idx.dim() == 0:
                task_idx = task_idx.expand(x.shape[0])
            elif len(task_idx) != x.shape[0]:
                 if len(task_idx) == 1:
                    task_idx = task_idx.expand(x.shape[0])

        # Critic Path
        critic_expert_outputs = torch.stack([expert(x) for expert in self.critic_experts], dim=1)
        # Using a separate critic weight matrix for flexibility, though some papers share weights
        c_weights = F.softmax(self.critic_weights[task_idx.long()], dim=1).unsqueeze(-1)
        value = (critic_expert_outputs * c_weights).sum(dim=1)

        # Actor Path
        actor_expert_outputs = torch.stack([expert(x) for expert in self.actor_experts], dim=1)
        a_weights = F.softmax(self.actor_weights[task_idx.long()], dim=1).unsqueeze(-1)
        actor_output = (actor_expert_outputs * a_weights).sum(dim=1)

        if self.is_continuous:
            action_mean = actor_output
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = torch.distributions.Normal(action_mean, action_std)
            
            if not sample: # selection for deterministic eval
                action = action_mean
            elif action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
        else:
            logits = actor_output
            probs = torch.distributions.Categorical(logits=logits)
            if not sample:
                action = torch.argmax(logits, dim=1)
            elif action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), value
            
    def get_architectural_metrics(self, task_idx):
        # Return entropy of the static composition weights for this task
        with torch.no_grad():
            if isinstance(task_idx, torch.Tensor): task_idx = task_idx.item()
            
            # Actor weights
            logits_a = self.actor_weights[task_idx]
            probs_a = F.softmax(logits_a, dim=0)
            entropy_a = -torch.sum(probs_a * torch.log(probs_a + 1e-8)).item()
            
            # Critic weights
            logits_c = self.critic_weights[task_idx]
            probs_c = F.softmax(logits_c, dim=0)
            entropy_c = -torch.sum(probs_c * torch.log(probs_c + 1e-8)).item()
            
            return {
                "weight_entropy_actor": entropy_a,
                "weight_entropy_critic": entropy_c,
                "max_weight_actor": probs_a.max().item()
            }

    def get_kl(self, task_idx):
        return torch.tensor(0.0, device=self.actor_weights.device)

class SoftModularActorCritic(nn.Module):
    """
    Soft Modularization Actor-Critic.
    Features a base network of modules and a routing network that computes
    soft selection weights based on state and task identity.
    """
    def __init__(self, observation_space, action_space, hidden_dim=64, num_tasks=1, 
                 num_modules=4, num_layers=2):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_modules = num_modules
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim # D in the paper
        
        obs_shape = int(np.array(observation_space.shape).prod())
        
        if hasattr(action_space, 'n'):
            self.is_continuous = False
            self.action_dim = int(action_space.n)
        else:
            self.is_continuous = True
            self.action_dim = int(np.array(action_space.shape).prod())

        # Encoder f(s)
        self.obs_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_shape, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh()
        )
        
        # Task Embedding h(z)
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)

        # Build Base Modules
        # We need separate actor and critic networks for stability in PPO
        self.actor_base = self._build_base(hidden_dim, self.action_dim)
        self.critic_base = self._build_base(hidden_dim, 1)

        # Build Routing Networks
        self.actor_routing_d = nn.ModuleList([
            layer_init(nn.Linear(hidden_dim, num_modules * num_modules))
            for _ in range(num_layers - 1)
        ])
        
        self.critic_routing_d = nn.ModuleList([
            layer_init(nn.Linear(hidden_dim, num_modules * num_modules))
            for _ in range(num_layers - 1)
        ])
        
        if self.is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def _build_base(self, hidden_dim, final_dim):
        layers = nn.ModuleList()
        # Layer 1 to L-1
        for l in range(self.num_layers - 1):
            modules = nn.ModuleList([
                layer_init(nn.Linear(hidden_dim, hidden_dim))
                for _ in range(self.num_modules)
            ])
            layers.append(modules)
        # Final Layer modules output the target dim
        final_modules = nn.ModuleList([
            layer_init(nn.Linear(hidden_dim, final_dim), std=0.01 if final_dim > 1 else 1.0)
            for _ in range(self.num_modules)
        ])
        layers.append(final_modules)
        return layers

    def _get_routing(self, f_s, h_z, routing_d_list):
        # f_s: [B, D], h_z: [B, D]
        context = f_s * h_z
        probs_list = []
        for d_layer in routing_d_list:
            logits = d_layer(F.relu(context)) # [B, n*n]
            # Softmax over source modules (j dimension)
            # Reshape to [B, n_out, n_in]
            logits = logits.view(-1, self.num_modules, self.num_modules)
            probs = F.softmax(logits, dim=2)
            probs_list.append(probs)
        return probs_list

    def get_value(self, x, task_idx=None):
        if task_idx is None:
            raise ValueError("task_idx required for SoftModularActorCritic")
        
        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=x.device).expand(x.shape[0])
        elif isinstance(task_idx, torch.Tensor):
            if task_idx.dim() == 0:
                task_idx = task_idx.expand(x.shape[0])
            elif len(task_idx) != x.shape[0]:
                 if len(task_idx) == 1:
                    task_idx = task_idx.expand(x.shape[0])

        f_s = self.obs_encoder(x)
        h_z = self.task_embedding(task_idx.long())
        
        # Critic routing
        probs = self._get_routing(f_s, h_z, self.critic_routing_d)
        
        # Base forward
        # Initial input: [B, n, D] - all modules get same f_s
        g = f_s.unsqueeze(1).repeat(1, self.num_modules, 1)
        
        for l in range(self.num_layers - 1):
            # Compute module outputs: [B, n, D]
            layer_outputs = []
            for j in range(self.num_modules):
                layer_outputs.append(F.relu(self.critic_base[l][j](g[:, j, :])))
            layer_outputs = torch.stack(layer_outputs, dim=2) # [B, D, n_in]
            
            # Apply routing: g_i = sum_j p_ij * out_j
            # p: [B, n_out, n_in], layer_outputs: [B, D, n_in]
            p = probs[l]
            g = torch.bmm(p, layer_outputs.transpose(1, 2)) # [B, n_out, D]

        # Final Layer sum
        final_outs = []
        for j in range(self.num_modules):
            final_outs.append(self.critic_base[-1][j](g[:, j, :]))
        return torch.stack(final_outs, dim=1).sum(dim=1)

    def get_action_and_value(self, x, action=None, task_idx=None, sample=True):
        if task_idx is None:
            raise ValueError("task_idx required for SoftModularActorCritic")

        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=x.device).expand(x.shape[0])
        elif isinstance(task_idx, torch.Tensor):
            if task_idx.dim() == 0:
                task_idx = task_idx.expand(x.shape[0])
            elif len(task_idx) != x.shape[0]:
                 if len(task_idx) == 1:
                    task_idx = task_idx.expand(x.shape[0])

        f_s = self.obs_encoder(x)
        h_z = self.task_embedding(task_idx.long())
        
        # 1. Critic Path
        c_probs = self._get_routing(f_s, h_z, self.critic_routing_d)
        g_c = f_s.unsqueeze(1).repeat(1, self.num_modules, 1)
        for l in range(self.num_layers - 1):
            outs = torch.stack([F.relu(mod(g_c[:, j, :])) for j, mod in enumerate(self.critic_base[l])], dim=2)
            g_c = torch.bmm(c_probs[l], outs.transpose(1, 2))
        value = torch.stack([mod(g_c[:, j, :]) for j, mod in enumerate(self.critic_base[-1])], dim=1).sum(dim=1)

        # 2. Actor Path
        a_probs = self._get_routing(f_s, h_z, self.actor_routing_d)
        g_a = f_s.unsqueeze(1).repeat(1, self.num_modules, 1)
        for l in range(self.num_layers - 1):
            outs = torch.stack([F.relu(mod(g_a[:, j, :])) for j, mod in enumerate(self.actor_base[l])], dim=2)
            g_a = torch.bmm(a_probs[l], outs.transpose(1, 2))
        actor_output = torch.stack([mod(g_a[:, j, :]) for j, mod in enumerate(self.actor_base[-1])], dim=1).sum(dim=1)

        if self.is_continuous:
            action_mean = actor_output
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = torch.distributions.Normal(action_mean, action_std)
            
            if not sample: 
                action = action_mean
            elif action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value
        else:
            logits = actor_output
            probs = torch.distributions.Categorical(logits=logits)
            if not sample:
                action = torch.argmax(logits, dim=1)
            elif action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), value

    def get_architectural_metrics(self, task_idx):
        # Calculate entropy of Softmax weights to check for collapse
        # [num_tasks, num_modules]
        with torch.no_grad():
            f_s = torch.zeros(1, self.hidden_dim, device=self.task_embedding.weight.device) # Dummy
            h_z = self.task_embedding(torch.tensor([task_idx], device=self.task_embedding.weight.device))
            
            # Check Actor Routing Layer 0
            # logits: [1, num_modules * num_modules] -> [1, n_out, n_in]
            routing_layer = self.actor_routing_d[0]
            context = f_s * h_z # Zeros if f_s is dummy? Wait, routing depends on state.
            # Routing depends on state f(s). If we want "general" metric, we need real states.
            # For now, let's return None or just weight norms if state dependent.
            # Actually, SoftMod routing is dynamic per state. Hard to summarize without batch.
            # Let's return None for now, or maybe just router weight norms.
            return None

    def get_kl(self, task_idx):
        return torch.tensor(0.0, device=self.task_embedding.weight.device)
