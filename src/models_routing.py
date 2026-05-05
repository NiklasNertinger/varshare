import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversal.apply(x, self.alpha)

class DetVarShareLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Shared parameters (theta)
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Task-specific parameters (mean shifts only)
        self.mus = nn.ParameterDict()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def add_task(self, task_id, mu_init=0.0, **kwargs):
        t_key = str(task_id)
        # Initialize task-specific shift with Gaussian noise scaled by mu_init
        self.mus[t_key] = nn.Parameter(torch.randn_like(self.theta, device=self.theta.device) * mu_init)

    def forward(self, x, task_id):
        if isinstance(task_id, torch.Tensor) and task_id.numel() > 1:
            out = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
            for t in torch.unique(task_id):
                t_key = str(t.item())
                mask = (task_id == t).flatten()
                if t_key not in self.mus:
                    out[mask] = F.linear(x[mask], self.theta, self.bias)
                else:
                    weight_mean = self.theta + self.mus[t_key]
                    out[mask] = F.linear(x[mask], weight_mean, self.bias)
            return out
        else:
            t_id = task_id.flatten()[0].item() if isinstance(task_id, torch.Tensor) else task_id
            task_key = str(t_id)
            if task_key not in self.mus:
                return F.linear(x, self.theta, self.bias)
            weight_mean = self.theta + self.mus[task_key]
            return F.linear(x, weight_mean, self.bias)

    def get_architectural_metrics(self, task_id):
        task_key = str(task_id)
        if task_key not in self.mus:
            return None
        
        with torch.no_grad():
            theta = self.theta
            mu = self.mus[task_key]
            
            norm_theta = torch.norm(theta).item()
            norm_mu = torch.norm(mu).item()
            
            return {
                "norm_theta": norm_theta,
                "norm_mu": norm_mu,
                "sharing_ratio": norm_theta / (norm_theta + norm_mu + 1e-8)
            }

class DetVarShareLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Shared parameters
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Task-specific LoRA parameters
        self.mus_A = nn.ParameterDict()
        self.mus_B = nn.ParameterDict()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def add_task(self, task_id, mu_init=0.0, **kwargs):
        t_key = str(task_id)
        device = self.theta.device
        # Init A with kaiming, B with zeros -> start identity (with Gaussian noise for A scaled by mu_init)
        self.mus_A[t_key] = nn.Parameter(torch.randn(self.out_features, self.rank, device=device) * mu_init)
        self.mus_B[t_key] = nn.Parameter(torch.zeros(self.rank, self.in_features, device=device))
        nn.init.kaiming_uniform_(self.mus_A[t_key], a=math.sqrt(5))
        nn.init.zeros_(self.mus_B[t_key])

    def forward(self, x, task_id):
        if isinstance(task_id, torch.Tensor) and task_id.numel() > 1:
            out = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
            for t in torch.unique(task_id):
                t_key = str(t.item())
                mask = (task_id == t).flatten()
                if t_key not in self.mus_A:
                    out[mask] = F.linear(x[mask], self.theta, self.bias)
                else:
                    mu_matrix = torch.matmul(self.mus_A[t_key], self.mus_B[t_key])
                    weight_mean = self.theta + mu_matrix
                    out[mask] = F.linear(x[mask], weight_mean, self.bias)
            return out
        else:
            t_id = task_id.flatten()[0].item() if isinstance(task_id, torch.Tensor) else task_id
            task_key = str(t_id)
            if task_key not in self.mus_A:
                return F.linear(x, self.theta, self.bias)
            mu_matrix = torch.matmul(self.mus_A[task_key], self.mus_B[task_key])
            weight_mean = self.theta + mu_matrix
            return F.linear(x, weight_mean, self.bias)

    def get_architectural_metrics(self, task_id):
        task_key = str(task_id)
        if task_key not in self.mus_A:
            return None
        
        with torch.no_grad():
            theta = self.theta
            mu_A = self.mus_A[task_key]
            mu_B = self.mus_B[task_key]
            
            mu_matrix = torch.matmul(mu_A, mu_B)
            
            norm_theta = torch.norm(theta).item()
            norm_mu = torch.norm(mu_matrix).item()
            
            return {
                "norm_theta": norm_theta,
                "norm_mu": norm_mu,
                "sharing_ratio": norm_theta / (norm_theta + norm_mu + 1e-8)
            }

class DetVarShareGatedLayer(nn.Module):
    """
    Soft Gating: Applies a learned, task-specific gate (sigmoid) to the shared theta backbone.
    y = (theta * g_t + mu_t) * x + bias
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.mus = nn.ParameterDict()
        self.gates = nn.ParameterDict()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def add_task(self, task_id, mu_init=0.0, **kwargs):
        t_key = str(task_id)
        device = self.theta.device
        self.mus[t_key] = nn.Parameter(torch.randn_like(self.theta, device=device) * mu_init)
        # Gate init: high positive value so sigmoid(gate) ~ 1.0 (fully shared initially)
        self.gates[t_key] = nn.Parameter(torch.ones_like(self.theta, device=device) * 3.0)

    def forward(self, x, task_id):
        if isinstance(task_id, torch.Tensor) and task_id.numel() > 1:
            out = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
            for t in torch.unique(task_id):
                t_key = str(t.item())
                mask = (task_id == t).flatten()
                if t_key not in self.mus:
                    out[mask] = F.linear(x[mask], self.theta, self.bias)
                else:
                    g_t = torch.sigmoid(self.gates[t_key])
                    weight_mean = (self.theta * g_t) + self.mus[t_key]
                    out[mask] = F.linear(x[mask], weight_mean, self.bias)
            return out
        else:
            t_id = task_id.flatten()[0].item() if isinstance(task_id, torch.Tensor) else task_id
            task_key = str(t_id)
            if task_key not in self.mus:
                return F.linear(x, self.theta, self.bias)
            g_t = torch.sigmoid(self.gates[task_key])
            weight_mean = (self.theta * g_t) + self.mus[task_key]
            return F.linear(x, weight_mean, self.bias)

    def get_architectural_metrics(self, task_id):
        task_key = str(task_id)
        if task_key not in self.mus: return None
        
        with torch.no_grad():
            theta = self.theta
            mu = self.mus[task_key]
            g_t = torch.sigmoid(self.gates[task_key])
            
            norm_theta = torch.norm(theta * g_t).item()
            norm_mu = torch.norm(mu).item()
            gate_sparsity = (g_t < 0.1).float().mean().item()
            
            return {
                "norm_theta": norm_theta,
                "norm_mu": norm_mu,
                "sharing_ratio": norm_theta / (norm_theta + norm_mu + 1e-8),
                "gate_sparsity": gate_sparsity
            }

class DetVarShareFiLMLayer(nn.Module):
    """
    FiLM Modulation: Applies multiplicative and additive shifts to the OUTGOING activations.
    y = gamma_t * (theta * x + bias) + beta_t
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # gamma (multiplicative), beta (additive) -> shape is out_features
        self.gammas = nn.ParameterDict()
        self.betas = nn.ParameterDict()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def add_task(self, task_id, mu_init=0.0, **kwargs):
        t_key = str(task_id)
        device = self.theta.device
        self.gammas[t_key] = nn.Parameter(torch.ones(self.out_features, device=device))  # start at 1
        self.betas[t_key] = nn.Parameter(torch.randn(self.out_features, device=device) * mu_init)

    def forward(self, x, task_id):
        if isinstance(task_id, torch.Tensor) and task_id.numel() > 1:
            out = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
            base_out = F.linear(x, self.theta, self.bias)
            for t in torch.unique(task_id):
                t_key = str(t.item())
                mask = (task_id == t).flatten()
                if t_key not in self.gammas:
                    out[mask] = base_out[mask]
                else:
                    gamma_t = self.gammas[t_key]
                    beta_t = self.betas[t_key]
                    out[mask] = gamma_t * base_out[mask] + beta_t
            return out
        else:
            t_id = task_id.flatten()[0].item() if isinstance(task_id, torch.Tensor) else task_id
            task_key = str(t_id)
            base_out = F.linear(x, self.theta, self.bias)
            if task_key not in self.gammas:
                return base_out
            gamma_t = self.gammas[task_key]
            beta_t = self.betas[task_key]
            return gamma_t * base_out + beta_t

    def get_architectural_metrics(self, task_id):
        task_key = str(task_id)
        if task_key not in self.gammas: return None
        
        with torch.no_grad():
            gamma_t = self.gammas[task_key]
            beta_t = self.betas[task_key]
            
            norm_gamma_shift = torch.norm(gamma_t - 1.0).item()
            norm_beta = torch.norm(beta_t).item()
            
            return {
                "norm_gamma_shift": norm_gamma_shift,
                "norm_mu": norm_beta, # Alias beta to mu for script plotting
            }

class DetVarShareHyperpriorLayer(nn.Module):
    """
    Learned Hyperprior: Each task learns a scalar softplus(beta_t) that multiplies its residual mu_t.
    This dynamically squashes/expands the amount of routing based on tasks needs.
    y = (theta + softplus(beta_t) * mu_t) * x + bias
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.mus = nn.ParameterDict()
        self.betas = nn.ParameterDict()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def add_task(self, task_id, mu_init=0.0, **kwargs):
        t_key = str(task_id)
        device = self.theta.device
        self.mus[t_key] = nn.Parameter(torch.randn_like(self.theta, device=device) * mu_init)
        # Init beta -> softplus(0.5413) ~ 1.0
        self.betas[t_key] = nn.Parameter(torch.tensor([0.5413], device=device))

    def forward(self, x, task_id):
        if isinstance(task_id, torch.Tensor) and task_id.numel() > 1:
            out = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
            for t in torch.unique(task_id):
                t_key = str(t.item())
                mask = (task_id == t).flatten()
                if t_key not in self.mus:
                    out[mask] = F.linear(x[mask], self.theta, self.bias)
                else:
                    scale = F.softplus(self.betas[t_key])
                    weight_mean = self.theta + (scale * self.mus[t_key])
                    out[mask] = F.linear(x[mask], weight_mean, self.bias)
            return out
        else:
            t_id = task_id.flatten()[0].item() if isinstance(task_id, torch.Tensor) else task_id
            task_key = str(t_id)
            if task_key not in self.mus:
                return F.linear(x, self.theta, self.bias)
            scale = F.softplus(self.betas[task_key])
            weight_mean = self.theta + (scale * self.mus[task_key])
            return F.linear(x, weight_mean, self.bias)

    def get_architectural_metrics(self, task_id):
        task_key = str(task_id)
        if task_key not in self.mus: return None
        
        with torch.no_grad():
            theta = self.theta
            mu = self.mus[task_key]
            scale = F.softplus(self.betas[task_key]).item()
            
            norm_theta = torch.norm(theta).item()
            norm_mu = torch.norm(mu * scale).item()
            
            return {
                "norm_theta": norm_theta,
                "norm_mu": norm_mu,
                "sharing_ratio": norm_theta / (norm_theta + norm_mu + 1e-8),
                "hyperprior_scale": scale
            }

class DetVarShareNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, variant="base", lora_rank=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.variant = variant.lower()
        
        if self.variant == "lora":
             LayerClass = DetVarShareLoRALayer
        elif self.variant == "gated":
             LayerClass = DetVarShareGatedLayer
        elif self.variant == "film":
             LayerClass = DetVarShareFiLMLayer
        elif self.variant == "hyperprior":
             LayerClass = DetVarShareHyperpriorLayer
        else: # base, l1, pcgrad, decay, ara
             LayerClass = DetVarShareLayer
        
        curr_in = input_dim
        for i in range(num_layers):
            if self.variant == "lora":
                layer = LayerClass(curr_in, hidden_dim, rank=lora_rank)
            else:
                layer = LayerClass(curr_in, hidden_dim)
            self.layers.append(layer)
            curr_in = hidden_dim

    def add_task(self, task_id, **kwargs):
        for layer in self.layers:
            layer.add_task(task_id, **kwargs)

    def forward(self, x, task_idx):
        for layer in self.layers:
            x = layer(x, task_idx)
            x = F.tanh(x)
        return x

    def get_metrics(self, task_idx):
        result = {}
        for i, layer in enumerate(self.layers):
            stats = layer.get_architectural_metrics(task_idx)
            if stats:
                for k, v in stats.items():
                    result[f"layer{i}_{k}"] = v
        return result

    def get_global_metrics(self):
        result = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "mus") and len(layer.mus) > 0:
                with torch.no_grad():
                    all_mus = torch.stack(list(layer.mus.values()))
                    mu_mean = all_mus.mean(dim=0)
                    adapter_variance = torch.mean(torch.norm(all_mus - mu_mean, dim=1)**2).item()
                    adapter_sparsity = (torch.abs(all_mus) < 1e-3).float().mean().item()
                    
                    result[f"layer{i}_adapter_variance"] = adapter_variance
                    result[f"layer{i}_adapter_sparsity"] = adapter_sparsity
        return result

class DetActorCritic(nn.Module):
    def __init__(
        self, 
        envs, 
        hidden_dim=256,
        num_layers=2,
        variant="base",
        lora_rank=4,
        is_continuous=False,
        use_task_embedding=False
    ):
        super().__init__()
        self.is_continuous = is_continuous
        self.use_task_embedding = use_task_embedding
        
        # Determine dimensions
        obs_dim = envs.single_observation_space.shape[0]
        if self.use_task_embedding:
            # Task embeddings allowed but normally we just rely on the VarShare layers
            num_tasks = envs.num_envs
            self.task_embedding = nn.Embedding(num_tasks, 16)
            input_dim = obs_dim + 16
        else:
            input_dim = obs_dim

        if self.is_continuous:
            self.action_dim = np.prod(envs.single_action_space.shape)
        else:
            self.action_dim = envs.single_action_space.n

        # Independent Backbones for Actor and Critic
        self.actor_backbone = DetVarShareNetwork(input_dim, hidden_dim, num_layers, variant, lora_rank)
        self.critic_backbone = DetVarShareNetwork(input_dim, hidden_dim, num_layers, variant, lora_rank)
        
        if variant == "lora":
             LayerClass = DetVarShareLoRALayer
        elif variant == "gated":
             LayerClass = DetVarShareGatedLayer
        elif variant == "film":
             LayerClass = DetVarShareFiLMLayer
        elif variant == "hyperprior":
             LayerClass = DetVarShareHyperpriorLayer
        else:
             LayerClass = DetVarShareLayer

        if variant == "lora":
            self.actor_head = LayerClass(hidden_dim, self.action_dim, rank=lora_rank)
            self.critic_head = LayerClass(hidden_dim, 1, rank=lora_rank)
        else:
            self.actor_head = LayerClass(hidden_dim, self.action_dim)
            self.critic_head = LayerClass(hidden_dim, 1)

        # ARA (Adversarial Representation Alignment) Components
        # We need to access the num_tasks dynamically, but in this setup num_tasks == envs.num_envs
        if variant == "ara":
             num_tasks = envs.num_envs
             self.ara_discriminator = nn.Sequential(
                 GradientReversalLayer(alpha=1.0),
                 nn.Linear(hidden_dim, hidden_dim // 2),
                 nn.ReLU(),
                 nn.Linear(hidden_dim // 2, num_tasks)
             )

        if self.is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def _get_input(self, x, task_idx):
        if self.use_task_embedding:
            if isinstance(task_idx, int):
                task_idx = torch.tensor([task_idx], device=x.device).expand(x.shape[0])
            elif getattr(task_idx, 'dim', lambda: 0)() == 0:
                task_idx = task_idx.expand(x.shape[0])
            embeds = self.task_embedding(task_idx.long())
            return torch.cat([x, embeds], dim=1)
        return x

    def get_value(self, x, task_idx=None):
        x_in = self._get_input(x, task_idx)
        features = self.critic_backbone(x_in, task_idx)
        return self.critic_head(features, task_idx)

    def get_action_and_value(self, x, action=None, task_idx=None, deterministic=False):
        x_in = self._get_input(x, task_idx)
        
        # Actor Path
        actor_features = self.actor_backbone(x_in, task_idx)
        action_mean = self.actor_head(actor_features, task_idx)
        
        # Critic Path
        critic_features = self.critic_backbone(x_in, task_idx)
        value = self.critic_head(critic_features, task_idx)
        
        # ARA Discriminator Path
        ara_logits = None
        if hasattr(self, 'ara_discriminator'):
            # Combine actor and critic features for the discriminator
            # (or just use actor features as the primary representation)
            ara_logits = self.ara_discriminator(actor_features)

        if self.is_continuous:
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = torch.distributions.Normal(action_mean, action_std)
            
            if deterministic:
                 action = action_mean
            elif action is None: 
                 action = probs.sample()
                 
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value, ara_logits
        else:
            probs = torch.distributions.Categorical(logits=action_mean)
            if deterministic:
                 action = torch.argmax(action_mean, dim=1)
            elif action is None: 
                 action = probs.sample()
                 
            return action, probs.log_prob(action), probs.entropy(), value, ara_logits
