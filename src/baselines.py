import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# =========================================================================================
# PaCo: Parameter Composition
# Ported directly from MTBench (RLC 2025) PyTorch Implementation
# =========================================================================================

class PaCoCompositionalLayer(nn.Module):
    def __init__(self, K: int, in_features: int, out_features: int, activation=F.relu, gain=None):
        super().__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self._activation = activation
        
        if gain is None:
            gain = nn.init.calculate_gain('relu') if activation == F.relu else 1.0
            
        V_hat = torch.empty(self.K, out_features, in_features)
        for i in range(self.K):
            nn.init.orthogonal_(V_hat[i], gain=gain)
        self.V_hat = nn.Parameter(V_hat)
        
        self.b_hat = nn.Parameter(torch.zeros(self.K, out_features))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x_exp = x.unsqueeze(0).expand(self.K, *x.shape)
        y = torch.baddbmm(self.b_hat.unsqueeze(1), x_exp, self.V_hat.transpose(1, 2))
        y = y.transpose(0, 1)
        y = torch.bmm(w.unsqueeze(1), y).squeeze(1)
        if self._activation is not None:
            y = self._activation(y)
        return y

class PaCoTaskEncoder(nn.Module):
    def __init__(self, num_tasks: int, K: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, K)
        # Initialize orthogonally
        nn.init.orthogonal_(self.embedding.weight, gain=1.0)

    def forward(self, task_idx: torch.Tensor) -> torch.Tensor:
        # task_idx: (B,)
        w = self.embedding(task_idx)
        return F.softmax(w, dim=-1)

class PaCoBackbone(nn.Module):
    def __init__(self, K: int, in_features: int, hidden_dims: list):
        super().__init__()
        self.layers = nn.ModuleList()
        
        curr_in = in_features
        for h_dim in hidden_dims:
            self.layers.append(PaCoCompositionalLayer(K, curr_in, h_dim, activation=F.relu))
            curr_in = h_dim
            
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, w)
        return x

class PaCoActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dims=[400, 400, 400], num_tasks=10, num_experts=5):
        super().__init__()
        
        obs_shape = int(np.array(observation_space.shape).prod())
        self.action_dim = int(np.array(action_space.shape).prod())
        
        # Task Encoder assigns K continuous weights per task
        self.task_encoder = PaCoTaskEncoder(num_tasks, K=num_experts)
        
        # Decoupled Backbones
        self.actor_backbone = PaCoBackbone(K=num_experts, in_features=obs_shape, hidden_dims=hidden_dims)
        self.critic_backbone = PaCoBackbone(K=num_experts, in_features=obs_shape, hidden_dims=hidden_dims)
        
        # Heads (Compositional as per MTBench)
        self.actor_head = PaCoCompositionalLayer(K=num_experts, in_features=hidden_dims[-1], out_features=self.action_dim, activation=None, gain=0.01)
        self.critic_head = PaCoCompositionalLayer(K=num_experts, in_features=hidden_dims[-1], out_features=1, activation=None, gain=1.0)
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def _format_task_idx(self, x, task_idx):
        if task_idx is None:
            raise ValueError("PaCo requires task_idx")
        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=x.device).expand(x.shape[0])
        elif isinstance(task_idx, torch.Tensor):
            if task_idx.dim() == 0:
                task_idx = task_idx.expand(x.shape[0])
            elif len(task_idx) != x.shape[0]:
                if len(task_idx) == 1:
                    task_idx = task_idx.expand(x.shape[0])
        return task_idx.long()

    def get_value(self, x, task_idx=None):
        task_idx = self._format_task_idx(x, task_idx)
        w = self.task_encoder(task_idx)
        critic_features = self.critic_backbone(x, w)
        return self.critic_head(critic_features, w)

    def get_action_and_value(self, x, action=None, task_idx=None, sample=True):
        task_idx = self._format_task_idx(x, task_idx)
        w = self.task_encoder(task_idx)
        
        actor_features = self.actor_backbone(x, w)
        action_mean = self.actor_head(actor_features, w)
        
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample() if sample else action_mean
            
        critic_features = self.critic_backbone(x, w)
        value = self.critic_head(critic_features, w)
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

# =========================================================================================
# Soft Modularization
# Ported directly from MTBench (RLC 2025) PyTorch Implementation
# =========================================================================================

class SoftModLinearExpert(nn.Module):
    def __init__(self, num_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.rand(self.num_experts, self.in_features, self.out_features))
        self.bias = nn.Parameter(torch.rand(self.num_experts, 1, self.out_features))

        # MTBench weight_init_moe_layer
        for i in range(self.num_experts):
            nn.init.xavier_uniform_(self.weight[i])
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_experts, B, in_features)
        return torch.baddbmm(self.bias, x, self.weight)

class SoftModRoutingNetwork(nn.Module):
    def __init__(self, hidden_features: int, num_experts_per_layer: int, num_layers: int):
        super().__init__()
        self.num_experts_per_layer = num_experts_per_layer
        self.W_d = nn.ModuleList([nn.Linear(hidden_features, num_experts_per_layer ** 2) for _ in range(num_layers)])
        self.W_u = nn.ModuleList([nn.Linear(num_experts_per_layer ** 2, hidden_features) for _ in range(num_layers - 1)])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # inp: (B, hidden_features)
        p = self.W_d[0](F.relu(inp))
        prob = [p]
        for W_u, W_d in zip(self.W_u, self.W_d[1:]):
            p = W_d(F.relu((W_u(prob[-1]) * inp)))
            prob.append(p)
            
        prob_tensor = torch.cat([
            F.softmax(logprob.reshape(logprob.shape[0], self.num_experts_per_layer, self.num_experts_per_layer), dim=2).unsqueeze(0)
            for logprob in prob
        ], dim=0)
        return prob_tensor # (num_layers, B, experts, experts)

class SoftModularizedMLP(nn.Module):
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
        
        self.routing_network = SoftModRoutingNetwork(
            hidden_features=hidden_features,
            num_layers=num_layers - 1,
            num_experts_per_layer=num_experts,
        )

    def forward(self, f_obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inp = f_obs * z
        probs = self.routing_network(inp) # (num_layers-1, B, E, E)
        num_experts = probs.shape[2]
        
        x = f_obs.unsqueeze(0).expand(num_experts, -1, -1)  # (E, B, D)
        for index, layer in enumerate(self.layers[:-1]):
            p = probs[index] # (B, E, E)
            x = layer(x) # (E, B, D)
            
            x_perm = x.permute(1, 0, 2) # (B, E, D)
            x_new = torch.bmm(p, x_perm) # (B, E, D)
            x = x_new.permute(1, 0, 2) # (E, B, D)
            
        out = self.layers[-1](x).sum(dim=0) # (B, out_dim)
        return out

class SoftModularActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dims=[256, 256, 256, 256], num_tasks=10, num_modules=2):
        super().__init__()
        obs_shape = int(np.array(observation_space.shape).prod())
        self.action_dim = int(np.array(action_space.shape).prod())
        hidden_dim = hidden_dims[0]
        
        self.task_embedding_dim = hidden_dim
        self.task_encoder = nn.Sequential(nn.Embedding(num_tasks, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        
        # State encoder logic mapping exactly to SoftModularizedMLPWrapper
        self.state_encoder = nn.Sequential(nn.Linear(obs_shape, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        
        # Actor Critic Core
        num_layers = len(hidden_dims)
        self.actor = SoftModularizedMLP(num_experts=num_modules, in_features=hidden_dim, out_features=self.action_dim, num_layers=num_layers, hidden_features=hidden_dim)
        self.critic = SoftModularizedMLP(num_experts=num_modules, in_features=hidden_dim, out_features=1, num_layers=num_layers, hidden_features=hidden_dim)
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def _format_task_idx(self, x, task_idx):
        if task_idx is None:
            raise ValueError("SoftMod requires task_idx")
        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=x.device).expand(x.shape[0])
        elif isinstance(task_idx, torch.Tensor):
            if task_idx.dim() == 0:
                task_idx = task_idx.expand(x.shape[0])
            elif len(task_idx) != x.shape[0]:
                if len(task_idx) == 1:
                    task_idx = task_idx.expand(x.shape[0])
        return task_idx.long()

    def get_value(self, x, task_idx=None):
        task_idx = self._format_task_idx(x, task_idx)
        f_obs = self.state_encoder(x)
        z = self.task_encoder(task_idx)
        return self.critic(f_obs, z)

    def get_action_and_value(self, x, action=None, task_idx=None, sample=True):
        task_idx = self._format_task_idx(x, task_idx)
        f_obs = self.state_encoder(x)
        z = self.task_encoder(task_idx)
        
        action_mean = self.actor(f_obs, z)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample() if sample else action_mean
            
        value = self.critic(f_obs, z)
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

# =========================================================================================
# CARE: Context-Aware Representation
# Ported directly from MTBench (RLC 2025) PyTorch Implementation
# =========================================================================================

import json

class CAREContextEncoder(nn.Module):
    def __init__(self, metadata_path: str, ordered_task_names: list, hidden_dim: int):
        super().__init__()
        with open(metadata_path) as f:
            metadata = json.load(f)
            
        pretrained_embedding = torch.Tensor([metadata[task] for task in ordered_task_names])
        self.pretrained_task_embedding_dim = pretrained_embedding.shape[1]
        
        self.pretrained_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)
        
        # Projection layer
        self.projection_layer = nn.Sequential(
            nn.Linear(self.pretrained_task_embedding_dim, 2 * self.pretrained_task_embedding_dim, bias=True),
            nn.ReLU(),
            nn.Linear(2 * self.pretrained_task_embedding_dim, self.pretrained_task_embedding_dim, bias=True),
            nn.ReLU()
        )
        
        # MLP context network (units: [50, 50])
        self.mlp = nn.Sequential(
            nn.Linear(self.pretrained_task_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        for layer in self.projection_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, task_idx: torch.Tensor) -> torch.Tensor:
        emb = self.pretrained_embedding(task_idx)
        proj = self.projection_layer(emb)
        return self.mlp(proj)

class CAREFeedForward(nn.Module):
    def __init__(self, num_experts: int, in_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        current_in = in_features
        for _ in range(num_layers - 1):
            self.layers.append(SoftModLinearExpert(num_experts, current_in, hidden_dim))
            self.layers.append(nn.ReLU())
            current_in = hidden_dim
            
        self.layers.append(SoftModLinearExpert(num_experts, current_in, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)
        out = x.unsqueeze(0).expand(self.layers[0].num_experts, -1, -1)
        for layer in self.layers:
            out = layer(out)
        return out # (num_experts, B, hidden_dim)

class CARENetwork(nn.Module):
    def __init__(self, in_features: int, num_experts: int, encoder_D: int, context_D: int, attention_D: int, care_hidden_dims: list, out_dim: int, num_layers: int = 2):
        super().__init__()
        self.temperature = 1.0
        
        self.phi = CAREFeedForward(num_experts, in_features, encoder_D, num_layers)
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(encoder_D, attention_D),
            nn.ReLU(),
            nn.Linear(attention_D, attention_D),
            nn.ReLU()
        )
        for layer in self.attention_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))

        network_layers = []
        curr_in = attention_D + context_D
        for h in care_hidden_dims:
            layer = nn.Linear(curr_in, h)
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            network_layers.append(layer)
            network_layers.append(nn.ReLU())
            curr_in = h
            
        self.network = nn.Sequential(*network_layers)
        
        self.head = nn.Linear(care_hidden_dims[-1], out_dim)
        nn.init.xavier_uniform_(self.head.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, true_obs: torch.Tensor, z_context: torch.Tensor) -> torch.Tensor:
        z_context_expanded = z_context.unsqueeze(1) # (B, 1, D)
        
        reprs = self.phi(true_obs) # (num_experts, B, D)
        reprs = reprs.permute(1, 2, 0) # (B, D, num_experts)
        
        attn_scores = z_context_expanded.detach() @ reprs # (B, 1, num_experts)
        attn_scores = attn_scores.squeeze(1) # (B, num_experts)
        alphas = F.softmax(attn_scores / self.temperature, dim=1) # (B, num_experts)
        
        z_enc = reprs @ alphas.unsqueeze(2) # (B, D, 1)
        z_enc = z_enc.squeeze(-1) # (B, D)
        
        z_enc = self.attention_mlp(z_enc) # (B, D)
        z_obs = torch.cat([z_enc, z_context], dim=1) # (B, 2*D)
        
        z_obs = self.network(z_obs)
        return self.head(z_obs)

class CAREActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dims=[400, 400, 400], num_tasks=10, num_experts=6):
        super().__init__()
        obs_shape = int(np.array(observation_space.shape).prod())
        self.action_dim = int(np.array(action_space.shape).prod())
        
        # Default MT10 ordering matching MTBench JSON keys
        ordered_task_names = ['reach', 'push', 'pick-place', 'door-open', 'drawer-open', 'drawer-close', 'button-press-topdown', 'peg-insert-side', 'window-open', 'window-close']
        metadata_path = 'temp/MTBench/isaacgymenvs/metadata/task_embedding/roberta_small/metaworld-mt10.json'
        
        # MTBench hardcodes context to 50, attention to 50, encoder_D to 50
        encoder_D = 50
        context_D = 50
        attention_D = 50
        
        self.context_encoder = CAREContextEncoder(metadata_path, ordered_task_names, hidden_dim=context_D)
        
        self.actor = CARENetwork(obs_shape, num_experts, encoder_D, context_D, attention_D, hidden_dims, self.action_dim)
        self.critic = CARENetwork(obs_shape, num_experts, encoder_D, context_D, attention_D, hidden_dims, 1)
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def _format_task_idx(self, x, task_idx):
        if task_idx is None:
            raise ValueError("CARE requires task_idx")
        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=x.device).expand(x.shape[0])
        elif isinstance(task_idx, torch.Tensor):
            if task_idx.dim() == 0:
                task_idx = task_idx.expand(x.shape[0])
            elif len(task_idx) != x.shape[0]:
                if len(task_idx) == 1:
                    task_idx = task_idx.expand(x.shape[0])
        return task_idx.long()

    def get_value(self, x, task_idx=None):
        task_idx = self._format_task_idx(x, task_idx)
        z_context = self.context_encoder(task_idx)
        return self.critic(x, z_context)

    def get_action_and_value(self, x, action=None, task_idx=None, sample=True):
        task_idx = self._format_task_idx(x, task_idx)
        z_context = self.context_encoder(task_idx)
        
        action_mean = self.actor(x, z_context)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample() if sample else action_mean
            
        value = self.critic(x, z_context)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

# =========================================================================================
# MOORE: Modular Orthogonal Representations
# Ported directly from MTBench (RLC 2025) PyTorch Implementation
# =========================================================================================

class MOOREOrthogonalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_experts, B, hidden_dim)
        x1 = torch.transpose(x, 0, 1) # (B, num_experts, hidden_dim)
        
        # Gram-Schmidt Orthogonalization
        basis = torch.unsqueeze(x1[:, 0, :] / (torch.unsqueeze(torch.linalg.norm(x1[:, 0, :], axis=1), 1) + 1e-8), 1)

        for i in range(1, x1.shape[1]):
            v = torch.unsqueeze(x1[:, i, :], 1)
            # Projection
            w = v - torch.matmul(torch.matmul(v, torch.transpose(basis, 2, 1)), basis)
            wnorm = w / (torch.unsqueeze(torch.linalg.norm(w, axis=2), 2) + 1e-8)
            basis = torch.cat([basis, wnorm], axis=1)

        basis = torch.transpose(basis, 0, 1) # (num_experts, B, hidden_dim)
        return basis

class MOORENetwork(nn.Module):
    def __init__(self, in_features: int, num_experts: int, hidden_dim: int, num_layers: int, out_dim: int, num_tasks: int):
        super().__init__()
        
        # Task Encoder (to generate w) -> units: [256] -> out_features: num_experts
        self.task_encoder = nn.Sequential(
            nn.Embedding(num_tasks, 256), # Using Embedding instead of raw one-hot for flexibility
            nn.ReLU(),
            nn.Linear(256, num_experts, bias=False)
        )
        nn.init.xavier_uniform_(self.task_encoder[-1].weight, gain=nn.init.calculate_gain('linear'))
        
        # Expert FeedForward Backbone
        self.phi = CAREFeedForward(num_experts, in_features, hidden_dim, num_layers)
        
        self.orthogonal_layer = MOOREOrthogonalLayer()
        
        # Single Head Architecture
        self.head = nn.Linear(hidden_dim + 256, out_dim)
        nn.init.xavier_uniform_(self.head.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, true_obs: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        task_embedding = self.task_encoder[0](task_idx)
        w = self.task_encoder[1:](task_embedding).unsqueeze(1) # (B, 1, num_experts)
        
        reprs = self.phi(true_obs) # (num_experts, B, D)
        ortho_reprs = self.orthogonal_layer(reprs) # (num_experts, B, D)
        
        ortho_reprs = ortho_reprs.permute(1, 0, 2) # (B, num_experts, D)
        
        # Linear aggregation
        ortho_reprs = w @ ortho_reprs # (B, 1, D)
        ortho_reprs = ortho_reprs.squeeze(1) # (B, D)
        
        # Concatenate task embedding with orthogonal representations
        inp = torch.cat([ortho_reprs, task_embedding], dim=1) # (B, D + 256)
        return self.head(inp)

class MOOREActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dims=[400, 400, 400], num_tasks=10, num_experts=4):
        super().__init__()
        obs_shape = int(np.array(observation_space.shape).prod())
        self.action_dim = int(np.array(action_space.shape).prod())
        hidden_dim = hidden_dims[0]
        num_layers = len(hidden_dims)
        
        self.actor = MOORENetwork(obs_shape, num_experts, hidden_dim, num_layers, self.action_dim, num_tasks)
        self.critic = MOORENetwork(obs_shape, num_experts, hidden_dim, num_layers, 1, num_tasks)
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def _format_task_idx(self, x, task_idx):
        if task_idx is None:
            raise ValueError("MOORE requires task_idx")
        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=x.device).expand(x.shape[0])
        elif isinstance(task_idx, torch.Tensor):
            if task_idx.dim() == 0:
                task_idx = task_idx.expand(x.shape[0])
            elif len(task_idx) != x.shape[0]:
                if len(task_idx) == 1:
                    task_idx = task_idx.expand(x.shape[0])
        return task_idx.long()

    def get_value(self, x, task_idx=None):
        task_idx = self._format_task_idx(x, task_idx)
        return self.critic(x, task_idx)

    def get_action_and_value(self, x, action=None, task_idx=None, sample=True):
        task_idx = self._format_task_idx(x, task_idx)
        
        action_mean = self.actor(x, task_idx)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample() if sample else action_mean
            
        value = self.critic(x, task_idx)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

class IndependentActorCritic(nn.Module):
    """
    Single-Task Baseline: 100% independent MLPs for each task.
    Takes batched inputs and routes them strictly to their assigned isolated MLPs.
    """
    def __init__(self, observation_space, action_space, hidden_dims=[400, 400, 400], num_tasks=10):
        super().__init__()
        self.num_tasks = num_tasks
        
        obs_shape = int(np.array(observation_space.shape).prod())
        
        if hasattr(action_space, 'n'):
            self.is_continuous = False
            self.action_dim = int(action_space.n)
        else:
            self.is_continuous = True
            self.action_dim = int(np.array(action_space.shape).prod())
            
        def build_mlp(out_dim, is_policy=False):
            layers = []
            curr_in = obs_shape
            for h in hidden_dims:
                layers.append(layer_init(nn.Linear(curr_in, h)))
                layers.append(nn.ReLU())
                curr_in = h
            # Standard PPO initialization: policy head std=0.01, value head std=1.0
            gain = 0.01 if is_policy else 1.0
            layers.append(layer_init(nn.Linear(curr_in, out_dim), std=gain))
            return nn.Sequential(*layers)
            
        self.actors = nn.ModuleList([build_mlp(self.action_dim, is_policy=True) for _ in range(num_tasks)])
        self.critics = nn.ModuleList([build_mlp(1, is_policy=False) for _ in range(num_tasks)])
        
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
        # Find linear layer indices dynamically
        linear_indices = [i for i, layer in enumerate(first_mlp) if isinstance(layer, nn.Linear)]
        
        g = x
        for step, l_idx in enumerate(linear_indices):
            # Stack weights and biases across all independent task MLPs
            weights = torch.stack([mlp[l_idx].weight for mlp in mlps])
            biases = torch.stack([mlp[l_idx].bias for mlp in mlps])
            
            # Index task-specific parameters for this batch
            w_b = weights[task_idx]
            b_b = biases[task_idx]
            
            # Batched matrix multiplication: (B, out_features, in_features) x (B, in_features, 1) -> (B, out_features)
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


class ActorCritic(nn.Module):
    """
    Standard Multilayer Perceptron Actor-Critic network for PPO.
    
    Supports optional Task Embeddings for Multi-Task Reinforcement Learning (MTRL).
    This architecture is used as the backbone for standard 'shared', 'pcgrad',
    and 'cagrad' baselines.
    """
    def __init__(
        self, 
        observation_space, 
        action_space, 
        hidden_dim: int = 64, 
        use_task_embedding: bool = False, 
        embedding_dim: int = 10, 
        num_tasks: int = 1,
        num_layers: int = 2,
        use_varshare: bool = False,
        **kwargs
    ):
        super().__init__()
        self.use_task_embedding = use_task_embedding
        self.num_tasks = num_tasks
        
        obs_shape = int(np.array(observation_space.shape).prod())
        
        # Check Action Space Type (Discrete vs. Continuous)
        if hasattr(action_space, 'n'):
            self.is_continuous = False
            self.action_dim = int(action_space.n)
        else:
            self.is_continuous = True
            self.action_dim = int(np.array(action_space.shape).prod())
        
        input_dim = obs_shape
        if use_task_embedding:
            # Task Embedding lookup layer for multi-task sharing
            self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
            # Initialize embedding orthogonally for stability
            nn.init.orthogonal_(self.task_embedding.weight, gain=1.0)
            input_dim += embedding_dim
            
        # Critic network definition
        critic_layers = []
        curr_in = input_dim
        for _ in range(num_layers):
            critic_layers.append(layer_init(nn.Linear(curr_in, hidden_dim)))
            critic_layers.append(nn.Tanh())
            curr_in = hidden_dim
        critic_layers.append(layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)
        
        # Actor network definition
        actor_layers = []
        curr_in = input_dim
        for _ in range(num_layers):
            actor_layers.append(layer_init(nn.Linear(curr_in, hidden_dim)))
            actor_layers.append(nn.Tanh())
            curr_in = hidden_dim
        # Policy head uses standard orthogonal initialization with a gain of 0.01
        # to ensure initial action distributions are near-uniform.
        actor_layers.append(layer_init(nn.Linear(hidden_dim, self.action_dim), std=0.01))
        self.actor_mean = nn.Sequential(*actor_layers)
        
        if self.is_continuous:
            # Shared log standard deviation for continuous Gaussian policies
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def _get_input(self, x: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        """
        Processes observation and optional task index to build the input vector.
        """
        if self.use_task_embedding:
            if task_idx is None:
                raise ValueError("task_idx is required when use_task_embedding is enabled")
            
            # Format and expand task index to match batch size
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

    def get_features(self, x: torch.Tensor, task_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Extracts feature representations from input.
        """
        return self._get_input(x, task_idx)

    def get_value(self, x: torch.Tensor, task_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the state-value estimate (Critic).
        """
        x_in = self._get_input(x, task_idx)
        return self.critic(x_in)

    def get_action_and_value(
        self, 
        x: torch.Tensor, 
        action: torch.Tensor = None, 
        task_idx: torch.Tensor = None, 
        sample: bool = True
    ):
        """
        Computes the policy distribution, action log probabilities, entropy, and values.
        """
        x_in = self._get_input(x, task_idx)
        
        if self.is_continuous:
            action_mean = self.actor_mean(x_in)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            
            probs = torch.distributions.Normal(action_mean, action_std)
            if not sample:
                action = action_mean
            elif action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x_in)
        else:
            logits = self.actor_mean(x_in)
            probs = torch.distributions.Categorical(logits=logits)
            if not sample:
                action = torch.argmax(logits, dim=1)
            elif action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(x_in)


