# lnn/liquid.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidLayer(nn.Module):
    """
    Liquid Time-constant (LTC) cell per Hasani et al.
    h <- h + (dt/tau) * ( -h + tanh(Wx x + Wh h + b) ), with tau > 0.
    Supports sequence inputs and an optional number of Euler steps.
    """
    def __init__(self, in_dim: int, hidden_dim: int, tau_min: float = 1.0, tau_max: float = 2.0):
        super().__init__()
        self.Wx = nn.Linear(in_dim, hidden_dim, bias=False)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        init_tau = torch.empty(hidden_dim).uniform_(tau_min, tau_max)
        self.log_tau = nn.Parameter(init_tau.log())  # tau = exp(log_tau)

    def forward(self, x: torch.Tensor, steps: int = 1, dt: float = 1.0):
        """
        x: (B, T, F) or (B, F) -> returns (B, T, H)
        """
        if x.dim() == 2:  # (B,F) -> (B,1,F)
            x = x.unsqueeze(1)
        B, T, _ = x.shape
        H = self.Wh.out_features
        h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        tau = torch.exp(self.log_tau)  # (H,)

        outs = []
        for t in range(T):
            xt = x[:, t, :]
            for _ in range(max(1, steps)):
                z = torch.tanh(self.Wx(xt) + self.Wh(h) + self.bias)  # (B,H)
                dh = (-h + z) / tau
                h  = h + dt * dh
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1)

class LiquidPolicyClassifier(nn.Module):
    """
    Simple 1-step classifier (legacy). Input: (B,F) -> logits (B,9)
    """
    def __init__(self, in_dim: int, d_model: int = 64, liquid_steps: int = 1, n_actions: int = 9):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.l1 = LiquidLayer(d_model, d_model)
        self.l2 = LiquidLayer(d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_actions)
        )
        self.liquid_steps = liquid_steps

    def forward(self, feats: torch.Tensor):
        x = F.relu(self.in_proj(feats))             # (B,d)
        x = self.l1(x, steps=self.liquid_steps)     # (B,1,d)
        x = self.l2(x, steps=self.liquid_steps)     # (B,1,d)
        x = x.squeeze(1)
        return self.head(x)

class LiquidPolicySeq(nn.Module):
    """
    Sequence trainer. Input: (B, T, F). Output: logits at last step (B,9)
    """
    def __init__(self, in_dim: int, d_model: int = 64, liquid_steps: int = 1, n_actions: int = 9):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.l1 = LiquidLayer(d_model, d_model)
        self.l2 = LiquidLayer(d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_actions)
        )
        self.liquid_steps = liquid_steps

    def forward(self, feats_seq: torch.Tensor):
        # feats_seq: (B, T, F)
        B, T, Fdim = feats_seq.shape
        x = F.relu(self.in_proj(feats_seq))         # (B, T, d)
        x = self.l1(x, steps=self.liquid_steps)     # (B, T, d)
        x = self.l2(x, steps=self.liquid_steps)     # (B, T, d)
        x_last = x[:, -1, :]                        # (B, d)
        return self.head(x_last)

class LiquidPolicyStateful(nn.Module):
    """
    Step-wise API with internal hidden state (for deployment).
    """
    def __init__(self, in_dim: int, d_model: int = 64, liquid_steps: int = 1, n_actions: int = 9):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.l1 = LiquidLayer(d_model, d_model)
        self.l2 = LiquidLayer(d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_actions)
        )
        self.liquid_steps = liquid_steps
        self._h1 = None
        self._h2 = None

    def reset_state(self, batch_size=1, device='cpu', dtype=torch.float32):
        self._h1 = torch.zeros(batch_size, self.l1.Wh.out_features, device=device, dtype=dtype)
        self._h2 = torch.zeros(batch_size, self.l2.Wh.out_features, device=device, dtype=dtype)

    def step(self, feat: torch.Tensor):
        """
        feat: (B, F) -> logits (B,9); keeps internal state across calls.
        """
        if self._h1 is None or self._h2 is None:
            self.reset_state(batch_size=feat.size(0), device=feat.device, dtype=feat.dtype)

        x = F.relu(self.in_proj(feat))  # (B,d)

        # one Euler step update using internal states for each liquid layer
        # layer 1
        z1   = torch.tanh(self.l1.Wx(x) + self.l1.Wh(self._h1) + self.l1.bias)
        tau1 = torch.exp(self.l1.log_tau)
        self._h1 = self._h1 + ((-self._h1 + z1) / tau1)

        # layer 2
        z2   = torch.tanh(self.l2.Wx(self._h1) + self.l2.Wh(self._h2) + self.l2.bias)
        tau2 = torch.exp(self.l2.log_tau)
        self._h2 = self._h2 + ((-self._h2 + z2) / tau2)

        return self.head(self._h2)
