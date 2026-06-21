import torch
import torch.nn as nn
from typing import Dict, Any

class HardRoutingSTEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft_probs):
        hard = torch.zeros_like(soft_probs)
        hard.scatter_(-1, soft_probs.argmax(dim=-1, keepdim=True), 1.0)
        return hard
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class LayerRouter(nn.Module):
    def __init__(self, num_heads: int, d_k: int = None):
        super().__init__()
        self.num_heads = num_heads
        
        self.input_norm = nn.LayerNorm(4)
        
        # Grouped 1D conv behaves as `num_heads` separate Linear layers
        self.fc1 = nn.Conv1d(num_heads * 4, num_heads * 16, kernel_size=1, groups=num_heads)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(num_heads * 16, num_heads * 4, kernel_size=1, groups=num_heads)
        
        self._last_probs = None
        
    def forward(self, features: torch.Tensor, hard_routing: bool = False):
        """
        Inputs:
            features: [batch, num_heads, 4]
        Output:
            probs: [batch, num_heads, 4]
        """
        B, H, _ = features.shape
        x = self.input_norm(features)
        
        x = x.view(B, H * 4, 1) # [B, in_channels, 1]
        
        h1 = self.relu(self.fc1(x)) # [B, H * 16, 1]
        logits = self.fc2(h1) # [B, H * 4, 1]
        
        logits = logits.squeeze(-1) # [B, H * 4]
        logits = logits.view(B, H, 4) # [B, H, 4]
        soft_probs = logits.softmax(dim=-1)
        
        self._last_probs_for_loss = soft_probs          # grad-attached, used in loss.backward()
        self._last_probs = soft_probs.detach()          # detached, safe for logging/stats only
        self._last_probs_detached = self._last_probs    # alias kept for compat
        
        if hard_routing:
            hard_probs = HardRoutingSTEstimator.apply(soft_probs)
            return hard_probs
        return soft_probs
        
    def get_last_routing_probs(self):
        return getattr(self, '_last_probs_detached', None)

def initialize_layer_router_from_profile(layer_router: LayerRouter, layer_idx: int, audit_priors: dict):
    """
    Bias the grouped conv weights for specific heads based on Phase 0 priors.
    """
    with torch.no_grad():
        for h in range(layer_router.num_heads):
            entries = audit_priors.get((layer_idx, h), [])
            if not entries:
                layer_router.fc2.bias.data[h*4 : (h+1)*4] = torch.zeros(4)
                continue
                
            local_error = next((r["attn_l_inf_natural_mean"] for r in entries if r["type"] == "local"), 1.0)
            sink_error = next((r["attn_l_inf_natural_mean"] for r in entries if r["type"] == "sink"), 1.0)
            
            if local_error < 0.10:
                layer_router.fc2.bias.data[h*4 : (h+1)*4] = torch.tensor([-2.0, 2.0, -2.0, -2.0], device=layer_router.fc2.bias.device)
            elif sink_error < 0.10:
                layer_router.fc2.bias.data[h*4 : (h+1)*4] = torch.tensor([2.0, -2.0, -2.0, -2.0], device=layer_router.fc2.bias.device)
            else:
                layer_router.fc2.bias.data[h*4 : (h+1)*4] = torch.zeros(4, device=layer_router.fc2.bias.device)


def initialize_layer_router_profile_a(layer_router: LayerRouter):
    """
    Profile A: Conservative initialization, all heads start with soft bias to full attention
    so entropy curriculum + training pressure can gradually pull heads to cheap paths.
    """
    with torch.no_grad():
        for h in range(layer_router.num_heads):
            layer_router.fc2.bias.data[h*4 : (h+1)*4] = torch.tensor(
                [-0.5, -0.5, -0.5, 1.0],  # much softer!
                device=layer_router.fc2.bias.device
            )
