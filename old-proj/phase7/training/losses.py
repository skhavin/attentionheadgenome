import torch
import torch.nn.functional as F

def routing_loss(mse_loss: torch.Tensor, all_routing_probs: list[torch.Tensor], entropy_weight: float = 0.01, compute_weight: float = 0.0):
    """
    Computes total loss = MSE + entropy_weight * entropy + compute_weight * expected_cost
    Entropy penalty encourages confident routing.
    Compute penalty encourages routing to cheaper paths.
    """
    if not all_routing_probs:
        return mse_loss, 0.0, 0.0, 0.0
        
    # Stack all routing probs [num_routers, batch, num_heads, 4]
    probs = torch.stack(all_routing_probs)
    
    # Entropy = - sum(p * log(p))
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
    
    # Expected Compute Cost
    # Paths: 0=sink, 1=local, 2=rec, 3=full
    cost_weights = torch.tensor([0.1, 0.2, 0.1, 1.0], device=probs.device, dtype=probs.dtype).view(1, 1, 1, 4)
    expected_cost = (probs * cost_weights).sum(dim=-1).mean()
    
    total_loss = mse_loss + (entropy_weight * entropy) + (compute_weight * expected_cost)
    cheap_pct = (probs.argmax(dim=-1) < 3).float().mean().item()
    
    return total_loss, entropy, expected_cost, cheap_pct
