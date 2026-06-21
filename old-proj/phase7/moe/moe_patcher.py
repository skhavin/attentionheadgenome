import torch
import torch.nn as nn
from phase7.moe.router import LayerRouter
from phase7.moe.paths import sink_path, local_path, full_path, recurrence_path

def compute_router_features(Q: torch.Tensor, K: torch.Tensor, window: int = 16) -> torch.Tensor:
    """
    Computes O(W) features per head. Q, K: [B, H, N, d]
    Returns: [B, H, 4]
    """
    Q_last = Q[:, :, -1:, :] # [B, H, 1, d]
    
    num_sink = 4
    K_sink = K[:, :, :num_sink, :]
    K_recent = K[:, :, -window:, :]
    
    K_combined = torch.cat([K_sink, K_recent], dim=-2) # [B, H, 4+W, d]
    
    scores = torch.einsum('bhid,bhjd->bhij', Q_last, K_combined) / (Q.shape[-1] ** 0.5)
    probs = scores.softmax(dim=-1)
    
    actual_sink = K_sink.shape[-2]
    sink_mass = probs[:, :, :, :actual_sink].sum(dim=-1) # [B, H, 1]
    
    local_probs = probs[:, :, :, actual_sink:] # [B, H, 1, W]
    local_entropy = -(local_probs * (local_probs + 1e-8).log().clamp(-10)).sum(dim=-1) # [B, H, 1]
    recency_mass = local_probs.sum(dim=-1) # [B, H, 1]
    max_sim = scores[:, :, :, actual_sink:].max(dim=-1).values # [B, H, 1]
    
    return torch.cat([
        local_entropy,
        sink_mass,
        recency_mass,
        max_sim
    ], dim=-1).to(Q.dtype) # [B, H, 4], preserve dtype!

class MoEPatcher:
    def __init__(self, model, num_sink_tokens: int = 4, local_window: int = 64, hard_routing: bool = False):
        self.model = model
        self.num_sink_tokens = num_sink_tokens
        self.local_window = local_window
        self.hard_routing = hard_routing
        self.routers = nn.ModuleDict() # LayerRouters
        self._original_forwards = {}
        self._hook_handles = []
        self.accumulated_mse = 0.0
        self.active_routing_layer = -1
        self.locked_layers = set()
        self.full_attention_only_layers = set()
        self.path_activations = None
        self.activation_count = 0
        self.feature_stats = None
        
        self.arch = self._determine_arch()
        self._install_routers()
        
        # Move routers to model dtype
        model_dtype = next(model.parameters()).dtype
        self.routers.to(model_dtype)

    def set_single_layer_routing(self, target_layer: int):
        """Only target layer uses MoE routing. All others use full path or locked paths."""
        self.active_routing_layer = target_layer

    def lock_layer(self, layer_idx: int):
        """Permanently lock a layer to hard routing using learned weights."""
        self.locked_layers.add(layer_idx)

    def force_full_attention(self, layer_idx: int):
        """Force a layer to always run exact full attention."""
        self.full_attention_only_layers.add(layer_idx)

    def _determine_arch(self):
        config_cls = type(self.model.config).__name__.lower()
        if "gpt2" in config_cls: return "gpt2"
        if "llama" in config_cls: return "llama"
        if "qwen" in config_cls: return "qwen"
        return "gpt2"

    def _get_head_info(self, attn_module):
        if self.arch == "gpt2":
            num_heads = getattr(self.model.config, "n_head", 12)
            d_head = attn_module.c_attn.weight.shape[1] // (3 * num_heads)
            num_kv_heads = num_heads
        else:
            num_heads = getattr(self.model.config, "num_attention_heads", 32)
            num_kv_heads = getattr(self.model.config, "num_key_value_heads", num_heads)
            if hasattr(self.model.config, "head_dim"):
                d_head = self.model.config.head_dim
            else:
                d_out = attn_module.q_proj.weight.shape[0]
                d_head = d_out // num_heads
        return num_heads, num_kv_heads, d_head

    def _install_routers(self):
        for name, module in self.model.named_modules():
            cls_name = type(module).__name__
            if "Attention" in cls_name and (hasattr(module, "q_proj") or hasattr(module, "c_attn")):
                self._patch_layer(name, module)

    def _patch_layer(self, name: str, attn_module):
        original_forward = attn_module.forward
        self._original_forwards[name] = original_forward

        parts = name.split(".")
        layer_idx = None
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break
        if layer_idx is None:
            layer_idx = 0

        num_heads, num_kv_heads, d_head = self._get_head_info(attn_module)

        router_name = f"l{layer_idx}"
        self.routers[router_name] = LayerRouter(num_heads, d_k=d_head).to(self.model.device)

        captured = {}
        
        def c_attn_hook(module, inp, output):
            B, N, d_out = output.shape
            q, k, v = output.split(d_out // 3, dim=2)
            captured["q"] = q.view(B, N, num_heads, d_head).transpose(1, 2)
            captured["k"] = k.view(B, N, num_heads, d_head).transpose(1, 2)
            captured["v"] = v.view(B, N, num_heads, d_head).transpose(1, 2)

        def q_hook(module, inp, output):
            B, N, d_out = output.shape
            captured["q"] = output.view(B, N, num_heads, d_head).transpose(1, 2)
            
        def k_hook(module, inp, output):
            B, N, d_out = output.shape
            k = output.view(B, N, num_kv_heads, d_head)
            groups = num_heads // num_kv_heads
            if groups > 1:
                k = k.repeat_interleave(groups, dim=2)
            captured["k"] = k.transpose(1, 2)

        def v_hook(module, inp, output):
            B, N, d_out = output.shape
            v = output.view(B, N, num_kv_heads, d_head)
            groups = num_heads // num_kv_heads
            if groups > 1:
                v = v.repeat_interleave(groups, dim=2)
            captured["v"] = v.transpose(1, 2)

        if self.arch == "gpt2":
            self._hook_handles.append(attn_module.c_attn.register_forward_hook(c_attn_hook))
            self._hook_handles.append(attn_module.c_proj.register_forward_pre_hook(lambda m, args: captured.update({"attn_out": args[0]})))
        else:
            self._hook_handles.append(attn_module.q_proj.register_forward_hook(q_hook))
            self._hook_handles.append(attn_module.k_proj.register_forward_hook(k_hook))
            self._hook_handles.append(attn_module.v_proj.register_forward_hook(v_hook))
            self._hook_handles.append(attn_module.o_proj.register_forward_pre_hook(lambda m, args: captured.update({"attn_out": args[0]})))

        patcher = self

        def patched_forward(*args, **kwargs):
            captured.clear()
            
            result = original_forward(*args, **kwargs)
            
            if "q" not in captured or "k" not in captured or "v" not in captured:
                return result
                
            Q = captured["q"] # [B, H, N, d]
            K = captured["k"]
            V = captured["v"]
            
            B, H, N, d_head = V.shape
            
            full_out = captured["attn_out"].view(B, N, H, d_head).transpose(1, 2) # [B, H, N, d]
            sink_out = sink_path(V, patcher.num_sink_tokens)
            local_out = local_path(V, patcher.local_window)
            recurrence_out = recurrence_path(V, alpha=0.9)
            
            router = patcher.routers[f"l{layer_idx}"]
            features = compute_router_features(Q, K, window=16)
            patcher.last_features = features.detach()
            
            if patcher.feature_stats is None:
                patcher.feature_stats = {
                    "min": torch.full((4,), float('inf'), device=Q.device),
                    "max": torch.full((4,), float('-inf'), device=Q.device),
                    "sum": torch.zeros(4, device=Q.device),
                    "count": 0
                }
            
            flat_features = features.detach().view(-1, 4)
            patcher.feature_stats["min"] = torch.minimum(patcher.feature_stats["min"], flat_features.min(dim=0).values)
            patcher.feature_stats["max"] = torch.maximum(patcher.feature_stats["max"], flat_features.max(dim=0).values)
            patcher.feature_stats["sum"] += flat_features.sum(dim=0)
            patcher.feature_stats["count"] += flat_features.size(0)
            
            if layer_idx in patcher.full_attention_only_layers:
                return result
                
            if layer_idx in patcher.locked_layers:
                probs = router(features, hard_routing=True)
            elif layer_idx == patcher.active_routing_layer:
                probs = router(features, hard_routing=False)
            elif patcher.active_routing_layer != -1:
                # In progressive training mode, all non-active non-locked layers default to full
                probs = torch.zeros(B, H, 4, device=Q.device, dtype=Q.dtype)
                probs[:, :, 3] = 1.0
            else:
                # Normal mode
                probs = router(features, patcher.hard_routing)
                
            if patcher.path_activations is None:
                patcher.path_activations = torch.zeros(4, device=Q.device, dtype=torch.float32)
            
            # Aggregate probabilities across batch and heads
            # probs shape: [B, H, 4]
            patcher.path_activations += probs.mean(dim=(0, 1)).float().detach()
            patcher.activation_count += 1
                
            p_sink = probs[:, :, 0].view(B, H, 1, 1)
            p_local = probs[:, :, 1].view(B, H, 1, 1)
            p_rec = probs[:, :, 2].view(B, H, 1, 1)
            p_full = probs[:, :, 3].view(B, H, 1, 1)
            
            combined_out = (
                p_sink * sink_out +
                p_local * local_out +
                p_rec * recurrence_out +
                p_full * full_out
            )
                
            combined_out = combined_out.transpose(1, 2).reshape(B, N, H * d_head)
            if self.arch == "gpt2":
                final_out = attn_module.c_proj(combined_out)
            else:
                final_out = attn_module.o_proj(combined_out)
                
            if getattr(patcher, "training", False):
                patcher.accumulated_mse = patcher.accumulated_mse + torch.nn.functional.mse_loss(final_out.float(), result[0].detach().float())

            return (final_out,) + result[1:]

        attn_module.forward = patched_forward

    def reset_loss(self):
        self.accumulated_mse = 0.0

    def reset_activation_stats(self):
        self.path_activations = None
        self.activation_count = 0
        self.feature_stats = None

    def get_activation_stats(self):
        if self.path_activations is None or self.activation_count == 0:
            return [0.0, 0.0, 0.0, 0.0]
        # Return as list of percentages (0-100)
        return (self.path_activations / self.activation_count * 100).cpu().tolist()

    def get_feature_stats(self):
        if self.feature_stats is None or self.feature_stats["count"] == 0:
            return None
        return {
            "min": self.feature_stats["min"].cpu().tolist(),
            "max": self.feature_stats["max"].cpu().tolist(),
            "mean": (self.feature_stats["sum"] / self.feature_stats["count"]).cpu().tolist()
        }

    def restore(self):
        for name, module in self.model.named_modules():
            if name in self._original_forwards:
                module.forward = self._original_forwards[name]
        self._original_forwards.clear()
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
