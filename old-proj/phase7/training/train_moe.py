import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from phase7.moe.moe_patcher import MoEPatcher
from phase7.moe.routing_logger import log_routing_decisions
from phase7.training.losses import routing_loss

def load_audit_priors(audit_path):
    with open(audit_path, "r") as f:
        audit_data = json.load(f)
    # Group by (layer, head)
    head_entries = defaultdict(list)
    # The JSON structure might be a list of dicts if we read the raw rows
    # The current audit outputs a dict with "rows"
    if isinstance(audit_data, dict) and "rows" in audit_data:
        audit_data = audit_data["rows"]
        
    for r in audit_data:
        head_entries[(r["layer"], r["head"])].append(r)
    return head_entries

from phase7.moe.router import initialize_layer_router_from_profile, initialize_layer_router_profile_a

def initialize_routers_from_audit(patcher, audit_priors):
    for name, router in patcher.routers.items():
        # name is l{layer}
        l = int(name[1:])
        initialize_layer_router_from_profile(router, l, audit_priors)


def initialize_routers_profile_a(patcher):
    for name, router in patcher.routers.items():
        initialize_layer_router_profile_a(router)

def train_stage(model, patcher, optimizer, scheduler, dataloader, device, desc="Training"):
    model.eval() # Base model always in eval mode
    patcher.training = True
    
    total_loss = 0.0
    progress = tqdm(dataloader, desc=desc)
    
    for batch_idx, batch in enumerate(progress):
        input_ids = batch.to(device)
        
        patcher.reset_loss()
        
        # Forward pass (model is frozen, but patcher hook computes gradients for routers)
        _ = model(input_ids)
        
        mse_loss = patcher.accumulated_mse
        
        # Get all routing probs for entropy loss
        all_probs = []
        for r in patcher.routers.values():
            if hasattr(r, '_last_probs_for_loss') and r._last_probs_for_loss is not None:
                all_probs.append(r._last_probs_for_loss)
                
        if "Stage 1" in desc:
            entropy_weight = 0.0   # pure reconstruction — no pressure yet
            compute_weight = 0.05
        elif "Stage 2" in desc:
            entropy_weight = 1.5   # entropy term now comparable to MSE
            compute_weight = 1.5
        else:
            entropy_weight = 3.0   # entropy dominates — forces commitment
            compute_weight = 0.10
            
        if batch_idx == 0:
            print(f"{desc} entropy weight: {entropy_weight}, compute weight: {compute_weight}")
            
        loss_tuple = routing_loss(mse_loss, all_probs, entropy_weight=entropy_weight, compute_weight=compute_weight)
        if isinstance(loss_tuple, tuple):
            loss, entropy_val, expected_cost, cheap_pct = loss_tuple
        else:
            loss = loss_tuple
            entropy_val = 0.0
            expected_cost = 0.0
            cheap_pct = 0.0
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(patcher.routers.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        
        # Debug print every 2 batches
        if batch_idx % 2 == 0:
            if all_probs:
                probs = torch.stack(all_probs).detach()
                ratio = (entropy_weight * entropy_val) / (mse_loss.item() + 1e-8)
                avg_max_path_prob = probs.max(dim=-1).values.mean().item()
            else:
                ratio = 0.0
                avg_max_path_prob = 0.0
            print(f"\nBatch {batch_idx}: MSE={mse_loss:.4f} | Ent={entropy_val:.4f} (w={entropy_weight*entropy_val:.4f}) | "
                  f"Cost={expected_cost:.4f} (w={compute_weight*expected_cost:.4f}) | CheapPct={cheap_pct*100:.1f}% | Total={loss:.4f}")
            if avg_max_path_prob > 0.95 and mse_loss.item() > 0.3:
                print("  WARNING: Router polarizing before MSE converged — may be locking in bad decisions")
        
        if batch_idx < 5 and "Stage 1" in desc:  # Reduce to first 5 batches to cut overhead
            features = patcher.last_features
            if features is not None:
                print(f"\nBatch {batch_idx} features: "
                      f"entropy={features[:,:,0].mean():.3f}, "
                      f"sink={features[:,:,1].mean():.3f}, "
                      f"recency={features[:,:,2].mean():.3f}, "
                      f"maxsim={features[:,:,3].mean():.3f}")
        
    # Count cheap path usage from last batch
    all_probs = []
    for r in patcher.routers.values():
        if hasattr(r, '_last_probs_detached') and r._last_probs_detached is not None:
            all_probs.append(r._last_probs_detached)
    if all_probs:
        probs = torch.stack(all_probs)  # [num_routers, B, H, 4]
        decisions = probs.argmax(dim=-1)  # 0=sink, 1=local, 2=rec, 3=full
        cheap = (decisions < 3).float().mean().item()
        print(f"\nCheap path %: {cheap*100:.1f}%")
        
    patcher.training = False
    return total_loss / len(dataloader)

def get_dataloader(data_path, type_filter=None, batch_size=2):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
        
    if type_filter:
        data = [d for d in data if d["type"] == type_filter]
        
    # Convert to batched tensors
    batches = []
    for i in range(0, len(data), batch_size):
        batch_items = data[i:i+batch_size]
        # Pad to max length in batch if necessary, or just truncate
        max_len = max(len(item["tokens"]) for item in batch_items)
        tensors = []
        for item in batch_items:
            t = item["tokens"]
            if len(t) < max_len:
                t = t + [50256] * (max_len - len(t)) # PAD
            tensors.append(torch.tensor(t))
        batches.append(torch.stack(tensors))
        
    return batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--stage2_data", default="outputs/phase7/stage2_mixed.jsonl")
    parser.add_argument("--audit_path", default="outputs/phase7/head_audit.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--quantize_4bit", action="store_true")
    
    # New arguments for smoke test / sizing
    parser.add_argument("--stage1_docs", type=int, default=-1)
    parser.add_argument("--stage2_docs", type=int, default=-1)
    parser.add_argument("--stage3_docs", type=int, default=-1)
    parser.add_argument("--output", default="outputs/phase7/routers/")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop_at_stage", type=int, choices=[1, 2, 3], help="Stop training after this stage")
    parser.add_argument("--profile", choices=["A", "B", "audit"], default="B", help="Router initialization profile: A (conservative, full attention biased), B (uniform), audit (from Phase 0 priors)")
    parser.add_argument("--audit-prefix", default="", help="Prefix for audit files (e.g., 'qwen_' for qwen_head_audit.json)")
    parser.add_argument("--start-stage", type=int, choices=[1, 2, 3], default=1, help="Stage to start training from")
    parser.add_argument("--smoke_test", action="store_true", help="Run tiny smoke test (50/100/20 docs)")
    args = parser.parse_args()
    
    # Use audit prefix for audit path if provided
    if args.audit_prefix:
        audit_dir = os.path.dirname(args.audit_path)
        audit_filename = os.path.basename(args.audit_path)
        args.audit_path = os.path.join(audit_dir, f"{args.audit_prefix}{audit_filename}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    print(f"Loading {args.model}...")
    if args.quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            device_map="auto",
            torch_dtype=torch.bfloat16 if "llama" in args.model.lower() or "qwen" in args.model.lower() else torch.float32,
            attn_implementation="sdpa"
        )
        
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
        
    # Initialize Patcher
    print("Installing routers...")
    patcher = MoEPatcher(model, hard_routing=False)
    
    # Force late layers to always use full attention — cheap paths have too high MSE there
    num_layers = model.config.num_hidden_layers
    protected_layers = max(1, num_layers - 4)  # last 4 layers always full attention
    for lidx in range(protected_layers, num_layers):
        patcher.force_full_attention(lidx)
    print(f"Layers {protected_layers}-{num_layers-1} forced to full attention (high approximation error)")
    
    # Routers dtype should match model dtype
    model_dtype = model.dtype
    for router in patcher.routers.values():
        router.to(model_dtype)
        if args.quantize_4bit:
            router.to(torch.bfloat16)
        for param in router.parameters():
            param.requires_grad = True

    if args.profile == "A":
        print("Initializing routers with Profile A (conservative, all heads start soft-biased to full attention)...")
        initialize_routers_profile_a(patcher)
    elif args.profile == "B":
        print("Initializing routers with Profile B (uniform)...")
        for router in patcher.routers.values():
            with torch.no_grad():
                for h in range(router.num_heads):
                    router.fc2.bias.data[h*4 : (h+1)*4] = torch.zeros(4, device=router.fc2.bias.device)
    else:  # audit
        print("Initializing routers from Phase 0 priors...")
        audit_priors = load_audit_priors(args.audit_path)
        initialize_routers_from_audit(patcher, audit_priors)

    optimizer = torch.optim.AdamW(patcher.routers.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0,  # start at full LR
        end_factor=1.0,
        total_iters=1
    )
    
    # Verify router params are being optimized
    router_params = [p for p in optimizer.param_groups[0]['params'] if p.requires_grad]
    total_router_params = sum(p.numel() for p in router_params)
    print(f"Router param tensors being optimized: {len(router_params)}")
    print(f"Total router params being optimized: {total_router_params}")
    
    # Load Datasets
    print("Loading datasets...")
    stage1_dl = get_dataloader(args.stage2_data, type_filter="natural", batch_size=args.batch_size)
    stage2_dl = get_dataloader(args.stage2_data, type_filter=None, batch_size=args.batch_size)
    
    if args.smoke_test:
        args.stage1_docs = 50
        args.stage2_docs = 100
        args.stage3_docs = 20
        
    # Truncate according to args
    if args.stage1_docs > 0:
        limit = args.stage1_docs // args.batch_size
        stage1_dl = stage1_dl[:limit]
        
    if args.stage2_docs > 0:
        limit = args.stage2_docs // args.batch_size
        stage2_dl = stage2_dl[:limit]
        
    dl3 = stage2_dl
    if args.stage3_docs > 0:
        limit = args.stage3_docs // args.batch_size
        dl3 = stage2_dl[:limit]
        
    os.makedirs(args.output, exist_ok=True)
    checkpoint_path = os.path.join(args.output, "moe_checkpoint.pt")
    
    start_stage = args.start_stage
    start_epoch = 1
    
    # If starting from stage > 1, load the previous stage's checkpoint
    if start_stage > 1:
        prev_stage = start_stage - 1
        prev_checkpoint = os.path.join(args.output, f"stage{prev_stage}_moe_checkpoint.pt")
        if os.path.exists(prev_checkpoint):
            print(f"Loading from {prev_checkpoint}...")
            ckpt = torch.load(prev_checkpoint, map_location=device)
            patcher.routers.load_state_dict(ckpt["routers"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if patcher.hard_routing != ckpt.get("hard_routing", False):
                patcher.hard_routing = ckpt.get("hard_routing", False)
        else:
            print(f"WARNING: {prev_checkpoint} not found, starting from scratch")
    
    elif args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        patcher.routers.load_state_dict(ckpt["routers"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_stage = ckpt.get("stage", 1)
        start_epoch = ckpt.get("epoch", 1)
        if patcher.hard_routing != ckpt.get("hard_routing", False):
            patcher.hard_routing = ckpt.get("hard_routing", False)

    print("\nVerifying feature scales...")
    with torch.no_grad():
        sample_batch = stage1_dl[0].to(device)
        _ = model(sample_batch)
        features = patcher.last_features
        if features is not None:
            print(f"Features after LayerNorm: mean={features.mean():.3f}, std={features.std():.3f}, max={features.abs().max():.3f}")

    # STAGE 1
    if start_stage <= 1:
        print("\n--- STAGE 1: Natural Text Only (Soft Routing) ---")
        patcher.hard_routing = False
        for epoch in range(start_epoch, args.epochs + 1):
            loss = train_stage(model, patcher, optimizer, scheduler, stage1_dl, device, desc=f"Stage 1 [Epoch {epoch}]")
            print(f"Stage 1 Avg Loss: {loss:.4f}")
            torch.save({
                "stage": 1, "epoch": epoch + 1 if epoch < args.epochs else 1,
                "routers": patcher.routers.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "hard_routing": True
            }, checkpoint_path)
        # Save separate stage 1 checkpoint
        stage1_checkpoint = os.path.join(args.output, "stage1_moe_checkpoint.pt")
        torch.save({
            "stage": 1, "epoch": 1,
            "routers": patcher.routers.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "hard_routing": False
        }, stage1_checkpoint)
        # Also save stage1 routers.pt
        stage1_routers = os.path.join(args.output, "stage1_routers.pt")
        torch.save({name: router.state_dict() for name, router in patcher.routers.items()}, stage1_routers)
        start_epoch = 1
        
    # STAGE 2
    if start_stage <= 2:
        print("\n--- STAGE 2: Mixed Text (Regime Switching) ---")
        for epoch in range(start_epoch, args.epochs + 1):
            loss = train_stage(model, patcher, optimizer, scheduler, stage2_dl, device, desc=f"Stage 2 [Epoch {epoch}]")
            print(f"Stage 2 Avg Loss: {loss:.4f}")
            torch.save({
                "stage": 2, "epoch": epoch + 1 if epoch < args.epochs else 1,
                "routers": patcher.routers.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "hard_routing": False
            }, checkpoint_path)
        # Save separate stage 2 checkpoint
        stage2_checkpoint = os.path.join(args.output, "stage2_moe_checkpoint.pt")
        torch.save({
            "stage": 2, "epoch": 1,
            "routers": patcher.routers.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "hard_routing": False
        }, stage2_checkpoint)
        # Also save stage2 routers.pt
        stage2_routers = os.path.join(args.output, "stage2_routers.pt")
        torch.save({name: router.state_dict() for name, router in patcher.routers.items()}, stage2_routers)
        start_epoch = 1
        
    # Evaluate Soft Routing
    if start_stage <= 2:
        log_routing_decisions(model, patcher, stage2_dl[:10], device)
        
    # STAGE 3
    if start_stage <= 3 and (args.stop_at_stage is None or args.stop_at_stage >= 3):
        print("\n--- STAGE 3: Hard Routing Fine-Tuning (STE) ---")
        patcher.hard_routing = True
        optimizer = torch.optim.AdamW(patcher.routers.parameters(), lr=1e-5) # Lower LR for STE
        scheduler = None # No warmup for stage 3
        if args.resume and start_stage == 3 and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            optimizer.load_state_dict(ckpt["optimizer"])
            
        for epoch in range(start_epoch, args.epochs + 1):
            loss = train_stage(model, patcher, optimizer, scheduler, dl3, device, desc=f"Stage 3 [Epoch {epoch}]")
            print(f"Stage 3 Avg Loss: {loss:.4f}")
            torch.save({
                "stage": 3, "epoch": epoch + 1 if epoch < args.epochs else 1,
                "routers": patcher.routers.state_dict(),
                "optimizer": optimizer.state_dict(),
                "hard_routing": True
            }, checkpoint_path)
        # Save separate stage 3 checkpoint
        stage3_checkpoint = os.path.join(args.output, "stage3_moe_checkpoint.pt")
        torch.save({
            "stage": 3, "epoch": 1,
            "routers": patcher.routers.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hard_routing": True
        }, stage3_checkpoint)
        # Also save stage3 routers.pt
        stage3_routers = os.path.join(args.output, "stage3_routers.pt")
        torch.save({name: router.state_dict() for name, router in patcher.routers.items()}, stage3_routers)
        
    # Evaluate Hard Routing
    print("\nFinal Hard Routing Statistics:")
    log_routing_decisions(model, patcher, stage2_dl[:10], device)
    
    # Save Router Weights
    final_path = os.path.join(args.output, "routers.pt")
    router_state = {name: router.state_dict() for name, router in patcher.routers.items()}
    torch.save(router_state, final_path)
    print(f"Router weights saved to {final_path}")

if __name__ == "__main__":
    main()
