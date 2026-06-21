"""
Diagnoses why PPL is high for MoE routers.
Checks: routing distribution, MSE vs full attention, per-path output norms.
"""
import os, sys, torch, math, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from phase7.moe.moe_patcher import MoEPatcher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--checkpoint", default="checkpoints/qwen2.5-0.5b-profile-b")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--chunks", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for p in model.parameters():
        p.requires_grad = False

    # ── 1. Baseline PPL (no router) ──────────────────────────────────────────
    print("\n[1] Measuring BASELINE PPL (full attention, no patcher)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len_total = input_ids.size(1)
    max_length = min(model.config.max_position_embeddings, 2048)
    stride = 512

    def compute_ppl(mdl, n_chunks):
        nlls, prev_end = [], 0
        for begin_loc in range(0, seq_len_total, stride):
            if len(nlls) >= n_chunks:
                break
            end_loc = min(begin_loc + max_length, seq_len_total)
            trg_len = end_loc - prev_end
            ids = input_ids[:, begin_loc:end_loc].to(device)
            tgt = ids.clone(); tgt[:, :-trg_len] = -100
            with torch.no_grad():
                out = mdl(ids, labels=tgt)
            nlls.append(out.loss)
            prev_end = end_loc
            if end_loc == seq_len_total:
                break
        return torch.exp(torch.stack(nlls).mean()).item()

    baseline_ppl = compute_ppl(model, args.chunks)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # ── 2. Install patcher and check routing distribution ────────────────────
    print("\n[2] Installing MoEPatcher (soft routing)...")
    patcher = MoEPatcher(model, hard_routing=False)

    router_path = args.checkpoint
    if os.path.isdir(router_path):
        fname = f"stage{args.stage}_routers.pt" if args.stage else "routers.pt"
        router_path = os.path.join(router_path, fname)

    if os.path.exists(router_path):
        state_dict = torch.load(router_path, map_location=device)
        for name, router in patcher.routers.items():
            if name in state_dict:
                router.load_state_dict(state_dict[name])
        print(f"  Loaded: {router_path}")
    else:
        print(f"  WARNING: {router_path} not found – using random weights")

    model_dtype = model.dtype
    for router in patcher.routers.values():
        router.to(model_dtype)

    # ── 3. Collect routing probs on a few batches ────────────────────────────
    print("\n[3] Routing distribution on first 5 chunks...")
    path_names = ["sink", "local", "recurrence", "full"]
    path_totals = torch.zeros(4)
    n_batches = 0

    for begin_loc in range(0, seq_len_total, stride):
        if n_batches >= 5:
            break
        end_loc = min(begin_loc + max_length, seq_len_total)
        ids = input_ids[:, begin_loc:end_loc].to(device)
        with torch.no_grad():
            model(ids)
        # collect last probs from all routers
        batch_probs = []
        for r in patcher.routers.values():
            if r._last_probs is not None:
                batch_probs.append(r._last_probs.float().cpu())
        if batch_probs:
            stacked = torch.stack(batch_probs)  # [L, B, H, 4]
            path_totals += stacked.mean(dim=(0, 1, 2))
            n_batches += 1

    if n_batches > 0:
        avg = path_totals / n_batches
        print(f"  Average routing weights across all layers/heads:")
        for i, name in enumerate(path_names):
            print(f"    {name:12s}: {avg[i].item():.4f}  ({avg[i].item()*100:.1f}%)")

    # ── 4. Soft-routing PPL ───────────────────────────────────────────────────
    print("\n[4] Measuring SOFT ROUTING PPL...")
    soft_ppl = compute_ppl(model, args.chunks)
    print(f"  Soft Routing PPL: {soft_ppl:.4f}")
    print(f"  Delta vs baseline: {soft_ppl - baseline_ppl:+.4f}")

    # ── 5. Force full-attention path (p_full=1) and re-measure ───────────────
    print("\n[5] Forcing all routers to full-attention (p_full=1.0)...")
    for router in patcher.routers.values():
        with torch.no_grad():
            for h in range(router.num_heads):
                # Set bias so softmax produces ~1.0 for full path
                router.fc2.bias.data[h*4 : (h+1)*4] = torch.tensor(
                    [-10.0, -10.0, -10.0, 10.0],
                    dtype=router.fc2.bias.dtype,
                    device=router.fc2.bias.device
                )

    forced_full_ppl = compute_ppl(model, args.chunks)
    print(f"  Forced-full PPL:  {forced_full_ppl:.4f}")
    print(f"  Delta vs baseline: {forced_full_ppl - baseline_ppl:+.4f}")

    if abs(forced_full_ppl - baseline_ppl) > 0.5:
        print("\n  *** PROBLEM: Even with p_full=1.0, PPL doesn't match baseline!")
        print("      This means the patcher itself is broken (o_proj is being called twice")
        print("      or combined_out is being passed through o_proj when full_out already has it).")
    else:
        print("\n  Patcher overhead is acceptable at p_full=1.0.")
        print("  High PPL is due to routing decisions sending weight to bad paths.")

    print("\nDone.")

if __name__ == "__main__":
    main()
