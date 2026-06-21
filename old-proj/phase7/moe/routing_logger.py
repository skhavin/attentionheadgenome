import torch
from collections import defaultdict

def log_routing_decisions(model, patcher, eval_dataloader, device):
    """
    Shows what % of tokens each head routes to each path.
    Evaluates over the provided dataloader.
    """
    routing_counts = defaultdict(lambda: [0, 0, 0, 0])
    
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
            else:
                input_ids = batch.to(device)
                
            _ = model(input_ids)
            
            for router_name, router in patcher.routers.items():
                probs = router.get_last_routing_probs()
                if probs is None:
                    continue
                # probs is [batch, num_heads, 4]
                decision = probs.argmax(dim=-1) # [batch, num_heads]
                
                # count per head
                B, H = decision.shape
                for h in range(H):
                    head_name = f"{router_name}_h{h}"
                    for path in range(4):
                        routing_counts[head_name][path] += (decision[:, h] == path).sum().item()
                    
    print(f"\n{'='*50}")
    print(f"Routing Statistics")
    print(f"{'='*50}")
    for router_name, counts in sorted(routing_counts.items()):
        total = sum(counts)
        if total == 0:
            continue
        print(f"{router_name}: sink={counts[0]/total:.1%}, "
              f"local={counts[1]/total:.1%}, rec={counts[2]/total:.1%}, full={counts[3]/total:.1%}")
    print(f"{'='*50}\n")
    return dict(routing_counts)

def main():
    import argparse
    import json
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from torch.utils.data import DataLoader
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from phase7.moe.moe_patcher import MoEPatcher
    from phase7.training.build_stage2_dataset import build_copy_trigger_prompts_simple

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--output", default="outputs/routing_stats_all_layers.json")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    if args.load_in_4bit:
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
            attn_implementation="eager"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.float16 if "llama" in args.model.lower() or "qwen" in args.model.lower() else torch.float32,
            attn_implementation="eager"
        )
        
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Installing MoEPatcher (Hard Routing = True)...")
    patcher = MoEPatcher(model, hard_routing=True)
    router_path = "outputs/phase7/routers/gpt2_routers.pt"
    if os.path.exists(router_path):
        state_dict = torch.load(router_path, map_location=device)
        for name, router in patcher.routers.items():
            if name in state_dict:
                router.load_state_dict(state_dict[name])
        print("Router weights loaded successfully.")
    else:
        print("Warning: Router weights not found. Using untrained routers.")

    if args.dataset == "induction_probes":
        print("Generating induction probes...")
        data = build_copy_trigger_prompts_simple(tokenizer, 512, 100)
    else:
        print("Loading wikitext...")
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        full_text = " ".join(row["text"] for row in ds.select(range(min(2000, len(ds)))) if row["text"].strip())
        all_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        data = []
        seq_len = 512
        for i in range(0, len(all_ids) - seq_len, seq_len):
            data.append(all_ids[i: i + seq_len].tolist())
            if len(data) >= 100:
                break
    max_len = max(len(d) for d in data)
    padded_data = [d + [tokenizer.pad_token_id] * (max_len - len(d)) for d in data]
    input_ids_tensor = torch.tensor(padded_data, dtype=torch.long)
    eval_dataloader = DataLoader(input_ids_tensor, batch_size=4)
    
    print("Evaluating routing statistics...")
    stats = log_routing_decisions(model, patcher, eval_dataloader, device)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {args.output}")

if __name__ == "__main__":
    main()
