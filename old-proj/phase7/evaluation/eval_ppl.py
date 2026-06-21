import os
import sys
import torch
import math
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from phase7.moe.moe_patcher import MoEPatcher

def evaluate_ppl(model, tokenizer, device, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="test", num_chunks=200, seq_len=1024):
    model.eval()
    
    print(f"Loading dataset {dataset_name} ({dataset_config}) ...")
    ds = load_dataset(dataset_name, dataset_config, split=split)
    encodings = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    
    max_length = model.config.max_position_embeddings
    if hasattr(model.config, "n_positions"):
        max_length = model.config.n_positions
    max_length = min(max_length, 2048) # Bound evaluation chunk size to prevent extreme sequence length hangs
    
    stride = 512
    seq_len_total = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    num_evals = min(num_chunks, (seq_len_total - 1) // stride + 1)
    
    for begin_loc in tqdm(range(0, seq_len_total, stride), total=num_evals, desc="Evaluating PPL"):
        if len(nlls) >= num_evals:
            break
            
        end_loc = min(begin_loc + max_length, seq_len_total)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # Ignore past tokens for loss calculation

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # NLL is just loss * trg_len
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len_total:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--mode", choices=["full", "hard_routing", "soft_routing"], default="hard_routing")
    parser.add_argument("--checkpoint", default="outputs/phase7/routers/gpt2_routers.pt")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Which stage checkpoint to load (stage1, stage2, stage3)")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--chunks", type=int, default=200)
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
            torch_dtype=torch.bfloat16 if "llama" in args.model.lower() or "qwen" in args.model.lower() else torch.float32,
            attn_implementation="eager"
        )
        
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    patcher = None
    if args.mode in ["hard_routing", "soft_routing"]:
        print(f"Installing MoEPatcher (Hard Routing = {args.mode == 'hard_routing'})...")
        patcher = MoEPatcher(model, hard_routing=(args.mode == 'hard_routing'))
        
        # Load weights
        router_path = args.checkpoint
        if os.path.isdir(router_path):
            if args.stage:
                router_path = os.path.join(router_path, f"stage{args.stage}_routers.pt")
            else:
                router_path = os.path.join(router_path, "routers.pt")
            
        if not os.path.exists(router_path):
            print(f"Warning: Router weights not found at {router_path}. Using uninitialized/prior weights.")
        else:
            state_dict = torch.load(router_path, map_location=device)
            # state_dict has keys like 'l0': { ... }, 'l1': { ... }
            # Wait! We saved it as `router.state_dict()`. Let's restore carefully.
            for name, router in patcher.routers.items():
                if name in state_dict:
                    router.load_state_dict(state_dict[name])
            print(f"Router weights loaded successfully from {router_path}.")
            
        model_dtype = model.dtype
        for router in patcher.routers.values():
            router.to(model_dtype)
            
    ppl = evaluate_ppl(
        model, 
        tokenizer, 
        device, 
        dataset_name="Salesforce/wikitext" if args.dataset == "wikitext" else args.dataset,
        num_chunks=args.chunks
    )
    
    print(f"\n======================================")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"PPL: {ppl:.4f}")
    if patcher is not None:
        stats = patcher.get_activation_stats()
        print(f"Path Activations: Sink: {stats[0]:.2f}%, Local: {stats[1]:.2f}%, Recurrence: {stats[2]:.2f}%, Full: {stats[3]:.2f}%")
        cheap_paths = stats[0] + stats[1] + stats[2]
        print(f"Total Cheap Paths: {cheap_paths:.2f}%")
    print(f"======================================\n")

if __name__ == "__main__":
    main()
