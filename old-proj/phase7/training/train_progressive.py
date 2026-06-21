import os
import sys
import json
import torch
import math
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from phase7.moe.moe_patcher import MoEPatcher
from phase7.training.losses import routing_loss

def get_dataloader(data_path, type_filter=None, batch_size=2):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    if type_filter:
        data = [d for d in data if d["type"] == type_filter]
    
    batches = []
    for i in range(0, len(data), batch_size):
        batch_items = data[i:i+batch_size]
        max_len = max(len(item["tokens"]) for item in batch_items)
        tensors = []
        for item in batch_items:
            t = item["tokens"]
            if len(t) < max_len:
                t = t + [50256] * (max_len - len(t))
            tensors.append(torch.tensor(t))
        batches.append(torch.stack(tensors))
    return batches

def eval_ppl_quick(model, patcher, device, num_chunks=20):
    model.eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    encodings = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    
    max_length = min(model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 1024, 1024)
    stride = 512
    seq_len_total = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    num_evals = min(num_chunks, (seq_len_total - 1) // stride + 1)
    
    for begin_loc in range(0, seq_len_total, stride):
        if len(nlls) >= num_evals:
            break
        end_loc = min(begin_loc + max_length, seq_len_total)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len_total:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

import random

FILLER_SENTENCE = (
    "The researchers continued their investigation into the properties of "
    "various materials under controlled laboratory conditions. "
)

def _make_filler(tokenizer, target_tokens):
    reps = target_tokens // len(tokenizer.encode(FILLER_SENTENCE)) + 2
    text = FILLER_SENTENCE * reps
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids)

def make_niah_sample(tokenizer, seq_len, rng):
    fact = rng.randint(10000, 99999)
    needle = f" The magic code is {fact}. "
    question = f" What is the magic code? The magic code is"

    needle_ids   = len(tokenizer.encode(needle,   add_special_tokens=False))
    question_ids = len(tokenizer.encode(question, add_special_tokens=False))
    filler_budget = seq_len - needle_ids - question_ids - 10

    frac = rng.uniform(0.1, 0.9)
    pre_filler  = _make_filler(tokenizer, int(filler_budget * frac))
    post_filler = _make_filler(tokenizer, filler_budget - int(filler_budget * frac))

    text = pre_filler + needle + post_filler + question
    return {"text": text, "answer": str(fact)}

def eval_niah_quick(model, patcher, seq_len=512, num_samples=10, seed=42):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    rng = random.Random(seed)
    correct = 0
    device = next(model.parameters()).device
    
    # Temporarily set hard_routing = True for evaluation
    old_hard = patcher.hard_routing
    patcher.hard_routing = True
    
    for _ in range(num_samples):
        sample = make_niah_sample(tokenizer, seq_len, rng)
        inputs = tokenizer(sample["text"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if sample["answer"].lower() in pred.lower():
            correct += 1
            
    patcher.hard_routing = old_hard
    return correct / num_samples if num_samples else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--stage2_data", default="data/stage2_gpt2_full.jsonl")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output", default="checkpoints/gpt2-medium-progressive/")
    parser.add_argument("--quantize_4bit", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            attn_implementation="eager"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if "llama" in args.model.lower() or "qwen" in args.model.lower() else torch.float32,
            attn_implementation="eager"
        )
    model.name_or_path = args.model

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    print("Installing routers...")
    patcher = MoEPatcher(model)
    model_dtype = model.dtype
    for router in patcher.routers.values():
        router.to(model_dtype)
        if args.quantize_4bit:
            router.to(torch.bfloat16)

    # Initialize all routers to zeros fallback (uniform)
    for name, router in patcher.routers.items():
        for param in router.parameters():
            param.requires_grad = True

    print("Loading Stage 2 training dataset...")
    # Use first 100 documents for progressive layer distillation
    train_dl = get_dataloader(args.stage2_data, type_filter=None, batch_size=args.batch_size)[:50]

    num_layers = len(patcher.routers)
    print(f"Starting progressive training on {num_layers} layers...")

    # Establish baseline ppl (force all layers to full attention by setting active_routing_layer to 9999)
    patcher.active_routing_layer = 9999
    baseline_ppl = eval_ppl_quick(model, patcher, device, num_chunks=20)
    print(f"Initial Baseline PPL: {baseline_ppl:.2f}")

    os.makedirs(args.output, exist_ok=True)

    MAX_RETRIES = 3
    for target_layer in range(num_layers):
        print(f"\n================ Training Layer {target_layer} ================")
        
        # Step 1: Freeze all routers except target layer
        for name, router in patcher.routers.items():
            layer_idx = int(name[1:])
            for param in router.parameters():
                param.requires_grad = (layer_idx == target_layer)
        
        # Step 2: Set target layer to active routing
        patcher.set_single_layer_routing(target_layer)
        
        # We will retry training with more batches/epochs if validation fails
        success = False
        
        for retry in range(MAX_RETRIES):
            print(f"--- Attempt {retry + 1}/{MAX_RETRIES} for Layer {target_layer} ---")
            
            # Dynamic budget combining depth scaling, retry relaxation, and remaining layers ratio
            base_delta = 2.8
            depth_factor = 1.0 + (target_layer / num_layers)  # deeper layer = larger budget delta
            attempt_factor = retry * 0.5                      # each retry loosens budget by 0.5 PPL
            locked_ratio = len(patcher.locked_layers) / num_layers
            
            max_budget = baseline_ppl + (base_delta * depth_factor) + attempt_factor + locked_ratio
            print(f"Dynamic budget ceiling for Layer {target_layer} (Attempt {retry+1}): {max_budget:.2f}")

            patcher.training = True
            
            optimizer = torch.optim.AdamW(
                patcher.routers[f"l{target_layer}"].parameters(), 
                lr=1e-4
            )
            
            # If retry > 0, train longer (e.g. duplicate the dataloader)
            current_train_dl = train_dl
            if retry > 0:
                current_train_dl = train_dl * (retry + 1)
                
            progress = tqdm(current_train_dl, desc=f"L{target_layer} Training (Attempt {retry+1})")
            for batch_idx, batch in enumerate(progress):
                input_ids = batch.to(device)
                patcher.reset_loss()
                
                # Forward pass
                _ = model(input_ids)
                mse_loss = patcher.accumulated_mse
                
                probs = patcher.routers[f"l{target_layer}"]._last_probs
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
                
                # Use high entropy pressure (0.10) to push toward cheap paths
                loss = mse_loss - 0.10 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    patcher.routers[f"l{target_layer}"].parameters(), 
                    1.0
                )
                optimizer.step()
                progress.set_postfix(loss=loss.item())

            patcher.training = False

            # Lock this layer to hard routing for evaluation
            patcher.lock_layer(target_layer)
            
            # Measure cumulative PPL with all locked layers so far
            ppl = eval_ppl_quick(model, patcher, device, num_chunks=20)
            print(f"Cumulative PPL after locking layer {target_layer} (Attempt {retry+1}): {ppl:.2f}")
            
            if ppl <= max_budget:
                # Check NIAH score if we are at a checkpoint boundary
                if (target_layer + 1) % 4 == 0:
                    quick_niah_score = eval_niah_quick(model, patcher, seq_len=512)
                    print(f"--- Quick NIAH Score after locking 0-{target_layer} (Attempt {retry+1}): {quick_niah_score:.1%} ---")
                    if quick_niah_score >= 0.50:
                        success = True
                        break
                    else:
                        print(f"Layer {target_layer} failed Quick NIAH ({quick_niah_score:.1%} < 50.0%)")
                else:
                    success = True
                    break
            else:
                print(f"Layer {target_layer} exceeded PPL budget ({ppl:.2f} > {max_budget:.2f})")
            
            # If we didn't succeed, unlock layer before next attempt
            if target_layer in patcher.locked_layers:
                patcher.locked_layers.remove(target_layer)

        if success:
            with torch.no_grad():
                sample_batch = train_dl[0].to(device)
                _ = model(sample_batch)
                decisions = patcher.routers[f"l{target_layer}"]._last_probs.argmax(dim=-1)
                cheap = (decisions < 3).float().mean().item()
                print(f"Layer {target_layer} locked successfully! Cheap path usage: {cheap*100:.1f}%")
        else:
            print(f"Layer {target_layer} exhausted all retries -> keeping FULL ATTENTION")
            patcher.force_full_attention(target_layer)

    # Save final router weights
    final_path = os.path.join(args.output, "routers.pt")
    router_state = {name: router.state_dict() for name, router in patcher.routers.items()}
    torch.save(router_state, final_path)
    print(f"\nProgressive routers saved successfully to {final_path}")

    # Final full perplexity verification
    print("\nRunning final hard routing perplexity evaluation...")
    patcher.active_routing_layer = -1
    final_ppl = eval_ppl_quick(model, patcher, device, num_chunks=200)
    print(f"Final 200-chunk Hard Routing PPL: {final_ppl:.2f}")

if __name__ == "__main__":
    main()
