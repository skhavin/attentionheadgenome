import os
import sys
import json
import argparse
import random
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from phase7.moe.moe_patcher import MoEPatcher

FILLER_SENTENCE = (
    "The researchers continued their investigation into the properties of "
    "various materials under controlled laboratory conditions. "
)

def _make_filler(tokenizer, target_tokens):
    reps = target_tokens // len(tokenizer.encode(FILLER_SENTENCE)) + 2
    text = FILLER_SENTENCE * reps
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids)

def make_vt_sample(tokenizer, seq_len, rng):
    var_name = rng.choice(["alpha", "beta", "gamma", "delta", "omega", "sigma"])
    value = rng.randint(100, 999)
    assignment = f" Let {var_name} = {value}. "
    question = f" What is the value of {var_name}? {var_name} ="

    assignment_ids = len(tokenizer.encode(assignment, add_special_tokens=False))
    question_ids   = len(tokenizer.encode(question,   add_special_tokens=False))
    filler_budget  = seq_len - assignment_ids - question_ids - 10

    frac = rng.uniform(0.1, 0.9)
    pre_filler  = _make_filler(tokenizer, int(filler_budget * frac))
    post_filler = _make_filler(tokenizer, filler_budget - int(filler_budget * frac))

    text = pre_filler + assignment + post_filler + question
    return {"text": text, "answer": str(value)}

def make_cwe_sample(tokenizer, seq_len, rng):
    word_pool = ["apple", "banana", "cherry", "date", "elderberry",
                 "fig", "grape", "honeydew", "kiwi", "lemon"]
    target_word = rng.choice(word_pool)
    other_words = [w for w in word_pool if w != target_word]

    question = f" What is the most common word in the passage? The most common word is"
    question_ids = len(tokenizer.encode(question, add_special_tokens=False))
    text_budget = seq_len - question_ids - 10

    words = []
    # Build text: target word appears ~40% of the time, others split the rest
    for _ in range(text_budget // 2): 
        if rng.random() < 0.40:
            words.append(target_word)
        else:
            words.append(rng.choice(other_words))
    
    text = " ".join(words) + question
    # truncate if necessary
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > seq_len:
        text = tokenizer.decode(tokens[:seq_len - question_ids - 5]) + question
        
    return {"text": text, "answer": target_word}

def make_mq_niah_sample(tokenizer, seq_len, rng):
    num_needles = 3
    needles = []
    for i in range(num_needles):
        track_id = f"TRK-{rng.randint(1000,9999)}"
        needles.append(track_id)
        
    needle_texts = [f" Tracking number {i+1} is {n}. " for i, n in enumerate(needles)]
    question = " What are all the tracking numbers mentioned? The tracking numbers are"
    
    needle_ids = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in needle_texts)
    question_ids = len(tokenizer.encode(question, add_special_tokens=False))
    
    filler_budget = seq_len - needle_ids - question_ids - 10
    segment_budget = filler_budget // (num_needles + 1)
    
    text = ""
    for i in range(num_needles):
        text += _make_filler(tokenizer, segment_budget)
        text += needle_texts[i]
    text += _make_filler(tokenizer, segment_budget)
    text += question
    
    # We expect all needles to be present in the output
    return {"text": text, "answer": needles}

def calculate_flop_savings(stats, seq_len):
    if not stats or sum(stats) == 0:
        return 0.0
    p_sink = stats[0] / 100.0
    p_local = stats[1] / 100.0
    p_rec = stats[2] / 100.0
    p_full = stats[3] / 100.0
    
    # Average tokens attended to per query in causal self-attention is N/2
    avg_full_attended = seq_len / 2.0
    
    moe_attended = (p_sink * 4) + (p_local * 64) + (p_rec * 1) + (p_full * avg_full_attended)
    
    # Percentage reduction in attention FLOPs
    savings = (1.0 - (moe_attended / avg_full_attended)) * 100.0
    return max(0.0, savings)

def evaluate_task(model, patcher, tokenizer, task_name, seq_len, num_samples, seed=42):
    rng = random.Random(seed)
    device = next(model.parameters()).device
    correct = 0
    
    if patcher is not None:
        patcher.reset_activation_stats()
        
    for _ in tqdm(range(num_samples), desc=f"{task_name} (sl={seq_len})", leave=False):
        if task_name == "vt":
            sample = make_vt_sample(tokenizer, seq_len, rng)
        elif task_name == "cwe":
            sample = make_cwe_sample(tokenizer, seq_len, rng)
        elif task_name == "mq-niah":
            sample = make_mq_niah_sample(tokenizer, seq_len, rng)
            
        inputs = tokenizer(sample["text"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=15 if task_name != "mq-niah" else 30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()
        
        if task_name == "mq-niah":
            # For MQ-NIAH, must find all needles
            found_all = all(n.lower() in pred for n in sample["answer"])
            if found_all:
                correct += 1
        else:
            if sample["answer"].lower() in pred:
                correct += 1
                
    acc = correct / num_samples
    
    stats = [0.0, 0.0, 0.0, 100.0]
    flops_saved = 0.0
    if patcher is not None:
        stats = patcher.get_activation_stats()
        flops_saved = calculate_flop_savings(stats, seq_len)
        
    return acc, stats, flops_saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--checkpoint_dir", default="checkpoints/latest-qwen-fixed")
    parser.add_argument("--num_samples", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # GPU context lengths supported per test:
    seq_lens = [512, 1024, 2048, 4096]
    tasks = ["vt", "cwe", "mq-niah"]
    
    results = {}
    output_file = "outputs/phase7/advanced_ruler_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)

    # We evaluate Stage 2 Router
    patcher = MoEPatcher(model)
    stage2_path = os.path.join(args.checkpoint_dir, "stage2_routers.pt")
    if os.path.exists(stage2_path):
        print(f"Loading Stage 2 routers from {stage2_path}")
        routers_dict = torch.load(stage2_path, map_location=device)
        for name, state_dict in routers_dict.items():
            patcher.routers[name].load_state_dict(state_dict)
    else:
        print("Warning: Stage 2 router not found! Exiting.")
        return

    print("\n--- Running Advanced RULER Benchmarks ---")
    
    for task in tasks:
        if task not in results:
            results[task] = {}
            
        for sl in seq_lens:
            sl_str = str(sl)
            if sl_str in results[task] and "moe_acc" in results[task][sl_str]:
                print(f"Skipping {task} at {sl} (already computed).")
                continue
                
            print(f"\nEvaluating {task} at SeqLen={sl}")
            
            # Baseline
            patcher.restore()
            base_acc, _, _ = evaluate_task(model, None, tokenizer, task, sl, args.num_samples)
            
            # MoE
            patcher = MoEPatcher(model)
            for name, state_dict in routers_dict.items():
                patcher.routers[name].load_state_dict(state_dict)
            patcher.hard_routing = True
            
            moe_acc, stats, flops = evaluate_task(model, patcher, tokenizer, task, sl, args.num_samples)
            
            results[task][sl_str] = {
                "baseline_acc": base_acc,
                "moe_acc": moe_acc,
                "stats": stats, # [Sink, Local, Rec, Full]
                "flops_saved": flops
            }
            
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
                
            print(f"  Base Acc: {base_acc:.1%} | MoE Acc: {moe_acc:.1%} | FLOP Savings: {flops:.1f}%")
            print(f"  Path Act: Sink={stats[0]:.1f}%, Local={stats[1]:.1f}%, Rec={stats[2]:.1f}%, Full={stats[3]:.1f}%")

    print("\n--- Final Matrix ---")
    print(f"{'Task':<10} | {'Sink':<6} | {'Local':<6} | {'Rec':<6} | {'Full':<6} | {'Match'}")
    for task in tasks:
        # Average stats across lengths for the matrix
        avg_sink = sum(results[task][str(sl)]["stats"][0] for sl in seq_lens) / len(seq_lens)
        avg_loc = sum(results[task][str(sl)]["stats"][1] for sl in seq_lens) / len(seq_lens)
        avg_rec = sum(results[task][str(sl)]["stats"][2] for sl in seq_lens) / len(seq_lens)
        avg_full = sum(results[task][str(sl)]["stats"][3] for sl in seq_lens) / len(seq_lens)
        
        avg_base = sum(results[task][str(sl)]["baseline_acc"] for sl in seq_lens) / len(seq_lens)
        avg_moe = sum(results[task][str(sl)]["moe_acc"] for sl in seq_lens) / len(seq_lens)
        
        match_str = f"{(avg_moe - avg_base):+.1%}"
        print(f"{task:<10} | {avg_sink:>5.1f}% | {avg_loc:>5.1f}% | {avg_rec:>5.1f}% | {avg_full:>5.1f}% | {match_str}")
        
if __name__ == "__main__":
    main()
