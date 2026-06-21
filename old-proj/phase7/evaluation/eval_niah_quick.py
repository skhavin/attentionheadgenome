import os
import sys
import argparse
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def eval_niah_quick(model, patcher, seq_len=512, num_samples=50, seed=42):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    rng = random.Random(seed)
    correct = 0
    device = next(model.parameters()).device
    
    # Temporarily set hard_routing = True for evaluation if patcher is present
    old_hard = getattr(patcher, "hard_routing", False)
    if patcher is not None:
        patcher.hard_routing = True
    
    for i in range(num_samples):
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
        is_correct = sample["answer"].lower() in pred.lower()
        if is_correct:
            correct += 1
        if i < 5:
            print(f"  [Sample {i+1}] Answer: {sample['answer']} | Pred: '{pred}' | Correct: {is_correct}")
            
    if patcher is not None:
        patcher.hard_routing = old_hard
    return correct / num_samples if num_samples else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--checkpoint_dir", default="checkpoints/latest-qwen-fixed")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if "llama" in args.model.lower() or "qwen" in args.model.lower() else torch.float32,
        attn_implementation="eager"
    )
    model.name_or_path = args.model

    # 1. Baseline Full Attention
    print("\nEvaluating Baseline (Full Attention)...")
    baseline_acc = eval_niah_quick(model, None, seq_len=args.seq_len, num_samples=args.num_samples)
    print(f"==> Baseline (Full Attention) Accuracy: {baseline_acc:.1%}\n")

    # 2. Stage 2 MoE Router
    stage2_path = os.path.join(args.checkpoint_dir, "stage2_routers.pt")
    if os.path.exists(stage2_path):
        print(f"Evaluating Stage 2 MoE Router from {stage2_path}...")
        patcher = MoEPatcher(model)
        routers_dict = torch.load(stage2_path, map_location=device)
        for name, state_dict in routers_dict.items():
            patcher.routers[name].load_state_dict(state_dict)
            
        stage2_acc = eval_niah_quick(model, patcher, seq_len=args.seq_len, num_samples=args.num_samples)
        print(f"==> Stage 2 MoE Router Accuracy: {stage2_acc:.1%}\n")
        patcher.restore()
    else:
        print(f"Stage 2 routers not found at {stage2_path}")

    # 3. Stage 3 MoE Router
    stage3_path = os.path.join(args.checkpoint_dir, "stage3_routers.pt")
    if os.path.exists(stage3_path):
        print(f"Evaluating Stage 3 MoE Router from {stage3_path}...")
        patcher = MoEPatcher(model)
        routers_dict = torch.load(stage3_path, map_location=device)
        for name, state_dict in routers_dict.items():
            patcher.routers[name].load_state_dict(state_dict)
            
        stage3_acc = eval_niah_quick(model, patcher, seq_len=args.seq_len, num_samples=args.num_samples)
        print(f"==> Stage 3 MoE Router Accuracy: {stage3_acc:.1%}\n")
        patcher.restore()
    else:
        print(f"Stage 3 routers not found at {stage3_path}")

if __name__ == "__main__":
    main()
