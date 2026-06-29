import os
import json
import torch
import numpy as np
from collections import defaultdict, Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

MODELS = {
    "GPT-2": "gpt2-medium",
    "Qwen-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen-1.5B": "Qwen/Qwen2.5-1.5B",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B"
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "outputs/phase9_semantics"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_SEQS = 50
SEQ_LEN = 128

def clean_token(t):
    """Normalize token strings (remove GPT-2 and SentencePiece artifacts)"""
    t = str(t).replace("Ġ", "").replace(" ", "").strip()
    return t if t else "[SPACE]"

def load_canonical_labels():
    with open("outputs/canonical_labels.json", "r") as f:
        return json.load(f)

def run_audit_for_model(model_name, hf_id, text_corpus, labels_data):
    print(f"\n--- Auditing {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    
    # Use float16 to save memory
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, 
        attn_implementation="eager", 
        torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    )
    model.eval().to(DEVICE)
    
    # Get model's canonical labels
    head_labels = {}
    if model_name in labels_data["models"]:
        head_labels = labels_data["models"][model_name]["heads"]
    
    # Store token counts per head: (layer, head) -> Counter
    head_vocab_counts = defaultdict(Counter)
    
    total_tokens_processed = 0
    
    for seq_idx in tqdm(range(NUM_SEQS), desc=f"Processing sequences"):
        text = text_corpus[seq_idx]
        
        inputs = tokenizer(text, return_tensors="pt", max_length=SEQ_LEN, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]
        
        if seq_len < 10:
            continue
            
        total_tokens_processed += seq_len
            
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        input_ids = inputs["input_ids"][0].cpu().tolist()
        tokens = [clean_token(tokenizer.decode([tid])) for tid in input_ids]
        
        # Aggregate top attended tokens
        for layer_idx, layer_attn in enumerate(outputs.attentions):
            # layer_attn is (batch=1, heads, seq, seq)
            attn = layer_attn[0].float().cpu().numpy()
            num_heads = attn.shape[0]
            
            for head_idx in range(num_heads):
                head_matrix = attn[head_idx] # (seq, seq)
                
                # For each query token (row), find the most attended key token (col)
                # Ignore self-attention (attending to the exact same current token) if possible,
                # but to be totally unbiased, we just take the pure argmax.
                max_attend_indices = np.argmax(head_matrix, axis=1)
                
                for q_idx, k_idx in enumerate(max_attend_indices):
                    # We skip the very first token to avoid edge cases
                    if q_idx == 0: continue
                    target_token = tokens[k_idx]
                    head_vocab_counts[(layer_idx, head_idx)][target_token] += 1
                    
    # Format results
    results = {}
    for (layer, head), counter in head_vocab_counts.items():
        hid_str = f"{layer}-{head}"
        
        # Get canonical label
        label = "Unknown"
        if hid_str in head_labels:
            label = head_labels[hid_str]["label"].capitalize()
            
        total = sum(counter.values())
        top_tokens = []
        for tok, count in counter.most_common(5):
            pct = (count / total) * 100 if total > 0 else 0
            top_tokens.append({"token": tok, "percentage": round(pct, 1)})
            
        results[hid_str] = {
            "layer": layer,
            "head": head,
            "label": label,
            "top_tokens": top_tokens
        }
        
    del model
    torch.cuda.empty_cache()
    
    # Save model-specific JSON
    out_path = os.path.join(OUT_DIR, f"vocab_audit_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")
    
    return results

def generate_global_html(all_model_results):
    print("\nGenerating Global HTML Report...")
    
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>HeadGenome Vocabulary Audit</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.css">
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.js"></script>
        <style>
            body { font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif; background: #0f172a; color: #f1f5f9; margin: 40px; }
            h1 { text-align: center; color: #38bdf8; font-size: 2.5em; margin-bottom: 10px; }
            p.subtitle { text-align: center; color: #94a3b8; max-width: 800px; margin: 0 auto 40px auto; line-height: 1.6; }
            .container { background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            
            table.dataTable { border-collapse: collapse !important; color: #f1f5f9; }
            table.dataTable thead th { border-bottom: 2px solid #38bdf8; color: #38bdf8; text-align: left; }
            table.dataTable tbody tr { background-color: #1e293b; border-bottom: 1px solid #334155; }
            table.dataTable tbody tr:nth-child(even) { background-color: #0f172a; }
            table.dataTable tbody tr:hover { background-color: #334155; }
            
            .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }
            .badge-sink { background-color: #dc2626; color: white; }
            .badge-local { background-color: #16a34a; color: white; }
            .badge-induction { background-color: #d97706; color: white; }
            .badge-retrieval { background-color: #2563eb; color: white; }
            .badge-unknown { background-color: #475569; color: white; }
            
            .target-tag { background: #334155; border: 1px solid #475569; padding: 2px 6px; border-radius: 4px; margin-right: 4px; font-family: monospace; font-size: 0.9em; display: inline-block; margin-bottom: 4px;}
            .target-tag span { color: #fca5a5; font-size: 0.8em; }
            
            /* DataTables Overrides for Dark Mode */
            .dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, .dataTables_wrapper .dataTables_info, .dataTables_wrapper .dataTables_processing, .dataTables_wrapper .dataTables_paginate { color: #cbd5e1; }
            .dataTables_wrapper .dataTables_paginate .paginate_button { color: #cbd5e1 !important; }
            .dataTables_wrapper .dataTables_paginate .paginate_button.current { background: #38bdf8 !important; border: none; color: #0f172a !important; }
        </style>
    </head>
    <body>
        <h1>HeadGenome: Global Vocabulary Audit</h1>
        <p class="subtitle">
            This table maps <strong>all specialized attention heads</strong> across GPT-2 and Qwen to their specific English vocabulary targets. 
            By processing thousands of natural tokens (WikiText-103), we identify the Top 5 absolute target words for every head. 
            Use the Search bar to filter by Model, Label, or specific Vocabulary Words.
        </p>
        
        <div class="container">
            <table id="auditTable" class="display" style="width:100%">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Layer</th>
                        <th>Head</th>
                        <th>Classification</th>
                        <th>Top 5 Lexical Targets (% of total attention mass)</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for model_name, results in all_model_results.items():
        for hid_str, data in results.items():
            layer = data["layer"]
            head = data["head"]
            label = data["label"]
            
            badge_class = f"badge-{label.lower()}"
            
            targets_html = ""
            for t in data["top_tokens"]:
                safe_t = t["token"].replace("<", "&lt;").replace(">", "&gt;")
                pct = t["percentage"]
                targets_html += f"<div class='target-tag'>{safe_t} <span>{pct}%</span></div>"
                
            html += f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td>{layer}</td>
                    <td>{head}</td>
                    <td><span class="badge {badge_class}">{label}</span></td>
                    <td>{targets_html}</td>
                </tr>
            """
            
    html += """
                </tbody>
            </table>
        </div>
        <script>
            $(document).ready(function() {
                $('#auditTable').DataTable({
                    "pageLength": 25,
                    "order": [[ 0, "asc" ], [ 1, "asc" ], [ 2, "asc" ]]
                });
            });
        </script>
    </body>
    </html>
    """
    
    out_path = os.path.join(OUT_DIR, "global_vocabulary_audit.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated massive searchable report at: {out_path}")


def main():
    print("Loading datasets and canonical labels...")
    labels_data = load_canonical_labels()
    
    # Load a chunk of wikitext
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    # Filter out empty lines or headers
    texts = [t for t in dataset["text"] if len(t.split()) > 30]
    
    all_model_results = {}
    
    for model_name, hf_id in MODELS.items():
        results = run_audit_for_model(model_name, hf_id, texts, labels_data)
        all_model_results[model_name] = results
        
    generate_global_html(all_model_results)

if __name__ == "__main__":
    main()
