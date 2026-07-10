import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import string
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hardcode stop words to avoid NLTK download issues
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

for p in string.punctuation:
    STOP_WORDS.add(p)
STOP_WORDS.update(["", " ", " \n", "\n"])

def classify_token(token_str, target_str):
    t_clean = token_str.strip().lower()
    tgt_clean = target_str.strip().lower()
    
    if t_clean == tgt_clean or tgt_clean in t_clean or (t_clean in tgt_clean and len(t_clean) > 2):
        return "Answer"
    
    if t_clean in STOP_WORDS or not t_clean.isalpha():
        return "Grammar"
        
    return "Concept"

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"Loading model: {model_name} on {DEVICE} for Phase 9 Generation Timeline")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.bfloat16)
    model.eval()

    with open("dataset_discovery_40.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    n_layers = model.config.num_hidden_layers
    
    first_concept_layers = []
    first_answer_layers = []
    
    # Track the macro timeline across all prompts
    timeline_matrix = {"Grammar": [], "Concept": [], "Answer": []}
    for _ in range(n_layers):
        for k in timeline_matrix:
            timeline_matrix[k].append(0)
            
    # For control (shuffled layers)
    control_concept_layers = []
    control_answer_layers = []

    for item in tqdm(prompts, desc="Evaluating Logit Lens Timeline"):
        tokens = tokenizer(item["prompt"], return_tensors="pt").to(DEVICE)
        target_token = item.get("target_full", item.get("target"))
        
        resid_stream = {}
        def resid_hook(m, args, output, l_idx):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3:
                resid_stream[l_idx] = hidden[0, -1, :].detach().clone()
            else:
                resid_stream[l_idx] = hidden[-1, :].detach().clone()
            
        handles = []
        for l in range(n_layers):
            handles.append(model.model.layers[l].register_forward_hook(
                lambda m, a, o, l_idx=l: resid_hook(m, a, o, l_idx)))
                
        with torch.no_grad():
            _ = model(**tokens)
        for h in handles: h.remove()
        
        trajectory = []
        for l in range(n_layers):
            with torch.no_grad():
                logits = model.lm_head(model.model.norm(resid_stream[l]))
                top_id = logits.argmax().item()
                top_str = tokenizer.decode(top_id)
                
            cat = classify_token(top_str, target_token)
            trajectory.append(cat)
            timeline_matrix[cat][l] += 1
            
        # Find first concept and answer
        try:
            c_layer = trajectory.index("Concept")
        except ValueError:
            c_layer = n_layers # never reached
            
        try:
            a_layer = trajectory.index("Answer")
        except ValueError:
            a_layer = n_layers # never reached
            
        first_concept_layers.append(c_layer)
        first_answer_layers.append(a_layer)
        
        # Control: shuffle the trajectory to destroy temporal ordering
        shuffled_traj = trajectory.copy()
        np.random.shuffle(shuffled_traj)
        try:
            cc_layer = shuffled_traj.index("Concept")
        except ValueError:
            cc_layer = n_layers
        try:
            ca_layer = shuffled_traj.index("Answer")
        except ValueError:
            ca_layer = n_layers
            
        control_concept_layers.append(cc_layer)
        control_answer_layers.append(ca_layer)

    print("\n=== Phase 9 Generation Timeline Results ===")
    med_concept = np.median(first_concept_layers)
    med_answer = np.median(first_answer_layers)
    print(f"True Median Layer of First 'Concept': {med_concept}")
    print(f"True Median Layer of First 'Answer':  {med_answer}")
    
    med_ctrl_concept = np.median(control_concept_layers)
    med_ctrl_answer = np.median(control_answer_layers)
    print(f"Control (Shuffled) Median First 'Concept': {med_ctrl_concept}")
    print(f"Control (Shuffled) Median First 'Answer':  {med_ctrl_answer}")
    
    print("\nTimeline Heatmap (Counts per layer across 40 prompts):")
    print("Layer | Grammar | Concept | Answer")
    print("-" * 35)
    for l in range(n_layers):
        g = timeline_matrix["Grammar"][l]
        c = timeline_matrix["Concept"][l]
        a = timeline_matrix["Answer"][l]
        print(f"L{l:02d}   |   {g:02d}    |   {c:02d}    |   {a:02d}")

if __name__ == "__main__":
    main()
