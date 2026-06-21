# All hyperparameters in one place. Every script imports from here.

import os

# --- Model ---
MODEL_NAME = "gpt2-medium"          # 345M params, ~700MB fp16
QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"                      # change to "cpu" if no GPU
USE_FP16 = True                      # half precision to save VRAM

# --- Dataset ---
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-v1"
DATASET_SPLIT = "train"

# --- Phase 0 ---
PHASE0_SENTENCES = [
    "The cat sat on the mat and looked at the bird outside the window.",
    "In 1969, Neil Armstrong became the first person to walk on the Moon.",
    "Machine learning models learn patterns from data to make predictions.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
    "She opened the door, walked inside, and placed her bag on the table.",
    "The stock market crashed in 2008 due to the subprime mortgage crisis.",
    "Transformers use attention mechanisms to process sequences in parallel.",
    "The ancient Romans built roads that connected their vast empire together.",
    "Python is a programming language known for its simplicity and readability.",
]

# --- Phase 1 ---
NUM_PROFILING_DOCS = 500             # docs to profile
TOP_K_ATTENTION = 10                 # top-k attended positions to record
NUM_CLUSTERS = 4                     # clusters per (layer, head)
STABILITY_DOC_COUNTS = [100, 300, 500]  # for stability check
MAX_SEQ_LEN = 512                    # truncate docs to this length

# --- Phase 2 ---
PREDICTION_TEST_DOCS = 100           # docs to test prediction on
RECALL_K_VALUES = [1, 3, 5]          # recall@k to measure

# --- Phase 3 ---
KV_BUDGETS = [128, 256, 512, 1024]   # tokens to keep in KV cache
NUM_BENCHMARK_DOCS = 50              # docs for benchmarking
GENERATE_LENGTH = 50                 # tokens to generate per doc

# --- Paths ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
PHASE0_DIR = os.path.join(OUTPUT_DIR, "phase0")
PHASE1_DIR = os.path.join(OUTPUT_DIR, "phase1")
PHASE2_DIR = os.path.join(OUTPUT_DIR, "phase2")
PHASE3_DIR = os.path.join(OUTPUT_DIR, "phase3")
PHASE4_DIR = os.path.join(OUTPUT_DIR, "phase4")
PHASE5_DIR = os.path.join(OUTPUT_DIR, "phase5")
PHASE6_DIR = os.path.join(OUTPUT_DIR, "phase6")
PHASE7_DIR = os.path.join(OUTPUT_DIR, "phase7")
PROTOTYPES_PATH = os.path.join(PHASE1_DIR, "prototypes.pkl")

# --- Phase 7 ---
# Qwen 0.5B first (fast iteration), LLaMA 8B for paper numbers.
# All models assumed to be downloaded on C: or D: drive.
PHASE7_QWEN_MODELS = [
    "Qwen/Qwen2.5-0.5B",
]
PHASE7_LLAMA_MODELS = [
    "unsloth/meta-llama-3.1-8B-bnb-4bit",
    "meta-llama/Meta-Llama-3.1-8B",
]
# PPL benchmark: shorter contexts first to validate the approximation fast
PHASE7_PPL_SEQ_LENS  = [512, 1024, 2048]
# RULER benchmark: only long contexts where O(N) vs O(N²) actually matters
PHASE7_RULER_SEQ_LENS = [4096, 8192]
# Performer random feature dimension = d_head for a fair comparison
PHASE7_PERFORMER_FEATURES = 128
