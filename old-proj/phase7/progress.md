
# Phase 7 Progress

## Problems Faced & Solutions

### 1. Dtype Mismatch for Mixed-Precision Models (bfloat16)
**Problem**: When running MoE patcher on modern models like Qwen2.5-0.5B that use bfloat16 dtype, we encountered:
- `RuntimeError: expected scalar type BFloat16 but found Float`
**Root Cause**:
- The `compute_router_features` function in `phase7/moe/moe_patcher.py` was producing float32 features even when Q/K were in bfloat16
- The router parameters weren't being moved to the model's dtype after initialization
**Solution**:
- Modified `compute_router_features` to use `to(Q.dtype)` when returning features
- Added code to `MoEPatcher.__init__` to automatically move all routers to `model_dtype` (retrieved via `next(model.parameters()).dtype`)

### 2. Slow Training Speed
**Problem**: Training was taking too long per iteration
**Root Cause**:
- We were unnecessarily calling `output_attentions=True` during training, which forces eager attention even when faster SDPA is available
- We also had some unnecessary nan checks in `moe_patcher.py`
**Solution**:
- Removed `kwargs["output_attentions"] = True` from `patched_forward` in `moe_patcher.py`
- Removed the nan checking assertions in `patched_forward`
- Switched to using `attn_implementation="sdpa"` in training for faster inference

### 3. High Perplexity (PPL) for Qwen Stage 2 Checkpoint
**Problem**: Initial evaluation of Qwen2.5-0.5B stage 2 checkpoint gave a PPL of 3350, way worse than baseline
**Root Cause**:
- The Qwen training didn't use the conservative "Profile A" initialization that biases routers towards full attention (which worked well for GPT-2)
- The dtype mismatch issues likely compounded the problem during training
**Diagnostic Tests**:
- Forced only 2/14 heads per layer to use cheap paths: Baseline PPL 12.48 → Soft Routing PPL 14.67 (+3.30 PPL)
- This shows when most heads use full attention, the system works correctly

## Enhancements Made
- Added `--stop_at_stage` argument to `train_moe.py` to stop training after specific stage
- Modified `train_moe.py` to save separate checkpoints for each stage
- Added `--stage` argument to `eval_ppl.py` to load specific stage checkpoint
- Updated `initialize_layer_router_profile_a` in `router.py` to use a soft bias to full attention for all heads
- Added `--output-prefix` argument to `audit_heads.py` to save audit results with model-specific filenames
- Added `--audit-prefix` argument to `train_moe.py` to load audit results with model-specific filenames
- Copied `substitutes.py` from `archive/` to `phase7/` to make it available as a proper importable module

## Current Work
- Re-training Qwen2.5-0.5B with Profile A, using 500 stage1 docs, 1000 stage2 docs, 200 stage3 docs
- Training is running in background: `6d306386-1623-41ff-a241-7afaa7e4f9b7`
