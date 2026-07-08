# Full Head Genome Record (Feature Schema & Triage Plan)

## 1. The Schema (200+ Feature List)
The master idea:
* Q/K tells WHERE the head reads.
* V/OV tells WHAT it writes.
* Residual attribution tells WHO built it.
* Attention distribution tells HOW it selects.
* Ablation tells WHETHER it matters.
* Routing sweep tells WHAT policy it can safely use.

### 1. Identity / architecture metadata
`model_name`, `model_family`, `parameter_count`, `layer_index`, `head_index`, `relative_depth`, `num_layers`, `num_attention_heads`, `num_kv_heads`, `head_dim`, `hidden_dim`, `mlp_dim`, `gqa_group_id`, `heads_per_kv_group`, `attention_type`, `rope_theta`, `rope_scaling`, `max_context_length`, `tokenizer_name`, `tokenizer_vocab_size`, `base_or_instruct`

### 2. Static projection weights
`W_Q_norm`, `W_K_norm`, `W_V_norm`, `W_O_norm`, `W_Q_frobenius_norm`, `W_K_frobenius_norm`, `W_V_frobenius_norm`, `W_O_frobenius_norm`, `W_Q_spectral_norm`, `W_K_spectral_norm`, `W_V_spectral_norm`, `W_O_spectral_norm`, `Q_to_K_norm_ratio`, `V_to_Q_norm_ratio`, `V_to_K_norm_ratio`, `O_to_V_norm_ratio`, `W_Q_mean`, `W_K_mean`, `W_V_mean`, `W_O_mean`, `W_Q_std`, `W_K_std`, `W_V_std`, `W_O_std`, `W_Q_sparsity`, `W_K_sparsity`, `W_V_sparsity`, `W_O_sparsity`

### 3. QK read circuit geometry
`QK_matrix_norm`, `QK_frobenius_norm`, `QK_spectral_norm`, `QK_effective_rank`, `QK_true_rank`, `QK_condition_number`, `QK_top_1_singular_value`, `QK_top_2_singular_value`, `QK_top_3_singular_value`, `QK_singular_value_entropy`, `QK_anisotropy`, `QK_trace`, `QK_determinant_if_square`, `QK_mean`, `QK_std`, `QK_max`, `QK_min`, `QK_skewness`, `QK_kurtosis`

### 4. OV write circuit geometry
`OV_matrix_norm`, `OV_frobenius_norm`, `OV_spectral_norm`, `OV_effective_rank`, `OV_true_rank`, `OV_condition_number`, `OV_top_1_singular_value`, `OV_top_2_singular_value`, `OV_top_3_singular_value`, `OV_singular_value_entropy`, `OV_anisotropy`, `OV_trace`, `OV_mean`, `OV_std`, `OV_max`, `OV_min`, `OV_skewness`, `OV_kurtosis`, `OV_alignment_with_embedding`, `OV_alignment_with_unembedding`, `OV_alignment_with_residual_stream`, `OV_alignment_with_copy_direction`, `OV_alignment_with_entity_direction`, `OV_alignment_with_previous_token_direction`

### 5. Component attribution
For Q, K, V, O separately: `embed_pct`, `top_layer`, `top_component`, `attribution_entropy`, `layer_centroid`, `layer_variance`, `depth_asymmetry`
Also record source type: `from_embedding_pct`, `from_attention_pct`, `from_mlp_pct`

### 6. Raw Q/K/V activation statistics
`activation_norm_mean/std/max/min`, `activation_anisotropy`, `token_variance`, `prompt_variance`, `Q_K_cosine_mean`, `Q_K_cosine_std`, `Q_K_cosine_top_token`, `Q_K_cosine_target_token`

### 7. Pre-softmax score behavior
`qk_score_mean`, `qk_score_std`, `qk_score_max`, `qk_score_min`, `qk_score_top1`, `qk_score_top2`, `qk_score_top5_mean`, `qk_score_top10_mean`, `qk_top1_minus_top2`, `qk_top1_minus_top5_mean`, `qk_top1_minus_mean`, `qk_top1_zscore`, `qk_score_entropy_before_softmax`, `qk_score_gini`, `qk_score_kurtosis`, `qk_score_noise_floor`, `qk_false_spike_rate`, `qk_true_target_margin`, `absolute_threshold_false_positive_rate`, `early_exit_wrong_hit_rate`, `first_hit_distance`, `true_target_rank_by_qk_score`, `true_target_score_percentile`

### 8. Softmax attention distribution
`attention_entropy`, `attention_entropy_mean`, `attention_entropy_std`, `attention_entropy_delta`, `attention_gini`, `attention_top1_mass`, `attention_top2_mass`, `attention_top5_mass`, `attention_top10_mass`, `attention_top1_token`, `attention_top1_token_type`, `attention_top1_distance`, `attention_top5_distance_mean`, `attention_concentration_ratio`, `attention_mass_inside_top_k`

### 9. Positional / distance behavior
`mean_attention_distance`, `median_attention_distance`, `p90/p95/max`, `local_mass_X`, `long_range_mass_X`, `distance_decay_slope/r2`, `exponential_decay_fit`, `power_law_decay_fit`, `distance_entropy`, `position_invariance_score`, `content_invariance_score`, `rope_phase_sensitivity`, `rope_distance_sensitivity`

### 10. Sink behavior
`bos_mass`, `first_token_mass`, `first_X_token_mass`, `punctuation_mass`, `period_mass`, `comma_mass`, `newline_mass`, `space_mass`, `delimiter_mass`, `quote_mass`, `bracket_mass`, `sink_mass_stability`, `sink_token_rank`, `sink_token_qk_score`, `sink_token_attention_margin`, `sink_dependency_for_ppl`, `sink_dependency_for_niah`

### 11. Local behavior
`local_window_needed_for_95_mass`, `local_window_needed_for_99_mass`, `best_lossless_local_window`, `attention_decay_half_life`, `nearby_function_word_mass`, `nearby_syntax_mass`, `previous_token_mass`, `previous_X_token_mass`, `next_to_previous_token_mass`, `locality_score`, `locality_stability`

### 12. Retrieval behavior
`needle_attention_mass`, `needle_token_rank`, `needle_qk_score`, `needle_qk_margin`, `needle_distance`, `needle_retrieval_success`, `entity_attention_mass`, `proper_noun_attention_mass`, `rare_token_attention_mass`, `number_attention_mass`, `identifier_attention_mass`, `quoted_string_attention_mass`, `capitalized_token_attention_mass`, `long_range_entity_mass`, `query_to_answer_attention`, `question_to_context_attention`, `relation_token_attention`, `attribute_token_attention`, `fact_value_attention`, `retrieval_head_precision`, `retrieval_head_recall`, `retrieval_stability_across_needles/depths/templates`

### 13. Induction / copy behavior
`previous_occurrence_attention_mass`, `next_after_previous_occurrence_mass`, `AB_A_to_B_score`, `duplicate_sequence_score`, `repeat_token_attention_mass`, `copy_score`, `induction_score`, `induction_target_rank`, `induction_qk_margin`, `delimiter_copy_score`, `variable_name_copy_score`, `bracket_pair_copy_score`, `quote_continuation_score`

### 14. Token-type specialization
Attention mass by token class: `bos`, `eos`, `punctuation`, `newline`, `space`, `delimiter`, `function_words`, `content_words`, `proper_nouns`, `common_nouns`, `verbs`, `relational_verbs`, `adjectives`, `numbers`, `dates`, `rare_tokens`, `subword_prefixes/suffixes`, `capitalized_words`, `identifiers`, `code_tokens`, `quotes`, `brackets`, `math_symbols`, `repeated_tokens`, `unique_tokens`, `high/low_frequency_tokens`

### 15. Prompt/task stability
`class_consistency_across_prompts`, `attention_pattern_cosine_across_prompts`, `attention_entropy_variance_across_prompts`, `distance_profile_variance`, `token_type_profile_variance`, `qk_margin_variance`, `output_direction_variance`, `role_switch_rate`, `local_to_retrieval_switch_rate`, `sink_to_local_switch_rate`, `retrieval_activation_frequency`

### 16. V vector behavior
`V_token_norm_mean/std`, `V_target/sink/entity/local_token_norm`, `V_information_content_score`, `V_alignment_with_token_identity/position/entity_features/syntax_features/copy_features`

### 17. Head output behavior
`head_output_norm_mean/std/max`, `head_output_token/prompt_variance`, `residual_update_norm`, `residual_update_cosine_to_input/final_residual`, `residual_update_rank`, `residual_update_entropy`, `output_direction_stability`, `output_token/class_specificity`

### 18. Logit lens / unembedding effect
`head_direct_logit_effect`, `head_logit_lens_top_token`, `head_logit_lens_entropy`, `head_boosts_correct_token`, `head_suppresses_wrong_token`, `head_changes_answer_rank/needle_rank/copy_token_rank`

### 19. Circuit composition
`upstream_head/mlp_dependency_score`, `which_previous_heads_build_Q/K/V`, `which_previous_mlps_build_Q/K/V`, `Q/K/V_composition_from_previous_heads`, `OV_composition_to_later_heads`, `downstream_heads/mlps_using_this_output`, `path_patching_effect`, `activation_patching_effect`, `causal_trace_score`

### 20. Ablation / replacement features
`delta_ppl_when_zeroed/windowed_32/64/128/256/512`, `delta_niah_when_zeroed/windowed_X`, `delta_copy/induction/entity/passkey/long_context`, `delta_when_qk_randomized`, `delta_when_ov_randomized`, `delta_when_v_randomized`, `delta_when_attention_pattern_frozen/replaced_local/full`, `delta_when_sink_preserved/removed`, `delta_when_entity/repeat_cache_preserved`

### 21. Routing response
`safe_to_window_X`, `best_lossless_window`, `needs_dense_attention`, `needs_bos/first_4_preservation`, `needs_entity/repeat/delimiter/rare_token_cache`, `ppl/niah/speedup/memory_reduction_under_best_policy`

### 22. Dynamic runtime policy features
`current_qk_margin`, `current_attention_entropy`, `current_local_mass_estimate`, `current_sink_score`, `current_entity/repeat/long_range_candidate_score`, `current_uncertainty_score`, `current_dense_fallback_probability`, `current_safe_prune_probability`

### 23. Cross-model normalization features
`raw/z/percentile_embed_k_pct`, `raw/relative/percentile_q_layer`, `raw/relative/percentile_k_layer`, `raw/z/percentile_ov_norm`, `raw/z/percentile_qk_margin`

### 24. Final label/policy targets
`canonical/mechanistic/activation_probe/causal/best_policy_label`, `is_sink/local/retrieval/induction/hybrid/prompt_dependent`, `optimal_attention_policy/window_size`, `requires_dense_for_ppl/niah`, `requires_sink_for_stability`

---

## 2. Triage & Execution Plan (The "Anti-Circularity" Protocol)

### Feature Triage
Before extracting all 200+ features, we MUST divide them into three bins to avoid circularity (rediscovering labels using tautological metrics):
1. **Causally upstream of function**: Static weights, QK/OV geometry, composition scores, ablation deltas. (Legitimate candidates).
2. **Behaviorally correlated but independently computed**: Raw distance profiles, token-type mass, position-vs-content shuffle survival. (Legitimate but requires identity-checks).
3. **Restatements of the label itself**: Anything with "score," "precision," "recall," or "success" derived from the same probes used to assign the label (e.g., `retrieval_head_precision`, `induction_score`). **Exclude these by default** to avoid tautologies.

### Core Ablation Priority (Action Items)
1. **QK-only vs OV-only Ablation**: Randomize/zero attention pattern while preserving OV, and vice versa. Tests the "where it reads vs. what it writes" mechanistic split.
2. **Content-Shuffle vs. Position-Shuffle Survival**: Non-circular validation (using the permutation-null harness from Phase 11).
3. **Upstream Composition Ablation**: Ablate previous-token heads and MLPs to check if downstream head classes change. (Expensive, run last).

### Methodology Restraints
- **Held-Out Prompts**: 20% of the prompt pool MUST be held out during feature extraction so we don't accidentally overfit to the exact same text used to generate the canonical labels.
- **Nested Cross-Validation**: If we train a router or find a simple rule, feature selection must happen on an inner LOAO (Leave-One-Architecture-Out) fold and evaluated on an outer fold.
- **Rare Class Disclaimer**: Retrieval (n=23) and Sink (n=28) are too rare to claim a massive statistical generalization without stringent nested-CV. Any "universal formula" primarily applies to Local/Induction heads unless proven otherwise.
