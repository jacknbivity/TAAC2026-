#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
# Optional MoE experiment flags:
#   --rank_moe_enable --rank_moe_num_experts 4 --rank_moe_top_k 1 --rank_moe_aux_weight 0.001
# Optional gap-bucket sequence feature:
#   --use_gap_buckets
# Optional EMA validation/checkpointing:
#   --use_ema --ema_decay 0.995 --use_ema_warmup --ema_warmup_steps 1000
# Optional EST-style asynchronous multi-epoch sparse reset:
#   --async_sparse_reset --async_sparse_reset_start_epoch 2
# Optional DIN-style per-token sequence reweighting (applied before the query
# generator and HyFormer blocks; independent of --seqContextwakeup):
#   --enable_seq_reweight [--seq_reweight_hidden_mult 2]
# Optional per-(block, sequence) target-aware additive bias injected into
# HyFormer cross-attention. Target = mean-pooled item NS tokens; per-(block,
# domain) scalar alpha starts at 0 so init equals vanilla cross-attention:
#   --bias_attention_up [--target_attn_bias_dropout 0.0]
# Optional LMF gating residual branch (reads the LMF user-CVR embedding slice
# of user_dense_feats and adds a small residual to the pre-classifier output;
# does not change num_ns or the d_model % T == 0 constraint):
#   --use_lmf_gating_branch
# Optional SWA-on-EMA fusion. At end of training, weighted-average the top-K
# best EMA-validated epochs (sparse Embeddings are fused too because each
# .ema_model checkpoint already stores them alongside the EMA dense shadow).
# Default weights 0.6 / 0.4 favour the best EMA epoch over the runner-up.
# Requires --use_ema. Output: <ckpt_dir>/swa_on_ema_model/model.pt
#   --use_swa_on_ema [--swa_top_k 2] [--swa_weights 0.6,0.4]
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_epochs 5\
    --enable_intraday_calendar_features \
    --enable_weekly_calendar_features \
    --enable_annual_calendar_features \
    --enable_history_time_bias \
    --seqContextwakeup \
    --num_workers 8 \
    --use_ema \
    --ema_decay 0.999 \
    --async_sparse_reset \
    --async_sparse_reset_start_epoch 2 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --enable_intraday_calendar_features \
#     --enable_weekly_calendar_features \
#     --enable_annual_calendar_features \
#     --seqContextwakeup \
#     --bias_attention_up \
#     --rank_moe_enable \
#     --rank_moe_num_experts 4 \
#     --rank_moe_top_k 1 \
#     --rank_moe_aux_weight 0.001 \
#     --num_workers 8 \
#     --use_ema \
#     --use_swa_on_ema \
#     --swa_top_k 2 \
#     --swa_weights 0.6,0.4 \
#     "$@"
