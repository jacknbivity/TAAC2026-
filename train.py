"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS, NUM_GAP_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def _parse_swa_weights(raw: str) -> Optional[List[float]]:
    """Parse the ``--swa_weights`` CLI string into a ``List[float]`` (or None).

    Accepts an empty / whitespace-only string as a signal to let the trainer
    pick its own default (currently 0.6, 0.4 for top_k=2 and uniform
    otherwise). Validation (length match, non-negativity, renormalisation) is
    handled inside ``PCVRHyFormerRankingTrainer._normalise_swa_weights``.
    """
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    return [float(token) for token in cleaned.split(',') if token.strip()]


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--use_gap_buckets', action='store_true', default=False,
                        help='Enable adjacent-history gap bucket embedding (default off)')
    parser.add_argument('--no_gap_buckets', dest='use_gap_buckets', action='store_false',
                        help='Disable adjacent-history gap bucket embedding')
    parser.add_argument('--enable_intraday_calendar_features', action='store_true', default=True,
                        help='Enable minute/hour/daypart calendar features (default on)')
    parser.add_argument('--no_intraday_calendar_features',
                        dest='enable_intraday_calendar_features',
                        action='store_false',
                        help='Disable minute/hour/daypart calendar features')
    parser.add_argument('--enable_weekly_calendar_features', action='store_true', default=True,
                        help='Enable day-of-week/hour-of-week/weekend calendar features (default on)')
    parser.add_argument('--no_weekly_calendar_features',
                        dest='enable_weekly_calendar_features',
                        action='store_false',
                        help='Disable day-of-week/hour-of-week/weekend calendar features')
    parser.add_argument('--enable_annual_calendar_features', action='store_true', default=True,
                        help='Enable day/month/year calendar features (default on)')
    parser.add_argument('--no_annual_calendar_features',
                        dest='enable_annual_calendar_features',
                        action='store_false',
                        help='Disable day/month/year calendar features')
    parser.add_argument('--enable_history_time_bias',
                        dest='enable_history_time_bias',
                        action='store_true', default=False,
                        help='Inject an MLP-projected (Hour, Day, Week) residual '
                             'onto every history sequence token (variant 1 of the '
                             'Time-Aware Gated Attention scheme). Off by default.')
    parser.add_argument('--no_history_time_bias',
                        dest='enable_history_time_bias',
                        action='store_false',
                        help='Disable the historical sequence time-perception residual')
    parser.add_argument('--history_time_bias_residual_scale', type=float, default=1.0,
                        help='Scalar applied to the temporal-bias residual before it '
                             'is added back to the sequence token embedding '
                             '(only effective when --enable_history_time_bias is set)')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--rank_moe_enable', action='store_true', default=False,
                        help='Enable sparse Top-k MoE for RankMixer per-token FFN')
    parser.add_argument('--rank_moe_num_experts', type=int, default=4,
                        help='Number of experts used by RankMixer MoE')
    parser.add_argument('--rank_moe_top_k', type=int, default=1,
                        help='Number of experts selected per token by RankMixer MoE')
    parser.add_argument('--rank_moe_aux_weight', type=float, default=0.0,
                        help='Weight for RankMixer MoE load-balancing auxiliary loss')
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Enable EMA for dense trainable parameters during validation/checkpointing')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false',
                        help='Disable EMA')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='EMA decay factor for dense trainable parameters')
    parser.add_argument('--use_ema_warmup', action='store_true', default=False,
                        help='Linearly warm up EMA decay from 0 to --ema_decay')
    parser.add_argument('--no_ema_warmup', dest='use_ema_warmup',
                        action='store_false',
                        help='Disable EMA decay warmup')
    parser.add_argument('--ema_warmup_steps', type=int, default=1000,
                        help='Number of optimizer steps used for EMA decay warmup')
    parser.add_argument('--use_swa_on_ema', action='store_true', default=False,
                        help='At end of training, weighted-average the top-K '
                             'best EMA-validated epoch checkpoints into a '
                             'single SWA model written to '
                             '<ckpt_dir>/swa_on_ema_model/model.pt. Sparse '
                             'Embeddings are fused alongside dense weights '
                             'because every .ema_model checkpoint already '
                             'stores them. Requires --use_ema.')
    parser.add_argument('--no_swa_on_ema', dest='use_swa_on_ema',
                        action='store_false',
                        help='Disable SWA-on-EMA fusion at end of training')
    parser.add_argument('--swa_top_k', type=int, default=2,
                        help='Number of best EMA-validated epochs (ranked by '
                             'validation AUC) to include in the SWA-on-EMA '
                             'fusion')
    parser.add_argument('--swa_weights', type=str, default='0.6,0.4',
                        help='Comma-separated weights matching --swa_top_k. '
                             'Weight i applies to the i-th best epoch (rank 1 '
                             'is the highest val_auc). Weights are '
                             'auto-renormalised to sum to 1.0 if needed.')
    parser.add_argument('--seqContextwakeup', dest='seqContextwakeup',
                        action='store_true', default=True,
                        help='Enable target-aware sequence context wakeup (default on)')
    parser.add_argument('--no_seqContextwakeup', dest='seqContextwakeup',
                        action='store_false',
                        help='Disable target-aware sequence context wakeup')
    parser.add_argument('--enable_seq_reweight', dest='enable_seq_reweight',
                        action='store_true', default=False,
                        help='Enable DIN-style per-token reweighting of sequence tokens '
                             'BEFORE the query generator and HyFormer blocks (default off). '
                             'Independent of --seqContextwakeup: the latter is an additive '
                             'residual after HyFormer, this one modifies seq tokens in place.')
    parser.add_argument('--no_seq_reweight', dest='enable_seq_reweight',
                        action='store_false',
                        help='Disable per-token sequence reweighting')
    parser.add_argument('--seq_reweight_hidden_mult', type=int, default=2,
                        help='Hidden multiplier for the per-token reweight MLP '
                             '(only effective when --enable_seq_reweight is set)')
    parser.add_argument('--bias_attention_up', dest='bias_attention_up',
                        action='store_true', default=False,
                        help='Enable per-(block, sequence) target-aware additive bias on '
                             'HyFormer cross-attention. Bias target = mean-pooled item NS '
                             'tokens; injected via attn_score_bias before softmax. '
                             'Per-(block, domain) scalar alpha starts at 0 so the network '
                             'is initially equivalent to vanilla cross-attention.')
    parser.add_argument('--no_bias_attention_up', dest='bias_attention_up',
                        action='store_false',
                        help='Disable target-aware cross-attention bias')
    parser.add_argument('--target_attn_bias_dropout', type=float, default=0.0,
                        help='Dropout applied to the (B, L) target-aware cross-attention '
                             'bias before injection (only effective when '
                             '--bias_attention_up is set)')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # LMF gating branch (uses the LMF user-CVR embedding slice of
    # user_dense_feats to gate ns_tokens; added as a small residual to output).
    parser.add_argument('--use_lmf_gating_branch', action='store_true', default=False,
                        help='Enable the LMF gating residual branch. The branch reads a '
                             'slice of user_dense_feats (defaults to [568, 568+320)), passes '
                             'it through LayerNorm + Linear + Sigmoid to produce a per-channel '
                             'gate over ns_tokens, projects the gated tokens back to d_model, '
                             'and adds the result as a small residual to the pre-classifier '
                             'output. Does NOT modify ns_tokens or change num_ns / T.')
    parser.add_argument('--no_lmf_gating_branch', dest='use_lmf_gating_branch',
                        action='store_false',
                        help='Disable the LMF gating residual branch')
    parser.add_argument('--lmf_gating_offset', type=int, default=568,
                        help='Start column of the LMF embedding slice inside user_dense_feats '
                             '(matches UserDenseFeatureEncoder.lmf_embedding_start)')
    parser.add_argument('--lmf_gating_length', type=int, default=320,
                        help='Length of the LMF embedding slice inside user_dense_feats '
                             '(matches UserDenseFeatureEncoder.lmf_embedding_dim)')
    parser.add_argument('--lmf_gating_dropout_rate', type=float, default=0.1,
                        help='Dropout applied to the LMF slice before the gate (training only)')
    parser.add_argument('--lmf_gating_residual_scale', type=float, default=0.05,
                        help='Scalar applied to the LMF gating vector before residual addition')
    parser.add_argument('--lmf_gating_max_norm', type=float, default=100.0,
                        help='L2-norm threshold above which an LMF row is treated as an '
                             'outlier and its contribution is zeroed out')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')
    parser.add_argument('--async_sparse_reset', action='store_true', default=False,
                        help='Enable EST-style asynchronous multi-epoch training: '
                             'restore all sparse Embedding parameters to their '
                             'initial snapshot at the beginning of each configured '
                             'epoch, while dense parameters / dense optimizer / EMA '
                             'continue accumulating.')
    parser.add_argument('--no_async_sparse_reset', dest='async_sparse_reset',
                        action='store_false',
                        help='Disable asynchronous sparse reset.')
    parser.add_argument('--async_sparse_reset_start_epoch', type=int, default=2,
                        help='1-based epoch from which async sparse reset starts. '
                             'Use 2 to train epoch 1 normally, then reset sparse '
                             'parameters to their initial snapshot before every '
                             'later epoch.')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        enable_intraday_calendar_features=args.enable_intraday_calendar_features,
        enable_weekly_calendar_features=args.enable_weekly_calendar_features,
        enable_annual_calendar_features=args.enable_annual_calendar_features,
        use_gap_buckets=args.use_gap_buckets,
        enable_history_time_bias=args.enable_history_time_bias,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "num_gap_buckets": NUM_GAP_BUCKETS if args.use_gap_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "enable_intraday_calendar_features": args.enable_intraday_calendar_features,
        "enable_weekly_calendar_features": args.enable_weekly_calendar_features,
        "enable_annual_calendar_features": args.enable_annual_calendar_features,
        "seqContextwakeup": args.seqContextwakeup,
        "enable_seq_reweight": args.enable_seq_reweight,
        "seq_reweight_hidden_mult": args.seq_reweight_hidden_mult,
        "bias_attention_up": args.bias_attention_up,
        "target_attn_bias_dropout": args.target_attn_bias_dropout,
        "rank_moe_enable": args.rank_moe_enable,
        "rank_moe_num_experts": args.rank_moe_num_experts,
        "rank_moe_top_k": args.rank_moe_top_k,
        "use_lmf_gating_branch": args.use_lmf_gating_branch,
        "lmf_gating_offset": args.lmf_gating_offset,
        "lmf_gating_length": args.lmf_gating_length,
        "lmf_gating_dropout_rate": args.lmf_gating_dropout_rate,
        "lmf_gating_residual_scale": args.lmf_gating_residual_scale,
        "lmf_gating_max_norm": args.lmf_gating_max_norm,
        "enable_history_time_bias": args.enable_history_time_bias,
        "history_time_bias_residual_scale": args.history_time_bias_residual_scale,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        rank_moe_aux_weight=args.rank_moe_aux_weight,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        use_ema_warmup=args.use_ema_warmup,
        ema_warmup_steps=args.ema_warmup_steps,
        async_sparse_reset=args.async_sparse_reset,
        async_sparse_reset_start_epoch=args.async_sparse_reset_start_epoch,
        use_swa_on_ema=args.use_swa_on_ema,
        swa_top_k=args.swa_top_k,
        swa_weights=_parse_swa_weights(args.swa_weights),
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
