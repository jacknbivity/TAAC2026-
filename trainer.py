"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import EarlyStopping
from model import ModelInput
from dataset import _HIST_TIME_FIELDS


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        rank_moe_aux_weight: float = 0.0,
        use_ema: bool = False,
        ema_decay: float = 0.995,
        use_ema_warmup: bool = False,
        ema_warmup_steps: int = 1000,
        async_sparse_reset: bool = False,
        async_sparse_reset_start_epoch: int = 2,
        use_swa_on_ema: bool = False,
        swa_top_k: int = 2,
        swa_weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.rank_moe_aux_weight: float = rank_moe_aux_weight
        self.use_ema: bool = use_ema
        self.ema_decay: float = float(ema_decay)
        self.use_ema_warmup: bool = bool(use_ema_warmup)
        self.ema_warmup_steps: int = max(1, int(ema_warmup_steps))
        self.async_sparse_reset: bool = bool(async_sparse_reset)
        self.async_sparse_reset_start_epoch: int = max(1, int(async_sparse_reset_start_epoch))
        self.initial_sparse_state: Dict[int, torch.Tensor] = (
            self._snapshot_sparse_params() if self.async_sparse_reset else {}
        )
        self.ema_update_count: int = 0
        self.ema_state: Dict[str, torch.Tensor] = (
            self._init_ema_state() if self.use_ema else {}
        )
        # SWA-on-EMA: weighted average of the top-K best EMA-validated epochs.
        # Sparse Embeddings are fused too because each .ema_model checkpoint
        # already stores them (dense=EMA shadow, sparse=live at epoch end), so
        # averaging full state_dicts naturally averages the embedding tables.
        self.use_swa_on_ema: bool = bool(use_swa_on_ema)
        self.swa_top_k: int = max(1, int(swa_top_k))
        self.swa_weights: List[float] = self._normalise_swa_weights(
            swa_weights, self.swa_top_k)
        self.swa_history: List[Tuple[float, str]] = []
        if self.use_swa_on_ema and not self.use_ema:
            logging.warning(
                "use_swa_on_ema=True requires use_ema=True; SWA-on-EMA will "
                "be skipped because no EMA checkpoints will be produced.")
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=(self.device != 'cpu'))
        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"async_sparse_reset={self.async_sparse_reset}, "
                     f"async_sparse_reset_start_epoch={self.async_sparse_reset_start_epoch}, "
                     f"use_ema={self.use_ema}, ema_decay={self.ema_decay}, "
                     f"use_ema_warmup={self.use_ema_warmup}, "
                     f"ema_warmup_steps={self.ema_warmup_steps}, "
                     f"use_swa_on_ema={self.use_swa_on_ema}, "
                     f"swa_top_k={self.swa_top_k}, "
                     f"swa_weights={self.swa_weights}")

        self.ranking_loss_fn = self.PairwiseRankingObjective(
            trainer=self,
            pairwise_loss_weight=0.05,
            max_pair_count=8192,
        ).to(self.device)
        self.current_epoch = 0

    @staticmethod
    def _normalise_swa_weights(
        weights: Optional[Sequence[float]], top_k: int,
    ) -> List[float]:
        """Validate and L1-normalise ``weights`` so that they sum to 1.

        - When ``weights`` is None, default to a linear ramp 0.6, 0.4, ... that
          favours the highest-AUC checkpoint. For top_k=2 this matches the
          user-requested (0.6, 0.4) split.
        - When ``weights`` is provided, its length must equal ``top_k``.
        - All weights must be >= 0 and have a positive sum; we renormalise so
          the SWA stays an affine (convex) combination regardless of input.
        """
        if weights is None:
            if top_k == 2:
                raw = [0.6, 0.4]
            else:
                raw = [1.0 / top_k] * top_k
        else:
            raw = [float(w) for w in weights]
        if len(raw) != top_k:
            raise ValueError(
                f"swa_weights has length {len(raw)} but swa_top_k={top_k}; "
                "they must match")
        if any(w < 0.0 for w in raw):
            raise ValueError(f"swa_weights must be non-negative, got {raw}")
        total = sum(raw)
        if total <= 0.0:
            raise ValueError(
                f"swa_weights must have a positive sum, got {raw}")
        return [w / total for w in raw]

    def _snapshot_sparse_params(self) -> Dict[int, torch.Tensor]:
        """Snapshot initial sparse parameters for async multi-epoch training.

        The snapshot is keyed by ``data_ptr`` and stored on CPU to avoid
        duplicating large embedding tables on GPU. At the beginning of every
        configured async epoch, sparse parameters are restored from this fixed
        initial state while dense parameters keep accumulating.
        """
        if not hasattr(self.model, 'get_sparse_params'):
            logging.warning(
                "async_sparse_reset=True but model has no get_sparse_params(); "
                "sparse reset will be disabled")
            self.async_sparse_reset = False
            return {}

        sparse_state: Dict[int, torch.Tensor] = {}
        sparse_params = self.model.get_sparse_params()
        for param in sparse_params:
            sparse_state[param.data_ptr()] = param.detach().cpu().clone()
        sparse_param_count = sum(t.numel() for t in sparse_state.values())
        logging.info(
            f"Snapshotted {len(sparse_state)} sparse tensors "
            f"({sparse_param_count:,} parameters) for async sparse reset")
        return sparse_state

    def _restore_initial_sparse_params(self, epoch: int) -> None:
        """Restore sparse parameters to the initial snapshot and reset Adagrad.

        This implements the EST-style asynchronous multi-epoch schedule:
        sparse embeddings restart from the same initial state each epoch while
        dense parameters and dense optimizer state keep training continuously.
        """
        if not self.async_sparse_reset or not self.initial_sparse_state:
            return
        if self.sparse_optimizer is None or not hasattr(self.model, 'get_sparse_params'):
            return

        restored = 0
        with torch.no_grad():
            for param in self.model.get_sparse_params():
                init_value = self.initial_sparse_state.get(param.data_ptr())
                if init_value is None:
                    continue
                param.copy_(init_value.to(device=param.device, dtype=param.dtype))
                restored += 1

        sparse_params = self.model.get_sparse_params()
        self.sparse_optimizer = torch.optim.Adagrad(
            sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
        )
        logging.info(
            f"Async sparse reset before epoch {epoch}: restored {restored} "
            "sparse tensors to the initial snapshot and reset Adagrad state")

    def _init_ema_state(self) -> Dict[str, torch.Tensor]:
        """Initialize EMA shadow weights for dense trainable parameters only."""
        sparse_ptrs = set()
        for module in self.model.modules():
            if isinstance(module, nn.Embedding):
                sparse_ptrs.add(module.weight.data_ptr())
        if hasattr(self.model, 'get_sparse_params'):
            sparse_ptrs.update({p.data_ptr() for p in self.model.get_sparse_params()})

        ema_state: Dict[str, torch.Tensor] = {}
        skipped_count = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad or not param.is_floating_point():
                continue
            if param.data_ptr() in sparse_ptrs or 'emb' in name.lower():
                skipped_count += 1
                continue
            ema_state[name] = param.detach().clone()

        logging.info(
            f"Initialized EMA for {len(ema_state)} dense parameter tensors; "
            f"skipped {skipped_count} Embedding/sparse tensors")
        return ema_state

    def _update_ema_state(self) -> None:
        """Update EMA after optimizer steps."""
        if not self.ema_state:
            return

        self.ema_update_count += 1
        ema_decay = self._get_current_ema_decay()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                ema_param = self.ema_state.get(name)
                if ema_param is None:
                    continue
                ema_param.mul_(ema_decay).add_(
                    param.detach(), alpha=1.0 - ema_decay)

    def _get_current_ema_decay(self) -> float:
        """Return the EMA decay to use for the current update.

        When ``use_ema_warmup`` is disabled, this is just the static
        ``ema_decay``. When enabled, the decay ramps linearly from 0 up to
        ``ema_decay`` over the first ``ema_warmup_steps`` EMA updates. The
        very low decay at the start makes the EMA shadow weights track the
        live weights closely, so they are not anchored at their (essentially
        random) initialization for the first several thousand steps.
        """
        if not self.use_ema_warmup:
            return self.ema_decay
        progress = min(1.0, self.ema_update_count / float(self.ema_warmup_steps))
        return self.ema_decay * progress

    def _swap_to_ema_weights(self) -> Dict[str, torch.Tensor]:
        """Copy EMA dense weights into the live model and return a backup."""
        backup: Dict[str, torch.Tensor] = {}
        if not self.ema_state:
            return backup

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                ema_param = self.ema_state.get(name)
                if ema_param is None:
                    continue
                backup[name] = param.detach().clone()
                param.copy_(ema_param.to(device=param.device, dtype=param.dtype))
        return backup

    def _restore_raw_weights(self, backup: Dict[str, torch.Tensor]) -> None:
        """Restore dense weights saved by ``_swap_to_ema_weights``."""
        if not backup:
            return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                raw_param = backup.get(name)
                if raw_param is None:
                    continue
                param.copy_(raw_param.to(device=param.device, dtype=param.dtype))

    def _build_ema_state_dict(self) -> Dict[str, torch.Tensor]:
        """Build full checkpoint state: EMA dense params + current sparse params."""
        state_dict = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }
        for name, ema_tensor in self.ema_state.items():
            target = state_dict.get(name)
            if target is None:
                continue
            state_dict[name] = ema_tensor.detach().to(
                device=target.device, dtype=target.dtype).clone()
        return state_dict

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _save_ema_step_checkpoint(self, global_step: int) -> str:
        """Save EMA dense weights with the current sparse Embedding weights."""
        dir_name = f"{self._build_step_dir_name(global_step)}.ema_model"
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self._build_ema_state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(
            f"Saved EMA checkpoint to {ckpt_dir}/model.pt "
            "(dense=EMA, sparse=current)")
        return ckpt_dir

    def _save_raw_step_checkpoint(self, global_step: int) -> str:
        """Save the current (raw) model state to a ``.raw_model`` sub-dir.

        Used at epoch end (before any EMA swap) so the live training weights
        are persisted alongside the EMA shadow weights, letting downstream
        evaluation compare raw vs EMA at the same training step.
        """
        dir_name = f"{self._build_step_dir_name(global_step)}.raw_model"
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved raw checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _record_swa_candidate(self, val_auc: float, ckpt_dir: str) -> None:
        """Track the EMA checkpoint at ``ckpt_dir`` as a SWA-on-EMA candidate.

        We keep the top-``swa_top_k`` checkpoints by validation AUC. Each entry
        stores only the directory path (not the loaded tensors) to keep CPU /
        GPU memory flat during training; the state_dicts are loaded once at
        the very end inside :meth:`_save_swa_checkpoint`.
        """
        if not self.use_swa_on_ema:
            return
        self.swa_history.append((float(val_auc), str(ckpt_dir)))
        # Sort descending by val_auc, then truncate.
        self.swa_history.sort(key=lambda item: item[0], reverse=True)
        self.swa_history = self.swa_history[: self.swa_top_k]
        snapshot = [
            (round(auc, 6), os.path.basename(d)) for auc, d in self.swa_history
        ]
        logging.info(f"SWA-on-EMA top-{self.swa_top_k} candidates: {snapshot}")

    def _save_swa_checkpoint(self) -> Optional[str]:
        """Weighted-average the top-K best EMA checkpoints into a SWA model.

        The averaging covers *every* floating tensor in the state_dict, which
        includes the sparse Embedding tables (each ``.ema_model`` checkpoint
        stores ``dense=EMA shadow`` together with the ``sparse=live`` weights
        for that epoch). Integer / boolean buffers are not averaged: they are
        taken from the highest-AUC checkpoint to preserve discrete semantics
        (e.g. step counters, bucket indices) that would be meaningless after
        a float average.

        Returns the SWA checkpoint directory on success, or ``None`` when the
        feature is disabled or no EMA candidates were ever recorded.
        """
        if not self.use_swa_on_ema:
            return None
        if not self.swa_history:
            logging.warning(
                "SWA-on-EMA requested but no EMA-validated epochs were "
                "recorded (e.g. EMA disabled or early-stopped before the "
                "first epoch end); skipping SWA checkpoint.")
            return None

        sources = list(self.swa_history)
        available = len(sources)
        if available < self.swa_top_k:
            used_weights = self.swa_weights[:available]
            total = sum(used_weights)
            if total <= 0.0:
                used_weights = [1.0 / available] * available
            else:
                used_weights = [w / total for w in used_weights]
            logging.warning(
                f"SWA-on-EMA requested top-{self.swa_top_k} but only "
                f"{available} EMA checkpoint(s) recorded; falling back to "
                f"top-{available} with renormalised weights {used_weights}")
        else:
            used_weights = list(self.swa_weights)

        loaded_state_dicts: List[Dict[str, torch.Tensor]] = []
        for _, ckpt_dir in sources:
            model_path = os.path.join(ckpt_dir, "model.pt")
            if not os.path.exists(model_path):
                logging.error(
                    f"SWA-on-EMA source missing: {model_path}; aborting "
                    "SWA checkpoint creation")
                return None
            loaded_state_dicts.append(
                torch.load(model_path, map_location="cpu")
            )

        reference = loaded_state_dicts[0]
        swa_state: Dict[str, torch.Tensor] = {}
        averaged_count = 0
        copied_count = 0
        skipped_count = 0
        for name, ref_tensor in reference.items():
            if not torch.is_tensor(ref_tensor):
                swa_state[name] = ref_tensor
                copied_count += 1
                continue
            if not ref_tensor.is_floating_point():
                swa_state[name] = ref_tensor.detach().clone()
                copied_count += 1
                continue
            # Float tensor: weighted average across loaded state_dicts. Any
            # state_dict missing the key, or whose tensor shape mismatches,
            # is silently skipped for that key (defensive: should not happen
            # since all .ema_model dirs come from the same model).
            accum = torch.zeros_like(ref_tensor, dtype=torch.float32)
            weight_used = 0.0
            for w, sd in zip(used_weights, loaded_state_dicts):
                candidate = sd.get(name)
                if candidate is None or candidate.shape != ref_tensor.shape:
                    skipped_count += 1
                    continue
                accum.add_(candidate.to(dtype=torch.float32), alpha=float(w))
                weight_used += float(w)
            if weight_used <= 0.0:
                swa_state[name] = ref_tensor.detach().clone()
                copied_count += 1
                continue
            if weight_used < 1.0:
                # Renormalise so a missing source does not silently dim the
                # tensor by the (now unused) weight share.
                accum.mul_(1.0 / weight_used)
            swa_state[name] = accum.to(dtype=ref_tensor.dtype)
            averaged_count += 1

        ckpt_dir = os.path.join(self.save_dir, "swa_on_ema_model")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(swa_state, os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        sources_repr = [
            (round(auc, 6), os.path.basename(d)) for auc, d in sources
        ]
        logging.info(
            f"Saved SWA-on-EMA checkpoint to {ckpt_dir}/model.pt "
            f"(averaged={averaged_count}, copied={copied_count}, "
            f"skipped_mismatch={skipped_count}) | "
            f"sources={sources_repr} | weights={used_weights}")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def _write_validation_summary(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
        validation_metrics: Dict[str, float],
    ) -> None:
        if not self.writer:
            return
        self.writer.add_scalar('AUC/valid', val_auc, total_step)
        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
        self._write_validation_metrics(total_step, validation_metrics)

    def _run_validation(
        self, epoch: int, total_step: int, label: str,
    ) -> Tuple[float, float]:
        """Evaluate the current live weights and update checkpoint tracking.

        Returns the (val_auc, val_logloss) so that callers (e.g. the EMA
        epoch-end branch in :meth:`train`) can pair the score with the
        checkpoint they are about to persist for SWA candidate tracking.
        """
        val_auc, val_logloss, validation_metrics = self.evaluate(epoch=epoch)
        logging.info(
            f"{label} Validation | AUC: {val_auc}, LogLoss: {val_logloss} | "
            f"{self._format_validation_metrics(validation_metrics)}")
        self._write_validation_summary(total_step, val_auc, val_logloss, validation_metrics)
        self._handle_validation_result(total_step, val_auc, val_logloss)
        return val_auc, val_logloss

    def _run_raw_validation(self, epoch: int, total_step: int, label: str) -> None:
        """Evaluate raw (non-EMA) weights for monitoring only.

        Does NOT update EarlyStopping and does NOT save a checkpoint. Useful
        when EMA is enabled and we want to observe the live model performance
        alongside the EMA-weight performance at every epoch end.
        """
        val_auc, val_logloss, validation_metrics = self.evaluate(epoch=epoch)
        logging.info(
            f"{label} Raw Validation | AUC: {val_auc}, LogLoss: {val_logloss} | "
            f"{self._format_validation_metrics(validation_metrics)}")
        if self.writer:
            self.writer.add_scalar('AUC/valid_raw', val_auc, total_step)
            self.writer.add_scalar('LogLoss/valid_raw', val_logloss, total_step)
            for metric_name, metric_value in validation_metrics.items():
                self.writer.add_scalar(
                    f'ValidationBreakdownRaw/{metric_name}', metric_value, total_step)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.

        The SWA-on-EMA checkpoint is materialised at every exit path (normal
        completion, step-level early stop, or epoch-level early stop) so that
        a partial run still yields a SWA model from whatever EMA epochs were
        validated before stopping.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0
        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0
            self.current_epoch = epoch
            if self.async_sparse_reset and epoch >= self.async_sparse_reset_start_epoch:
                self._restore_initial_sparse_params(epoch)
            for step, batch in train_pbar:
                loss, bce_loss, pairwise_loss, moe_aux_loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)
                    self.writer.add_scalar('BCE/train', bce_loss, total_step)
                    self.writer.add_scalar('PairwiseLoss/train', pairwise_loss, total_step)
                    self.writer.add_scalar('MoEAuxLoss/train', moe_aux_loss, total_step)


                train_pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "moe_aux": f"{moe_aux_loss:.4f}",
                })

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    self._run_validation(epoch, total_step, f"Step {total_step}")
                    self.model.train()
                    torch.cuda.empty_cache()

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        self._save_swa_checkpoint()
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            backup = {}
            if self.use_ema and self.ema_state:
                # 1) Raw validation (monitoring only).
                self._run_raw_validation(epoch, total_step, f"Epoch {epoch}")
                self.model.train()
                torch.cuda.empty_cache()
                # 2) Persist the raw training weights for this epoch.
                self._save_raw_step_checkpoint(total_step)
                # 3) Swap to EMA for the gating validation + EMA checkpoint.
                backup = self._swap_to_ema_weights()
            try:
                val_auc, _val_logloss = self._run_validation(
                    epoch, total_step, f"Epoch {epoch}")
                self.model.train()
                torch.cuda.empty_cache()

                # Always persist this epoch's checkpoint, even on early stop,
                # so the EMA / raw weights for the last completed epoch are
                # never lost.
                if backup:
                    # EMA active: save EMA-labeled epoch checkpoint and
                    # register it as a SWA-on-EMA candidate (sparse
                    # Embeddings travel inside the saved state_dict and will
                    # be fused alongside the dense EMA shadow at the end of
                    # training).
                    ema_ckpt_dir = self._save_ema_step_checkpoint(total_step)
                    self._record_swa_candidate(val_auc, ema_ckpt_dir)
                else:
                    # EMA disabled: save the live weights as the unlabeled
                    # epoch checkpoint (legacy layout).
                    self._save_step_checkpoint(
                        global_step=total_step,
                        is_best=False,
                        skip_model_file=False,
                    )

                if self.early_stopping.early_stop:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
            finally:
                if backup:
                    self._restore_raw_weights(backup)
            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if (
                not self.async_sparse_reset
                and epoch >= self.reinit_sparse_after_epoch
                and self.sparse_optimizer is not None
            ):
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

        # End-of-training SWA-on-EMA fuse. Safe to call even when disabled or
        # when no candidates were registered (returns None with a log line).
        self._save_swa_checkpoint()

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_gap_buckets: Dict[str, torch.Tensor] = {}
        # ``seq_history_time_ids`` stays ``None`` unless the dataset actually
        # emitted the temporal ids (i.e., ``enable_history_time_bias=True``),
        # so the model side can treat its presence as the feature flag.
        seq_history_time_ids: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_gap_buckets[domain] = device_batch.get(
                f'{domain}_gap_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            # Detect the temporal-bias branch via the presence of the first
            # canonical field. When enabled, the dataset always emits the full
            # 8-field set keyed by ``{domain}_hist_{field_name}``.
            probe_key = f'{domain}_hist_month_of_year'
            if probe_key in device_batch:
                if seq_history_time_ids is None:
                    seq_history_time_ids = {}
                seq_history_time_ids[domain] = {
                    field_name: device_batch[f'{domain}_hist_{field_name}']
                    for field_name in _HIST_TIME_FIELDS
                }
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            calendar_time_features=device_batch['calendar_time_features'],
            seq_gap_buckets=seq_gap_buckets,
            seq_history_time_ids=seq_history_time_ids,
        )

    def compute_bce_pairwise_ranking_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pairwise_loss_weight: float,
        max_pair_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = logits.view(-1)
        labels = labels.view(-1).float()

        binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(logits, labels)
        if pairwise_loss_weight <= 0:
            return self._loss_without_pairwise_term(binary_cross_entropy_loss, logits.device)

        positive_logits, negative_logits = self._split_logits_by_label(logits, labels)
        if self._has_no_valid_pairs(positive_logits, negative_logits):
            return self._loss_without_pairwise_term(binary_cross_entropy_loss, logits.device)

        score_margin = self._build_pairwise_score_margin(
            positive_logits, negative_logits, max_pair_count)
        pairwise_ranking_loss = F.softplus(-score_margin).mean()
        total_loss = binary_cross_entropy_loss + pairwise_loss_weight * pairwise_ranking_loss
        return total_loss, binary_cross_entropy_loss, pairwise_ranking_loss

    @staticmethod
    def _loss_without_pairwise_term(
        binary_cross_entropy_loss: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero_pairwise_loss = torch.tensor(0.0, device=device)
        return binary_cross_entropy_loss, binary_cross_entropy_loss, zero_pairwise_loss

    @staticmethod
    def _split_logits_by_label(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positive_logits = logits[labels > 0.5]
        negative_logits = logits[labels <= 0.5]
        return positive_logits, negative_logits

    @staticmethod
    def _has_no_valid_pairs(
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
    ) -> bool:
        return len(positive_logits) == 0 or len(negative_logits) == 0

    def _build_pairwise_score_margin(
        self,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        max_pair_count: int,
    ) -> torch.Tensor:
        possible_pair_count = len(positive_logits) * len(negative_logits)
        if possible_pair_count <= max_pair_count:
            return self._build_full_pairwise_margin(positive_logits, negative_logits)
        return self._sample_pairwise_margin(positive_logits, negative_logits, max_pair_count)

    @staticmethod
    def _build_full_pairwise_margin(
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
    ) -> torch.Tensor:
        score_margin = positive_logits.unsqueeze(1) - negative_logits.unsqueeze(0)
        return score_margin.view(-1)

    @staticmethod
    def _sample_pairwise_margin(
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        max_pair_count: int,
    ) -> torch.Tensor:
        positive_indices = torch.randint(
            0, len(positive_logits), (max_pair_count,), device=positive_logits.device)
        negative_indices = torch.randint(
            0, len(negative_logits), (max_pair_count,), device=negative_logits.device)
        return positive_logits[positive_indices] - negative_logits[negative_indices]

    def _train_step(self, batch: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        target_pairwise_loss_weight = 0.05
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            model_input = self._make_model_input(device_batch)
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.current_epoch < 0:
                current_pairwise_loss_weight = 0.0
            else:
                current_pairwise_loss_weight = (
                    target_pairwise_loss_weight * min(1.0, (self.current_epoch - 0) / 2.0)
                )
            self.ranking_loss_fn.pairwise_loss_weight = current_pairwise_loss_weight

            main_loss, bce_loss, pairwise_loss = self.ranking_loss_fn(logits, label)
            moe_aux_loss = self._get_rank_moe_aux_loss()
            loss = main_loss + self.rank_moe_aux_weight * moe_aux_loss

        loss.backward()

        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()
        self._update_ema_state()

        return loss.item(), bce_loss.item(), pairwise_loss.item(), moe_aux_loss.item()

    def _get_rank_moe_aux_loss(self) -> torch.Tensor:
        if self.rank_moe_aux_weight <= 0 or not hasattr(self.model, 'get_moe_aux_loss'):
            return next(self.model.parameters()).new_tensor(0.0)
        return self.model.get_moe_aux_loss()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float, Dict[str, float]]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss, metrics)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0).float()
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Filter NaN predictions (may appear if gradients explode).
        valid_prediction_mask = ~torch.isnan(all_logits)
        if not bool(valid_prediction_mask.all()):
            n_nan = int((~valid_prediction_mask).sum().item())
            logging.warning(
                f"[Evaluate] {n_nan}/{len(all_logits)} predictions are NaN, filtering them out")

        valid_logits = all_logits[valid_prediction_mask]
        valid_labels = all_labels[valid_prediction_mask]

        # Binary AUC via sklearn.
        probs = torch.sigmoid(valid_logits).numpy()
        labels_np = valid_labels.numpy()

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        validation_metrics = self._compute_validation_breakdown(valid_logits, valid_labels)
        return auc, logloss, validation_metrics

    def _compute_validation_breakdown(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        if len(logits) == 0:
            return self._empty_validation_breakdown()

        labels = labels.long()
        probabilities = torch.sigmoid(logits)
        positive_mask = labels == 1
        negative_mask = labels == 0
        predictions = probabilities >= 0.5

        positive_mean_probability = self._masked_mean(probabilities, positive_mask)
        negative_mean_probability = self._masked_mean(probabilities, negative_mask)
        positive_loss = self._masked_bce_loss(logits, labels, positive_mask)
        negative_loss = self._masked_bce_loss(logits, labels, negative_mask)

        true_positive = ((predictions == 1) & positive_mask).sum().item()
        false_positive = ((predictions == 1) & negative_mask).sum().item()
        true_negative = ((predictions == 0) & negative_mask).sum().item()
        false_negative = ((predictions == 0) & positive_mask).sum().item()

        return {
            'positive_loss': positive_loss,
            'negative_loss': negative_loss,
            'positive_mean_probability': positive_mean_probability,
            'negative_mean_probability': negative_mean_probability,
            'probability_gap': positive_mean_probability - negative_mean_probability,
            'positive_precision': self._safe_divide(true_positive, true_positive + false_positive),
            'negative_precision': self._safe_divide(true_negative, true_negative + false_negative),
            'positive_accuracy': self._safe_divide(true_positive, int(positive_mask.sum().item())),
            'negative_accuracy': self._safe_divide(true_negative, int(negative_mask.sum().item())),
        }

    @staticmethod
    def _empty_validation_breakdown() -> Dict[str, float]:
        return {
            'positive_loss': 0.0,
            'negative_loss': 0.0,
            'positive_mean_probability': 0.0,
            'negative_mean_probability': 0.0,
            'probability_gap': 0.0,
            'positive_precision': 0.0,
            'negative_precision': 0.0,
            'positive_accuracy': 0.0,
            'negative_accuracy': 0.0,
        }

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return values[mask].mean().item()

    @staticmethod
    def _masked_bce_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        if not bool(mask.any()):
            return 0.0
        return F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float()).item()

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    @staticmethod
    def _format_validation_metrics(metrics: Dict[str, float]) -> str:
        return (
            f"PosLoss: {metrics['positive_loss']:.6f}, "
            f"NegLoss: {metrics['negative_loss']:.6f}, "
            f"PosProb: {metrics['positive_mean_probability']:.6f}, "
            f"NegProb: {metrics['negative_mean_probability']:.6f}, "
            f"ProbGap: {metrics['probability_gap']:.6f}, "
            f"PosPrecision: {metrics['positive_precision']:.6f}, "
            f"NegPrecision: {metrics['negative_precision']:.6f}, "
            f"PosAccuracy: {metrics['positive_accuracy']:.6f}, "
            f"NegAccuracy: {metrics['negative_accuracy']:.6f}"
        )

    def _write_validation_metrics(self, total_step: int, metrics: Dict[str, float]) -> None:
        if not self.writer:
            return
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'ValidationBreakdown/{metric_name}', metric_value, total_step)

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            model_input = self._make_model_input(device_batch)
            logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
            logits = logits.squeeze(-1)  # (B,)

        return logits, label

    class PairwiseRankingObjective(nn.Module):
        def __init__(
            self,
            trainer: "PCVRHyFormerRankingTrainer",
            pairwise_loss_weight: float = 0.05,
            max_pair_count: int = 8192,
        ) -> None:
            super().__init__()
            self.trainer = trainer
            self.pairwise_loss_weight = pairwise_loss_weight
            self.max_pair_count = max_pair_count

        def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.trainer.compute_bce_pairwise_ranking_loss(
                logits=logits,
                labels=labels,
                pairwise_loss_weight=self.pairwise_loss_weight,
                max_pair_count=self.max_pair_count,
            )