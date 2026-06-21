"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""
import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            'entries': self.entries,
            'total_dim': self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1

# Adjacent-history gap bucket boundaries. The ids reserve:
# 0 = padding, 1 = latest behavior, 2+ = bucketized adjacent time gap.
GAP_BUCKET_BOUNDARIES = BUCKET_BOUNDARIES.copy()
NUM_GAP_BUCKETS = len(GAP_BUCKET_BOUNDARIES) + 3

LOCAL_TIME_OFFSET_SECONDS = 8 * 3600

# Per-position calendar fields emitted for the historical sequence
# time-perception branch. The order MUST stay aligned with
# ``HistoricalTemporalBiasInjector._TEMPORAL_FIELD_CONFIG`` on the model side
# so both halves agree on the field set; the model is robust to missing keys
# but matching the full set keeps behaviour reproducible.
_HIST_TIME_FIELDS: Tuple[str, ...] = (
    'month_of_year',
    'week_of_year',
    'day_of_year',
    'week_of_month',
    'day_of_week',
    'hour_of_day',
    'is_weekend',
    'time_period',
)


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        enable_intraday_calendar_features: bool = True,
        enable_weekly_calendar_features: bool = True,
        enable_annual_calendar_features: bool = True,
        use_gap_buckets: bool = False,
        enable_history_time_bias: bool = False,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.enable_intraday_calendar_features = enable_intraday_calendar_features
        self.enable_weekly_calendar_features = enable_weekly_calendar_features
        self.enable_annual_calendar_features = enable_annual_calendar_features
        self.use_gap_buckets = use_gap_buckets
        # When True, ``_convert_batch`` emits per-position hour / day-of-week /
        # week-of-year ids for every history item so the model can inject a
        # temporal-bias residual onto each sequence token.
        self.enable_history_time_bias = enable_history_time_bias
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._calendar_time_feature_buffer = np.zeros((B, 9), dtype=np.int64)

        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_gap = {}
        self._buf_seq_lens = {}
        # Pre-allocated buffers for the per-position temporal-bias ids.
        # Each entry is a dict {field_name: (B, max_len) int64 array}.
        self._buf_seq_hist_time: Dict[str, Dict[str, "npt.NDArray[np.int64]"]] = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_gap[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)
            if self.enable_history_time_bias:
                # Eight calendar fields matching the reference implementation
                # consumed by ``HistoricalTemporalBiasInjector``.
                self._buf_seq_hist_time[domain] = {
                    name: np.zeros((B, max_len), dtype=np.int64)
                    for name in _HIST_TIME_FIELDS
                }

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list

        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len

        padded[padded <= 0] = 0
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_dim), dtype=np.float32)

        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]

        return padded

    @staticmethod
    def _write_intraday_calendar_features(
        calendar_time_features: "npt.NDArray[np.int64]",
        hour_of_day: "npt.NDArray[np.int64]",
        minute_of_hour: "npt.NDArray[np.int64]",
    ) -> None:
        calendar_time_features[:, 0] = hour_of_day * 60 + minute_of_hour
        calendar_time_features[:, 1] = hour_of_day
        calendar_time_features[:, 8] = hour_of_day // 6

    @staticmethod
    def _write_weekly_calendar_features(
        calendar_time_features: "npt.NDArray[np.int64]",
        hour_of_day: "npt.NDArray[np.int64]",
        day_of_week: "npt.NDArray[np.int64]",
    ) -> None:
        calendar_time_features[:, 2] = day_of_week
        calendar_time_features[:, 3] = day_of_week * 24 + hour_of_day
        calendar_time_features[:, 7] = (day_of_week >= 5).astype(np.int64)

    @staticmethod
    def _write_annual_calendar_features(
        calendar_time_features: "npt.NDArray[np.int64]",
        day_of_month: "npt.NDArray[np.int64]",
        month_of_year: "npt.NDArray[np.int64]",
        day_of_year: "npt.NDArray[np.int64]",
    ) -> None:
        calendar_time_features[:, 4] = day_of_month
        calendar_time_features[:, 5] = month_of_year
        calendar_time_features[:, 6] = day_of_year

    @staticmethod
    def _compute_calendar_fields(
        timestamps: "npt.NDArray[np.int64]",
    ) -> Tuple[
        "npt.NDArray[np.int64]",
        "npt.NDArray[np.int64]",
        "npt.NDArray[np.int64]",
        "npt.NDArray[np.int64]",
        "npt.NDArray[np.int64]",
        "npt.NDArray[np.int64]",
    ]:
        """Decompose Unix timestamps into calendar fields using only NumPy.

        Returns ``(hour_of_day, minute_of_hour, day_of_week, day_of_month,
        month_of_year, day_of_year)``. ``day_of_week`` follows the
        Monday=0..Sunday=6 convention; ``day_of_month``/``month_of_year``/
        ``day_of_year`` are all 1-based, matching pandas' semantics.
        """
        ts = timestamps.astype(np.int64, copy=False)

        # Intraday / weekday fields via integer arithmetic.
        seconds_in_day = ts % 86400
        hour_of_day = (seconds_in_day // 3600).astype(np.int64)
        minute_of_hour = ((seconds_in_day // 60) % 60).astype(np.int64)

        # 1970-01-01 was a Thursday -> weekday index 3 (Mon=0..Sun=6).
        days_since_epoch = ts // 86400
        day_of_week = ((days_since_epoch + 3) % 7).astype(np.int64)

        # Year / month / day-of-year via datetime64 truncation arithmetic.
        dt64_s = ts.astype('datetime64[s]')
        dt64_d = dt64_s.astype('datetime64[D]')
        dt64_m = dt64_s.astype('datetime64[M]')
        dt64_y = dt64_s.astype('datetime64[Y]')

        day_of_month = ((dt64_d - dt64_m).astype(np.int64) + 1).astype(np.int64)
        month_of_year = ((dt64_m - dt64_y).astype(np.int64) + 1).astype(np.int64)
        day_of_year = ((dt64_d - dt64_y).astype(np.int64) + 1).astype(np.int64)

        return (
            hour_of_day,
            minute_of_hour,
            day_of_week,
            day_of_month,
            month_of_year,
            day_of_year,
        )

    @staticmethod
    def _fill_history_time_buffers(
        ts_padded: "npt.NDArray[np.int64]",
        valid_pos_mask: "npt.NDArray[np.bool_]",
        buffers: Dict[str, "npt.NDArray[np.int64]"],
        batch_rows: int,
    ) -> None:
        """Populate the 8-field per-position temporal id buffers in place.

        Mirrors the reference implementation: all fields use 1-based ids with
        index 0 reserved for padding, and padding positions are zeroed out at
        the end so the model's ``padding_idx=0`` embedding rows stay neutral.

        Args:
            ts_padded: ``(B, max_len)`` raw (UTC) unix-second timestamps; ``0``
                marks padding slots.
            valid_pos_mask: ``(B, max_len)`` boolean mask, ``True`` for real
                history positions.
            buffers: pre-allocated id buffers keyed by field name; each entry
                is sliced ``[:batch_rows]`` for write-back.
            batch_rows: ``B`` (used to slice the pre-allocated buffers).
        """
        # Shift to local time once; padding still has ts=0 but is masked at the end.
        local_seq_ts = (ts_padded + LOCAL_TIME_OFFSET_SECONDS).astype(np.int64)

        # Intraday / weekday via integer arithmetic.
        raw_hour = ((local_seq_ts % 86400) // 3600).astype(np.int64)  # 0~23
        days_since_epoch_seq = (local_seq_ts // 86400).astype(np.int64)
        # 1970-01-01 was Thursday (idx 3 with Mon=0..Sun=6); shift to 1..7.
        day_of_week_id = ((days_since_epoch_seq + 3) % 7 + 1).astype(np.int64)

        # Year / month / day-of-year via datetime64 truncation.
        local_dt_s = local_seq_ts.astype('datetime64[s]')
        local_dt_d = local_dt_s.astype('datetime64[D]')
        local_dt_m = local_dt_s.astype('datetime64[M]')
        local_dt_y = local_dt_s.astype('datetime64[Y]')
        day_of_month_raw = ((local_dt_d - local_dt_m).astype(np.int64) + 1).astype(np.int64)
        month_of_year_id = ((local_dt_m - local_dt_y).astype(np.int64) + 1).astype(np.int64)
        day_of_year_id = ((local_dt_d - local_dt_y).astype(np.int64) + 1).astype(np.int64)

        # Derived fields.
        week_of_year_id = ((day_of_year_id - 1) // 7 + 1).astype(np.int64)
        week_of_month_id = ((day_of_month_raw - 1) // 7 + 1).astype(np.int64)
        # is_weekend: 1=weekday, 2=Sat/Sun (day_of_week_id >= 6).
        is_weekend_id = np.where(day_of_week_id >= 6, 2, 1).astype(np.int64)
        # Day-part time slot (7 bins of varying width, hour-of-day driven).
        time_period_id = np.ones_like(raw_hour)               # 凌晨
        time_period_id = np.where(raw_hour >= 6,  2, time_period_id)  # 早上
        time_period_id = np.where(raw_hour >= 9,  3, time_period_id)  # 上午
        time_period_id = np.where(raw_hour >= 12, 4, time_period_id)  # 中午
        time_period_id = np.where(raw_hour >= 14, 5, time_period_id)  # 下午
        time_period_id = np.where(raw_hour >= 18, 6, time_period_id)  # 晚上
        time_period_id = np.where(raw_hour >= 22, 7, time_period_id)  # 深夜
        # Hour-of-day stored as 1..24 so padding=0 stays reserved.
        hour_of_day_id = (raw_hour + 1).astype(np.int64)

        # Mask out padding positions in a single vectorised pass and write back
        # into the caller's pre-allocated buffers.
        field_to_array = {
            'month_of_year': month_of_year_id,
            'week_of_year':  week_of_year_id,
            'day_of_year':   day_of_year_id,
            'week_of_month': week_of_month_id,
            'day_of_week':   day_of_week_id,
            'hour_of_day':   hour_of_day_id,
            'is_weekend':    is_weekend_id,
            'time_period':   time_period_id.astype(np.int64),
        }
        for field_name, raw_ids in field_to_array.items():
            buffers[field_name][:batch_rows] = np.where(valid_pos_mask, raw_ids, 0)

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = (
            batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
            + LOCAL_TIME_OFFSET_SECONDS
        )
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()
        calendar_time_features_np = self._calendar_time_feature_buffer[:B]
        calendar_time_features_np[:] = 0

        (
            hour_of_day,
            minute_of_hour,
            day_of_week,
            day_of_month,
            month_of_year,
            day_of_year,
        ) = self._compute_calendar_fields(timestamps)

        if self.enable_intraday_calendar_features:
            self._write_intraday_calendar_features(
                calendar_time_features_np, hour_of_day, minute_of_hour)
        if self.enable_weekly_calendar_features:
            self._write_weekly_calendar_features(
                calendar_time_features_np, hour_of_day, day_of_week)
        if self.enable_annual_calendar_features:
            self._write_annual_calendar_features(
                calendar_time_features_np,
                day_of_month,
                month_of_year,
                day_of_year,
            )

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
            'calendar_time_features': torch.from_numpy(calendar_time_features_np.copy()),
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))
            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            # Values <= 0 -> 0.
            out[out <= 0] = 0

            # Check out-of-bound values per feature's vocab_size.
            # vs==0 means no vocab info; force the whole slice to 0 so that
            # the model's 1-slot Embedding is never indexed out of range.
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing.
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                # Pad timestamps into shape (B, max_len).
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                # np.searchsorted returns values in [0, len(BUCKET_BOUNDARIES)].
                # After +1 the nominal range is [1, len(BUCKET_BOUNDARIES)+1];
                # the upper bound only appears when time_diff exceeds the
                # largest boundary (~1 year) and would index past
                # nn.Embedding(NUM_TIME_BUCKETS=len(BUCKET_BOUNDARIES)+1).
                # Clip raw result to [0, len(BUCKET_BOUNDARIES)-1] so the final
                # bucket id (after +1) stays within [1, len(BUCKET_BOUNDARIES)]
                # and is always a valid Embedding index. Time-diffs beyond the
                # largest boundary collapse into the last bucket.
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

            # ---- Historical temporal-bias ids (8 calendar fields) ----
            # Only emit when the model is configured to inject temporal biases;
            # otherwise we'd pay the datetime64 cost for no benefit. The field
            # set mirrors ``HistoricalTemporalBiasInjector._TEMPORAL_FIELD_CONFIG``.
            if self.enable_history_time_bias:
                hist_time_bufs = self._buf_seq_hist_time[domain]
                for _field in _HIST_TIME_FIELDS:
                    hist_time_bufs[_field][:B] = 0
                if ts_ci is not None:
                    valid_pos_mask = ts_padded > 0  # (B, max_len)
                    if valid_pos_mask.any():
                        self._fill_history_time_buffers(
                            ts_padded=ts_padded,
                            valid_pos_mask=valid_pos_mask,
                            buffers=hist_time_bufs,
                            batch_rows=B,
                        )
                for _field in _HIST_TIME_FIELDS:
                    result[f'{domain}_hist_{_field}'] = torch.from_numpy(
                        hist_time_bufs[_field][:B].copy()
                    )

            if self.use_gap_buckets:
                gap_bucket = self._buf_seq_gap[domain][:B]
                gap_bucket[:] = 0
                if ts_ci is not None:
                    for i in range(B):
                        valid_len = int(lengths[i])
                        if valid_len <= 0:
                            continue
                        gap_bucket[i, 0] = 1
                        if valid_len == 1:
                            continue
                        adjacent_time_gaps = np.maximum(
                            ts_padded[i, :valid_len - 1] - ts_padded[i, 1:valid_len],
                            0,
                        )
                        gap_bucket[i, 1:valid_len] = (
                            np.searchsorted(GAP_BUCKET_BOUNDARIES, adjacent_time_gaps) + 2
                        )
                result[f'{domain}_gap_bucket'] = torch.from_numpy(gap_bucket.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    valid_num_workers: Optional[int] = None,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    The validation split is taken as the last ``valid_ratio`` fraction of Row
    Groups (in the file order returned by ``glob``).

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    # train_ratio: use only the first N% of the training Row Groups.
    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        **kwargs,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
        **kwargs,
    )
    if valid_num_workers is None:
        valid_num_workers = num_workers
    _valid_kw = {}
    if valid_num_workers > 0:
        _valid_kw['persistent_workers'] = True
        _valid_kw['prefetch_factor'] = 2

    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=valid_num_workers, pin_memory=use_cuda, **_valid_kw,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}, "
                 f"valid_num_workers={valid_num_workers}")

    return train_loader, valid_loader, train_dataset
