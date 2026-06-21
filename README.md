# TAAC2026 - PCVRHyFormer

基于 HyFormer 混合 Transformer 的转化率预估模型。

## 项目结构

```
├── train.py          # 训练入口
├── infer.py          # 推理入口
├── model.py          # PCVRHyFormer 模型定义
├── trainer.py        # 训练器（含 EMA / 混合精度 / Pairwise Ranking Loss）
├── dataset.py        # Parquet 数据集加载
├── utils.py          # 工具函数
├── ns_groups.json    # NS Token 分组配置
├── run.sh            # 启动脚本
└── baseline/         # 基线版本（未跟踪）
```

## 环境依赖

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy
- PyArrow

```bash
pip install torch numpy pyarrow
```

## 训练

编辑 `run.sh` 中的路径配置，然后：

```bash
bash run.sh
```

或直接调用：

```bash
python train.py \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --enable_intraday_calendar_features \
    --enable_weekly_calendar_features \
    --enable_annual_calendar_features \
    --enable_history_time_bias \
    --seqContextwakeup \
    --use_ema --ema_decay 0.999 \
    --async_sparse_reset --async_sparse_reset_start_epoch 2 \
    --num_workers 8
```

## 推理

```bash
python infer.py --ckpt_dir <checkpoint_dir> --data_path <parquet_path>
```
