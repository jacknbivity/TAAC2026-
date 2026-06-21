# TAAC2026 - PCVRHyFormer

基于 HyFormer 混合 Transformer 的转化率预估模型，在 baseline 基础上新增以下优化。


## 项目结构

```
├── train.py          # 训练入口
├── infer.py          # 推理入口
├── model.py          # 模型定义
├── trainer.py        # 训练器
├── dataset.py        # 数据集
├── utils.py          # 工具函数
├── ns_groups.json    # NS Token 分组
└── run.sh            # 启动脚本
```

## 相对 baseline 的优化

### 1. 序列上下文唤醒 (`--seqContextwakeup`)

从用户/物品 NS Token 构建候选目标锚点，通过 Top-K 目标感知注意力 + 域门控聚合 + 残差投影，让序列编码感知当前候选 item。

### 2. 日历时间特征

- `--enable_intraday_calendar_features`：日内时间 Embedding（分钟、小时、时段）
- `--enable_weekly_calendar_features`：周内时间 Embedding（星期几、周时、周末）
- `--enable_annual_calendar_features`：年度时间 Embedding（日、月、年日）

将时间戳分解为 9 个日历字段，通过 `CalendarTimeFeatureEncoder` 注入 NS Token。

### 3. 历史序列时间偏置 (`--enable_history_time_bias`)

为每条历史行为序列的每个位置注入 8 字段时间感知残差（月/周/年日/周月/星期/时/周末/时段），使用离散 Embedding + 周期编码双表示，零初始化保证初始等价于恒等映射。

### 4. EMA 指数移动平均 (`--use_ema --ema_decay 0.999`)

对 Dense 参数维护 EMA 影子权重，验证时使用 EMA 参数提升泛化能力。

### 5. 异步稀疏重置 (`--async_sparse_reset --async_sparse_reset_start_epoch 2`)

EST 风格多 epoch 训练：从第 2 个 epoch 起，每个 epoch 开始时将稀疏 Embedding 重置回初始快照，Dense 参数持续训练。

### 6. Dense 特征编码

替代 baseline 的简单 `nn.Linear + nn.LayerNorm` 投影，将用户 Dense 特征分解为**统计段 + Sum Embedding 段 + LMF Embedding 段**三部分，分别投影后用 SiLU 融合，得到更丰富的用户表示。

### 7. 联合损失函数

在 BCE Loss 基础上叠加 Pairwise Ranking Loss：

$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda \cdot \text{softplus}(-(\text{pos} - \text{neg}))$$

Pairwise 权重在前 2 个 epoch 从 0 线性 ramp 至 0.05。同时使用 bf16 混合精度加速训练。

## 训练

```bash
bash run.sh
```
