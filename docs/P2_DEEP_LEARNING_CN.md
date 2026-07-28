# MSCapital P2 单卡深度学习第一版

生成日期：2026-07-29

## 当前结论

第一版深度学习工程闭环已经实现，并在真实的前 1,000 个训练样本上完成了：

1. 从 CanonicalStore 构建定长 float16 序列缓存。
2. Market TCN、Transaction Event Transformer 和 Transaction Grid TCN 前向与反向。
3. 门控 Late Fusion、MSE/可选 cosine loss、梯度累积与梯度裁剪。
4. checkpoint 原子保存、重新加载和一致性推理。
5. 按月份滚动 OOF、测试集推理、与 LightGBM 的 OOF 融合。

当前开发机没有 NVIDIA GPU 或驱动，因此只能完成 CPU 功能验收，不能声称 RTX 5070 Ti
的 15 GB 显存验收已经通过，也没有在当前机器上启动全量四折训练。

## 模型使用哪些数据

第一版深度模型只使用 `market` 和 `transaction`，不把 `sample_id`、`month` 或原始绝对价格
输入网络。`order` 已完整用于当前 783 列 LightGBM，后续仅在消融证明有增益时再增加独立的
Order Event Encoder，避免第一版把显存和训练时间同时扩大。

### Market 分支

- 最多 212 步，覆盖预测前约 10 分钟。
- 每步 15 列：归一化时间、mid 相对最终参考 mid、mid return、spread、microprice、
  L1/L2 imbalance、L1/L2 depth、聚合成交价格/量/笔数、成交 mask、盘口 mask 和时间间隔。
- 编码器为 5 个 Residual TCN block，dilation 为 `1/2/4/8/16`。
- 使用 masked attention pooling 和最后有效时刻 pooling。

### Transaction Event 分支

- 保留最近 256 条逐笔成交，较早事件从左侧截断，短序列从左侧 padding。
- 每条事件 6 个连续特征：时间、事件间隔、相对 mid 成交价、价格变化、`log1p(volume)`、
  是否延续上一笔方向；买卖方向单独使用 embedding。
- 编码器为 4 层 Pre-LN Transformer，hidden size 128，4 heads，padding mask 始终生效。

### Transaction Grid 分支

- 所有逐笔成交映射到完整 60 个一秒格，弥补“最近 256 条”对高密度样本早期事件的丢失。
- 每格包含主动买卖量/笔数、量和笔数 imbalance、VWAP、最大成交量、最后成交价和成交 mask。
- 使用 4 个轻量 Residual TCN block 编码。

### 门控融合

Event Transformer 与 Grid TCN 先合成 Transaction embedding，再计算：

```text
gate = sigmoid(W([market, transaction]))
fused = market + gate * transaction
prediction = residual_mlp(fused)
```

生产宽度模型共有 `1,774,465` 个参数。模型规模低于最初上限，目的是先验证有效性；若完整
OOF 明确超过单信号基线，再优先扩大 TCN channel 或增加 Order Encoder，而不是无证据堆参数。

## 序列缓存

缓存位于 `artifacts/cache/sequences/v1/`，按连续 25,000 个 `sample_id` 分片。每个分片包含：

```text
sample_id.npy
market_values.npy
market_mask.npy
transaction_values.npy
transaction_side.npy
transaction_mask.npy
transaction_grid.npy
```

连续值使用 float16，side 使用 int8，mask 使用 bool。每个文件原子写入并附带 SHA-256；
manifest 记录源 CanonicalStore digest、shape、dtype、输入列语义和整体内容摘要。当前完整训练加测试
缓存共 1,095 个文件、21,658,388,855 字节（20.171 GiB）。DataLoader 使用分片 mmap、pinned
memory 和分片内 batch，避免随机跨文件读取。

## 训练与验证

- 原始 `target`，默认纯 MSE；`cosine_loss_weight > 0` 时按 batch target variance 缩放 cosine 项。
- AdamW，学习率 `2e-4`，weight decay `1e-3`，warmup + cosine decay。
- 5070 Ti 默认 physical batch 32、梯度累积 8、effective batch 256。
- CUDA 自动启用 AMP；梯度范数裁剪为 1.0。
- CUDA OOM 时丢弃本次尝试，从相同 seed 重新开始并把 physical batch 减半。
- 验证按 `0-46 -> 47-54`、`0-54 -> 55-62`、`0-62 -> 63-66`、
  `0-66 -> 67-70`，禁止随机拆分正式 OOF。
- 每折根据 validation cosine 早停，保存最佳 checkpoint、预测、history 和峰值显存。

## 已完成的真实数据冒烟

真实前 1,000 个样本都来自 month 0，因此只用于工程冒烟，并采用固定 seed 的 800/200 随机拆分。

| 配置 | 训练量 | 参数量 | Valid cosine | 用途 |
|---|---:|---:|---:|---|
| 缩小宽度、完整 1 epoch | 800 | 89,185 | 0.092544 | 验证完整训练循环 |
| 生产宽度、1 个训练 batch | 16 | 1,774,465 | 0.021329 | 验证正式架构反向/保存/重载 |

这些分数没有比赛统计意义：样本极少、没有跨月份验证，生产模型也只更新了一次。正式方案是否有效，
必须看四折 OOF 以及与 `0.153723` LightGBM OOF 的融合结果。

## 复现命令

先安装适配显卡驱动的官方 CUDA PyTorch，再执行：

```powershell
$env:PYTHONPATH = "$PWD\src"

python -m mscapital.cli --config configs/base.toml build-sequences `
  --split both --resume

python -m mscapital.cli --config configs/base.toml train-deep `
  --seed 17 --resume --device cuda

python -m mscapital.cli --config configs/base.toml predict-deep-test `
  --run-id <deep-run-id> --resume --device cuda

python -m mscapital.cli --config configs/base.toml blend-tabular-deep `
  --tabular-run-id ensemble-5da978eaa0f118c1 `
  --deep-run-id <deep-run-id>

python -m mscapital.cli --config configs/base.toml make-submission `
  --run-id <blend-run-id>
```

开发机的快速真实数据检查：

```powershell
python -m mscapital.cli --config configs/base.toml smoke-deep `
  --max-samples 1000 --resume --device cpu

python -m mscapital.cli --config configs/base.toml smoke-deep `
  --max-samples 1000 --resume --device cpu --full-model --max-train-batches 1
```

## 进入正式训练前的验收线

1. 5070 Ti 上完整 forward/backward/validation 峰值显存不超过 15 GB。
2. 四折 OOF 每折都有有限预测，样本与月份严格对齐。
3. Deep OOF 至少显著超过对应单信号基线；否则先做输入与模型消融。
4. Deep + 783 列 LightGBM 的统一 OOF 权重优于纯 LightGBM，才生成融合提交。
5. 不根据 Public Leaderboard 反复调融合权重。
