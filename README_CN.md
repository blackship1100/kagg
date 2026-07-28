# MSCapital - Real Financial Market Forecasting

比赛链接：https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting

## 比赛状态

- 类别：Kaggle Community
- Kaggle API 奖励字段：`Kudos`
- 主办方页面补充奖励：前十名可获上海线下活动邀请，并可报销最高 5000 元人民币的交通、餐饮和住宿费用
- 截止时间：2026-10-09 16:00 UTC
- 任务：根据真实市场微观结构数据预测未来收益
- 指标：`cos(prediction, target)`，即预测向量与目标向量的余弦相似度

## 数据结构

总下载体积约 9.95 GB。

| 文件 | 实测行数 | 字节数 | 内容 |
|---|---:|---:|---|
| `train/label.feather` | 1,257,637 | 10,012,456 | `month`、`sample_id`、`target` |
| `train/market.feather` | 221,756,611 | 4,396,491,768 | 约 10 分钟的等间隔盘口/聚合成交序列 |
| `train/order.feather` | 170,056,583 | 1,296,050,852 | 预测前约 60 秒的原始委托与撤单流 |
| `train/transaction.feather` | 103,970,264 | 499,265,544 | 预测前约 60 秒的逐笔成交流 |
| `test/market.feather` | 118,359,166 | 2,497,837,624 | 测试市场序列 |
| `test/order.feather` | 119,465,287 | 906,047,012 | 测试委托流 |
| `test/transaction.feather` | 70,909,408 | 337,477,704 | 测试成交流 |
| `submission.csv` | 647,896 | 7,015,767 | `sample_id`、`prediction` |

训练月份为 0-70；测试集合并了月份 71-108，实际自然月没有提供。

训练集的三个序列表都覆盖 `sample_id=0..1,257,636`，测试集都覆盖
`sample_id=0..647,895`。序列表已按 `sample_id` 连续存放，可以流式读取；仍建议在
数据加载器中按 `sample_id, seconds_before_predict` 显式排序。`month` 只存在于训练标签表，
需要通过 `sample_id` 连接到三类序列。

### 文件格式与字段类型

除提交模板为 CSV 外，其余文件都是 Apache Feather V2/Arrow IPC，一列一个强类型数组。
价格和时间均为 `float32`，体量和计数为整数，适合用 PyArrow、Polars 或 pandas 读取。

| 数据表 | 字段与类型 |
|---|---|
| `label` | `month:int16`, `sample_id:int32`, `target:float32` |
| `market` | `sample_id:int32`, `seconds_before_predict:float32`, `transaction_avgprice:float32`, `transaction_volume:int32`, `transaction_count:int32`, 两档买卖价 `float32`，两档买卖量 `int32` |
| `order` | `sample_id:int32`, `seconds_before_predict:float32`, `price:float32`, `volume:int32`, `side:int8`, `order_action:int8` |
| `transaction` | `sample_id:int32`, `seconds_before_predict:float32`, `price:float32`, `volume:int32`, `side:int8` |
| `submission` | `sample_id`, `prediction`；模板预测值全部为 0 |

### 每个样本的序列长度

| 数据 | 时间范围 | 平均长度 | 中位数 | P95 | 最小-最大 |
|---|---:|---:|---:|---:|---:|
| 训练 `market` | 约 0-599 秒 | 176.3 | 189 | 199 | 1-212 |
| 测试 `market` | 约 0-597 秒 | 182.7 | 194 | 199 | 18-200 |
| 训练 `order` | 0-60 秒 | 135.2 | 88 | 426 | 1-999 |
| 测试 `order` | 0-60 秒 | 184.4 | 121 | 582 | 1-999 |
| 训练 `transaction` | 0-60 秒 | 82.7 | 48 | 279 | 1-999 |
| 测试 `transaction` | 0-60 秒 | 109.4 | 65 | 366 | 1-999 |

`order` 和 `transaction` 的长度上限恰好为 999，应按可能存在截断上限来设计
padding、mask 和采样逻辑。训练与测试的事件密度也有明显差异，标准化统计不能直接混用。

### market 字段

- `sample_id`：样本主键。
- `seconds_before_predict`：距预测时点的秒数，约覆盖 10 分钟。
- `transaction_avgprice`、`transaction_volume`、`transaction_count`：该时间片的聚合成交价、成交量和成交笔数。
- `ask_price_1`, `ask_price_2`：卖一、卖二价。
- `bid_price_1`, `bid_price_2`：买一、买二价。
- `ask_volume_1`, `ask_volume_2`：卖一、卖二挂单量。
- `bid_volume_1`, `bid_volume_2`：买一、买二挂单量。

`transaction_avgprice` 是唯一有缺失值的字段：训练集缺失 68,629,744 行（30.948%），
测试集缺失 31,260,702 行（26.412%）。这些行的 `transaction_volume` 和
`transaction_count` 通常为 0，表示该时间片没有成交；不能简单把平均成交价当作 0，
应使用前值/中间价填充并额外保留 `has_transaction` 掩码。

### order 字段

- `sample_id`, `seconds_before_predict`, `price`, `volume`
- `side`: 0=Buy，1=Sell
- `order_action`: 0=New order，1=Cancel order

训练集中新委托 128,101,670 条、撤单 41,954,913 条；所有字段均无缺失值。

### transaction 字段

- `sample_id`, `seconds_before_predict`, `price`, `volume`
- `side`: 0=Aggressive buy，1=Aggressive sell

训练集中主动买 50,261,948 条、主动卖 53,708,316 条；所有字段均无缺失值。

### 标签分布

- 71 个训练月份，编号 0-70，每月 17,187-17,852 个样本。
- `target` 均值约 -0.0000123，标准差约 0.002618，中位数为 0。
- 1%/99% 分位数约为 -0.006722/0.007831，极值约为 -0.06377/0.08360。
- 标签、三张训练序列表都可通过连续的 `sample_id` 一一对应，且主键与标签均无缺失。

## 当前下载方式

`download_parallel.py` 使用 Kaggle 官方逐文件接口和 Google Storage Range 请求：

- 1 MiB 分块
- 32 个持久连接并发
- 失败自动重试和签名 URL 自动刷新
- `.parts/` 保存断点，任务中断后可继续
- 下载结束后逐文件组装并按官方字节数验证
- Token 只从 `KAGGLE_API_TOKEN` 环境变量读取，不写入代码、日志或状态文件

下载已完成并按官方文件大小复核：8 个正式文件共 9,950,198,727 字节。重复的
`.parts` 分片和 7 MB 中断归档已在 2026-07-26 清理，共释放 9,940,510,536 字节
（约 9.258 GiB）；正式数据未删除。

下载状态：`download_status.json`

标准输出：`download_stdout.log`

错误日志：`download_stderr.log`

恢复命令：

```powershell
$env:KAGGLE_API_TOKEN='<your-token>'
python .\download_parallel.py --output-dir .\data --workers 32 --chunk-mib 1
```

## 建模路线

P1 全量流水线已经完成，P1.1 已增加订单类别事件/体量占比特征。初始结果见
[`docs/P1_RESULT_CN.md`](docs/P1_RESULT_CN.md)，当前最佳结果见
[`docs/P1_1_RESULT_CN.md`](docs/P1_1_RESULT_CN.md)。当前最佳三种子 OOF 整体 cosine 为
`0.153723`，提交文件为
`artifacts/submissions/ensemble-5da978eaa0f118c1.csv`。该版本已于 2026-07-29 提交，
Public Score 为 `0.132`，查询时约为第 36 名；原基线为 `0.129`。

### 第一阶段：聚合特征基线

先按 `sample_id` 聚合每个数据源，建立三种单信号基线和 LightGBM 强基线：

- mid price、spread、microprice、盘口 imbalance
- 价格收益、波动率、最高/最低/最后值
- 成交量、成交笔数、主动买卖 imbalance
- 新委托/撤单数量与体量、买卖方向 imbalance
- 多个时间窗口：最后 5/10/30/60 秒和完整窗口

验证必须按 `month` 向前切分，例如训练 0-54、验证 55-62、最终验证 63-70；不能随机拆分行。

### 第二阶段：序列深度学习

每个 `sample_id` 有三种不同时间尺度的序列：

1. `market`：10 分钟低频序列，可使用 TCN、GRU 或 Transformer encoder。
2. `order`：60 秒事件流，可使用事件 embedding、TCN 或轻量 Transformer。
3. `transaction`：60 秒成交流，可使用方向/价格/体量 embedding 与 attention pooling。

三路编码器输出拼接后接 Residual MLP 回归头。价格应转换为相对最后 mid price 的收益或 basis points，体量使用 `log1p`，避免不同股票价格尺度直接进入网络。

深度学习只按单卡设计：RTX 5070 Ti 16 GB 是最低兼容目标，启用 AMP、梯度累积、
padding mask、梯度裁剪和必要时的 gradient checkpointing；RTX 5090 复用同一架构，
只提高物理 batch 或在消融有效后把事件长度从 256 提高到 512。项目不规划多机、DDP、
NCCL、Slurm 或 PRO 6000 专用优化。

### 第三阶段：指标与集成

- 损失可使用 `MSE + lambda * (1 - cosine_similarity)`。
- OOF 预测按月保存，检查每月 cosine 和整体 cosine。
- 对不同 seed、不同窗口和 GBDT/NN 做线性加权。
- 余弦指标对整体缩放不敏感，但对方向、异常值和样本间相对幅度敏感；提交前应处理极端预测。

## 已确认的小文件

- `train/label.feather`：1,257,637 行、3 列，无缺失值。
- `submission.csv`：647,896 行、2 列。
