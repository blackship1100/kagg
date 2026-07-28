# MSCapital P1 全量结果

生成日期：2026-07-28

## 结论

P1 强表格基线已经完成从只读 Feather、CanonicalStore、四类聚合特征、月份滚动
OOF、全量训练、测试预测到提交文件的闭环。模型通过了进入 P2 的预设门槛。

## 数据与特征

- 训练样本：1,257,637；测试样本：647,896。
- CanonicalStore：训练 51 个、测试 26 个连续 `sample_id` 分片。
- 特征：Market 371、Order 188、Transaction 134、Cross 42，共 735 列。
- 训练/测试特征名称、顺序和 `float32` 类型完全一致。
- 原始七张 Feather 的大小、修改时间和 SHA256 在全量流程后保持不变。
- 全量缓存占用：Canonical 约 24.106 GiB，特征 Parquet 约 3.204 GiB。

## 单信号基线

| 信号 | 全量 cosine |
|---|---:|
| 10 秒主动成交量 imbalance | 0.068452 |
| 最后 microprice deviation | 0.047584 |
| 60 秒 mid momentum | -0.002644 |

## LightGBM OOF

Run ID：`lgbm-d2ad087ac6787e68`

| 指标 | 结果 |
|---|---:|
| OOF 覆盖行数 | 424,504 |
| OOF 整体 cosine | 0.151838 |
| 四折 cosine 均值 | 0.153081 |
| 逐月 cosine 均值 | 0.148695 |
| 最差月份 | month 70 / 0.117951 |
| 全量训练轮数 | 313 |

| Fold | 模型 cosine | 当折最佳单信号 |
|---|---:|---:|
| 0-46 -> 47-54 | 0.147031 | 0.071868 |
| 0-54 -> 55-62 | 0.146252 | 0.074150 |
| 0-62 -> 63-66 | 0.181387 | 0.083099 |
| 0-66 -> 67-70 | 0.137651 | 0.058307 |

四折平均和最后两折均超过最佳单信号，`baseline_gate.passed=true`。

## 最终产物

- OOF、模型和指标：`artifacts/runs/lgbm-d2ad087ac6787e68/`
- 三种子平均测试预测：`artifacts/runs/lgbm-d2ad087ac6787e68/test_prediction.npy`
- 提交文件：`artifacts/submissions/lgbm-d2ad087ac6787e68.csv`

提交文件已经验证：647,896 行、`sample_id=0..647895`、预测全部有限、非零范数，
且 CSV 数值与 NPY 预测一致。

## Kaggle 提交记录

- 提交时间：2026-07-28 11:56:57 UTC。
- 描述：`P1 LightGBM 735 features rolling OOF 0.15184`。
- 状态：`SubmissionStatus.COMPLETE`。
- Public Score：`0.129`。
- 提交后即时名次：49 / 76；排行榜会随其他队伍提交而变化。

## 复现命令

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m mscapital.cli --config configs/base.toml build-cache --split both --table all --resume
python -m mscapital.cli --config configs/base.toml build-features --split both --resume
python -m mscapital.cli --config configs/base.toml run-baselines --resume
python -m mscapital.cli --config configs/base.toml train-tabular --resume
python -m mscapital.cli --config configs/base.toml predict-test --run-id lgbm-d2ad087ac6787e68 --resume
python -m mscapital.cli --config configs/base.toml make-submission --run-id lgbm-d2ad087ac6787e68 --resume
```

## P2 状态

配置已经固定为单进程、单 GPU、RTX 5070 Ti 16 GB 最低兼容目标；RTX 5090 复用同一
架构。当前执行机器没有安装 PyTorch，也未检测到 NVIDIA GPU，因此尚未执行 P2 的
CUDA 前向/反向和峰值显存验收。P2 应在目标 GPU 机器上新增 CUDA PyTorch 后进行，
不得把未实测的显存结果标记为通过。
