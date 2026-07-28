# MSCapital GPU 电脑接手说明与后续优化路线

更新日期：2026-07-29

## 1. 接手目标

另一台带 NVIDIA GPU 的电脑接手后，不需要重新设计工程。当前代码已经完成：

- 783 列聚合特征 LightGBM 强基线。
- Market TCN、Transaction Event Transformer、Transaction Grid TCN 和门控融合。
- 月份滚动 OOF、测试预测、断点恢复、显存 OOM 自动减半 batch。
- Deep 与 LightGBM 的 OOF 对齐、权重搜索和测试预测融合。
- 全量 train/test 序列缓存构建和真实数据 CPU 冒烟。

GPU 电脑的首要任务是得到可信的四折 Deep OOF，而不是立即扩模型。只有确认 Deep 单模或
Deep + LightGBM 融合有效后，才进入 Order Encoder、加宽模型和多 seed 阶段。

## 2. 当前可复现基线

### LightGBM P1.1

- 聚合特征：783 列。
- 四折 OOF cosine：`0.153723`。
- Public Score：`0.132`。
- 查询时约第 36 名，榜首约 `0.154`。
- seed 17 Run：`lgbm-5cd1b19ae1c1d824`。
- seed 43/97 Run：`lgbm-530a84ed6162c258`。
- 三 seed 集成：`ensemble-5da978eaa0f118c1`。

### Deep P2 第一版

- 生产模型参数量：`1,774,465`。
- Market：212 步、15 特征、5 层多尺度 TCN。
- Transaction Event：最近 256 条、6 个连续特征、方向 embedding、4 层 Transformer。
- Transaction Grid：完整 60 秒、每秒 10 特征、4 层 TCN。
- Loss：默认原始 target 的 MSE，可选方差缩放 batch cosine。
- 训练：AdamW、AMP、梯度累积、clip norm 1.0、validation cosine 早停。
- 正式四折 Deep OOF 尚未运行；CPU 的 1,000 样本随机拆分结果仅用于工程验收。

## 3. Git 能带走什么

代码仓库：`https://github.com/blackship1100/kagg`

核心 Deep 实现提交：`dbbc2ef Add single-GPU deep sequence pipeline`。接手时应使用最新的
`origin/main`，因为接手说明和后续修订可能位于更新的提交。

Git 包含：

- `src/` 全部训练、缓存、模型与融合代码。
- `configs/base.toml` 默认单卡配置。
- `tests/` 41 项单元和集成测试。
- `docs/` 中文架构、结果和接手说明。

Git 不包含：

- `data/` 原始竞赛数据。
- `artifacts/cache/` Canonical、聚合特征和深度序列缓存。
- `artifacts/runs/` LightGBM/Deep 模型、OOF 和 test prediction。
- `artifacts/submissions/` 提交文件。
- Kaggle Token 或其他凭据。

禁止把原始数据、20 GB 序列缓存、模型 checkpoint 或 Kaggle Token 直接提交到 GitHub。

## 4. 需要单独迁移的本地产物

当前电脑实测大小：

| 路径 | 文件数 | 大小 | 是否必须 |
|---|---:|---:|---|
| `data/` | 15 | 9.294 GiB | 必须，或从 Kaggle 重下 |
| `artifacts/cache/canonical/` | 4,227 | 24.159 GiB | 可重建 |
| `artifacts/cache/features/` | 696 | 3.216 GiB | 继续 LightGBM 实验时需要 |
| `artifacts/cache/sequences/` | 1,095 | 20.171 GiB | 正式 Deep 训练需要 |
| 两个 P1.1 LightGBM Run | 106 | 约 0.098 GiB | 复现模型时建议复制 |
| P1.1 Ensemble Run | 4 | 约 0.009 GiB | Deep 融合时必须或重新生成 |

最省事的迁移方式是把整个项目放到与当前相同的绝对路径：

```text
D:\VibeCoding\kaggle\current_competitions\mscapital
```

使用能保留文件修改时间的复制工具复制 `data/` 和 `artifacts/`。相同绝对路径与原始文件时间
可以直接复用现有 CanonicalStore。若新电脑路径不同，CanonicalStore 的源身份会变化，建议只复制
原始数据和 Sequence 缓存，然后在新路径重建 Market/Transaction canonical manifest。

### 最小迁移方案

只复制：

1. `data/`。
2. `artifacts/cache/sequences/`。
3. `artifacts/runs/ensemble-5da978eaa0f118c1/`。

在新电脑先重建 Market/Transaction canonical，再验证已复制的 Sequence 缓存：

```powershell
python -m mscapital.cli --config configs/base.toml build-cache `
  --split both --table market --resume

python -m mscapital.cli --config configs/base.toml build-cache `
  --split both --table transaction --resume

python -m mscapital.cli --config configs/base.toml build-sequences `
  --split both --resume
```

如果 Sequence fingerprint 一致，最后一条命令只做校验；不一致时会安全重建。当前电脑从 canonical
构建完整 train/test Sequence 缓存耗时约 5 分 36 秒。

### 完全重建方案

只复制或重新下载 `data/`，然后运行：

```powershell
python -m mscapital.cli --config configs/base.toml build-cache `
  --split both --table all --resume

python -m mscapital.cli --config configs/base.toml build-features `
  --split both --resume

python -m mscapital.cli --config configs/base.toml build-sequences `
  --split both --resume
```

## 5. 新电脑环境初始化

建议使用本地 NVMe SSD、Windows 高性能电源模式和较新的 NVIDIA Studio/Game Ready 驱动。

```powershell
git clone https://github.com/blackship1100/kagg.git D:\VibeCoding\kaggle
cd D:\VibeCoding\kaggle\current_competitions\mscapital

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch 必须使用 PyTorch 官方安装选择器提供的 CUDA 命令。不要直接照搬当前 CPU 电脑的
`torch==2.13.0+cpu`。`requirements-deep-lock.txt` 锁定 Python 包版本，但 CUDA wheel 的来源仍应
以 GPU 电脑当时的驱动和 PyTorch 官方说明为准。

安装后必须确认：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期 `torch.cuda.is_available()` 为 `True`，设备名与实际 5070 Ti、4090 或 5090 一致。

运行回归测试：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

当前基准是 `41/41` 通过。

## 6. GPU 上的第一天工作顺序

### 第一步：生产模型冒烟

```powershell
$env:PYTHONPATH = "$PWD\src"

python -m mscapital.cli --config configs/base.toml smoke-deep `
  --max-samples 1000 --resume --device cuda --full-model
```

检查：

- `device` 必须是 `cuda`。
- `amp` 必须是 `true`。
- `peak_vram_gb` 必须小于 15 GB。
- checkpoint 重载后的预测必须一致。

### 第二步：确定 physical batch

保持 effective batch 先固定为 256，仅调整 physical batch 和梯度累积：

| GPU | 首次设置 | 第二次尝试 | effective batch |
|---|---:|---:|---:|
| RTX 5070 Ti 16 GB | 32 | 64 | 256 |
| RTX 4090 24 GB | 64 | 128 | 256 |
| RTX 5090 32 GB | 128 | 256 | 256 |

先不要同时扩大 hidden size、事件长度和 batch，否则无法判断速度、显存或 OOF 变化来自哪里。

### 第三步：正式四折 seed 17

```powershell
python -m mscapital.cli --config configs/base.toml train-deep `
  --seed 17 --epochs 20 --resume --device cuda
```

模型会按 Fold 1 到 Fold 4 顺序训练。每折完成后 checkpoint、valid prediction 和 metadata 都已
原子保存，可以随时中断并用同一命令 `--resume`。建议 Fold 1 完成后先检查指标；如果 loss 发散、
prediction 近常数或 cosine 明显异常，应停止并排查，而不是盲目跑完四折。

### 第四步：测试预测、融合和提交

```powershell
python -m mscapital.cli --config configs/base.toml predict-deep-test `
  --run-id <deep-run-id> --resume --device cuda

python -m mscapital.cli --config configs/base.toml blend-tabular-deep `
  --tabular-run-id ensemble-5da978eaa0f118c1 `
  --deep-run-id <deep-run-id>

python -m mscapital.cli --config configs/base.toml make-submission `
  --run-id <blend-run-id>
```

## 7. 5070 Ti、4090 和 5090 的预计速度

以下是针对当前 1.77M 参数模型、AMP、NVMe mmap 和 effective batch 256 的保守范围。第一轮
GPU 冒烟后应使用真实 samples/s 更新估算。

| GPU | 建议 physical batch | 相对 5070 Ti | 四折各 1 epoch | 常见早停总耗时 |
|---|---:|---:|---:|---:|
| RTX 5070 Ti 16 GB | 32-64 | 1.0x | 2-4 小时 | 10-30 小时 |
| RTX 4090 24 GB | 64-128 | 1.4-1.8x | 1.2-2.5 小时 | 6-20 小时 |
| RTX 5090 32 GB | 128-256 | 2.0-3.0x | 0.6-1.5 小时 | 4-12 小时 |

“常见早停”假设每折大约 5-8 epoch。若四折全部跑满 20 epoch，保守估计为：

- 5070 Ti：40-80 小时。
- 4090：24-50 小时。
- 5090：15-32 小时。

当前模型较小，5090 在 physical batch 32 时会明显吃不满。5090 的优势只有在 batch 提高、
DataLoader 不成为瓶颈后才会充分体现。4090 对当前第一版已经足够，通常可以过夜完成一轮；
5090 更适合后续加入 Order Encoder、事件长度 512、多 seed 和并行消融。

## 8. 判断 Deep 是否有价值

不能只看 Deep 单模 cosine。最终目标是提升 LightGBM 的 OOF 方向，因此至少检查：

1. Deep 整体和逐折 cosine。
2. Deep 逐月均值、标准差和最差月份。
3. Deep 与 LightGBM OOF prediction 的相关系数。
4. 每折最优融合权重是否方向一致。
5. 统一全局权重的融合 OOF 是否高于 `0.153723`。

一个 cosine 略低、但与 LightGBM 相关性较低的 Deep 模型，可能比“单模高分但高度同质”的模型
更有融合价值。反之，如果 Deep 与 LightGBM 相关性接近 1 且单模更差，不应继续堆 seed。

建议进入下一阶段的最低标准：

- 四折预测全部有限且无零范数。
- 最近两折没有明显崩溃。
- Deep + LightGBM OOF 至少稳定提升约 `0.001`。
- 融合收益不是只由单个月份或单折贡献。

## 9. 后续优化计划（按优先级）

### P2.0：建立可信基线

1. 固定 seed 17、纯 MSE、event length 256、hidden 128。
2. 跑完四折并保存 GPU 吞吐、峰值显存、最佳 epoch 和逐月 cosine。
3. 计算 Deep/LightGBM OOF 相关性与统一融合权重。
4. 只有融合 OOF 提升后才预测 test 和提交。

### P2.1：低成本消融

每次只改一个变量，优先在一个完整折上比较：

1. Market only。
2. Transaction Event only。
3. Transaction Grid only。
4. Market + Event，不使用 Grid。
5. 完整三路 Fusion。
6. `cosine_loss_weight = 0 / 0.05 / 0.1`。
7. dropout `0.05 / 0.10 / 0.20`。

目的不是找单折最高点，而是确认每个分支是否提供稳定增量，以及 Grid 是否真正弥补 256 条截断。

### P2.2：增加 Order Event Encoder

这是当前最值得尝试的结构增量。Order 原始流包含新单、撤单、方向、价格距离和体量，LightGBM
虽然使用了聚合统计，但没有完整保留事件顺序。建议：

- 最近 256 条 Order Event Transformer。
- 完整 60 秒 Order Grid TCN。
- side embedding、action embedding、相对最终 mid 的价格距离、`log1p(volume)`。
- 先与现有 Transaction embedding 合成 Flow embedding，再和 Market 门控融合。

先在单折验证 Order 分支是否提高融合 OOF，再决定是否全四折训练。

### P2.3：扩大容量

只在 P2.0/P2.2 有增益后进行：

- hidden size 128 -> 192。
- Market TCN channel 128 -> 192/256。
- 5090 上 event length 256 -> 512。
- FFN 3x hidden -> 4x hidden。
- 使用 gradient checkpointing 控制显存。

事件长度 512 不应默认开启。先统计被 256 截断的样本比例，并确认 Grid 分支仍不能覆盖这部分信息。

### P2.4：稳健性和集成

1. seed 17 有明确增益后，再训练 43 和 97。
2. 对 checkpoint 做均权，不用 Public Leaderboard 反推权重。
3. 检查 prediction 极值、按月尺度漂移和最差月份。
4. 比较原始预测、轻度分位裁剪和 rank/scale 混合，但所有选择只看 OOF。

### 暂不优先

- 多机、多卡 DDP。
- 直接上大型全注意力 Market Transformer。
- 未经消融就把事件长度提高到 999。
- 使用测试榜分反复调融合权重。
- 一开始就训练三个 Deep seed。

## 10. 常见故障与恢复

### CUDA 不可用

- 检查 `nvidia-smi`。
- 确认安装的是 CUDA PyTorch，不是 `+cpu` wheel。
- 重启终端并重新激活虚拟环境。

### 显存不足

训练器会自动减半 physical batch。若仍不足：

1. 把 `physical_batch_size` 降为 16/32。
2. 保持 effective batch 256，让梯度累积自动增加。
3. 开启 `gradient_checkpointing = true`。
4. 不要先缩短数据或改变模型语义。

### 训练中断

使用完全相同的 config、seed、fold 和 epochs，并加 `--resume`。改变这些参数会生成新的 Run ID，
不能复用旧 checkpoint。

### 找不到缓存

通常是新电脑项目绝对路径、数据修改时间或 CanonicalStore identity 发生变化。不要手工改 manifest；
重新执行 `build-cache --resume` 和 `build-sequences --resume`，系统会按 checksum 安全复用或重建。

### 融合找不到 LightGBM Run

确认已复制：

```text
artifacts/runs/ensemble-5da978eaa0f118c1/
```

该目录只有约 9 MB，至少应包含 `oof.parquet`、`test_prediction.npy`、`metrics.json` 和
`manifest.json`。

## 11. 接手完成检查表

- [ ] `origin/main` 已更新到最新提交。
- [ ] `data/` 8 个正式竞赛文件完整。
- [ ] `validate-data` 全部通过。
- [ ] Sequence train 为 1,257,637 行、51 分片。
- [ ] Sequence test 为 647,896 行、26 分片。
- [ ] `torch.cuda.is_available()` 为 True。
- [ ] 41 项测试全部通过。
- [ ] full-model CUDA smoke 完成且显存低于 15 GB。
- [ ] physical batch 已实测确定。
- [ ] seed 17 四折 OOF 完成。
- [ ] Deep/LightGBM 相关性和融合 OOF 已记录。
- [ ] 只有 OOF 确认提升后才生成 Kaggle 提交。
