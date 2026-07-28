# MSCapital P1.1 自动提分结果

生成日期：2026-07-29

## 结论

在 P1 的 735 列聚合特征基础上，新增 48 列订单类别占比特征。三个随机种子的
四折 OOF 全部超过原三种子基线，Kaggle Public Score 从 `0.129` 提升到 `0.132`。

新增特征来自每个 `1/2/5/10/30/60` 秒窗口中的四类订单：

- 买入新单
- 买入撤单
- 卖出新单
- 卖出撤单

每类分别计算事件数占比和体量占比，共 `6 × 4 × 2 = 48` 列。占比直接从已有缓存
中的类别计数和 `volume_logsum` 确定性派生，训练集和测试集使用完全相同的公式，
不需要重建 10 GB 原始数据缓存。

## OOF 对比

| 指标 | P1 基线 | P1.1 | 增量 |
|---|---:|---:|---:|
| 整体 cosine | 0.151838 | 0.153723 | +0.001885 |
| 四折均值 | 0.153081 | 0.155427 | +0.002346 |
| Fold 1 | 0.147031 | 0.147566 | +0.000535 |
| Fold 2 | 0.146252 | 0.146958 | +0.000705 |
| Fold 3 | 0.181387 | 0.187348 | +0.005961 |
| Fold 4 | 0.137651 | 0.139834 | +0.002183 |

OOF 覆盖 424,504 行，四个验证折全部提升，最近两折也同时提升。P1.1 逐月 cosine
均值为 `0.150018`，最差月份为 month 70，得分 `0.119720`。

## 三种子产物

为避免重复训练 seed 17，训练产物拆成两个兼容 Run，再按每个种子等权组合：

- seed 17：`lgbm-5cd1b19ae1c1d824`
- seed 43/97：`lgbm-530a84ed6162c258`
- 三种子组合：`ensemble-5da978eaa0f118c1`

组合器会校验模型参数、预处理、实验设置、时间折、特征摘要和特征名称完全一致。
默认权重按组件中的种子数计算，因此两个组件权重为 `1/3` 和 `2/3`。

## Kaggle 提交

- 文件：`artifacts/submissions/ensemble-5da978eaa0f118c1.csv`
- 行数：647,896
- `sample_id`：连续唯一的 `0..647895`
- OOF：`0.153723`
- 提交状态：`SubmissionStatus.COMPLETE`
- Public Score：`0.132`
- 原 Public Score：`0.129`
- 查询时排名：约第 36 名，榜首 `0.154`

测试预测全部为有限数，范数非零，CSV 与 NPY 数值一致；与 P1 基线测试预测的相关性
为 `0.963449`，说明新方案保留了主信号并加入了有效订单结构增量。

## 已淘汰实验

以下实验只保留离线产物，没有提交：

| 实验 | 结论 |
|---|---|
| 12 个月时间半衰期 | 前两折明显下降 |
| 0.5%/99.5% 标签裁剪 | Fold 1、Fold 3 下降 |
| 删除 79 个覆盖/密度特征 | Fold 4 下降 |
| 删除 43 个绝对密度特征 | Fold 2、Fold 4 下降 |
| Huber，默认 alpha=0.9 | 标签尺度过小，实际等价于 L2 |
| Huber，alpha=0.005 | Fold 3 下降 |
| 按月 target z-score | Fold 3 明显下降 |

## 复现命令

```powershell
$env:PYTHONPATH = "$PWD\src"

python -m mscapital.cli --config configs/base.toml train-tabular --resume `
  --seeds 17 --derived-feature-set order_category_ratios

python -m mscapital.cli --config configs/base.toml train-tabular --resume `
  --seeds 43 97 --derived-feature-set order_category_ratios

python -m mscapital.cli --config configs/base.toml predict-test `
  --run-id lgbm-5cd1b19ae1c1d824 --resume

python -m mscapital.cli --config configs/base.toml predict-test `
  --run-id lgbm-530a84ed6162c258 --resume

python -m mscapital.cli --config configs/base.toml ensemble-runs `
  --run-id lgbm-5cd1b19ae1c1d824 lgbm-530a84ed6162c258

python -m mscapital.cli --config configs/base.toml make-submission `
  --run-id ensemble-5da978eaa0f118c1 --resume
```
