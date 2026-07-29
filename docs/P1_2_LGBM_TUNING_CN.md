# P1.2 LightGBM 调参与融合结果

生成日期：2026-07-29

## 结论

P1.2 保留 P1.1 已验证有效的 783 个聚合特征，使用滚动月份 OOF 做模型筛选，不依据 Public LB 调参。

- 原 P1.1 三 seed OOF：`0.153723`
- 新保守 LightGBM 三 seed OOF：`0.153639`
- 两个模型按单位范数 50/50 融合：`0.154495`
- 融合向量中心化后使用 signed-power `1.10`：`0.155539`
- 相比 P1.1 的绝对提升：`+0.001816`

最终后处理后的四折 OOF 为：

| Fold | P1.1 | P1.2 最终版本 | 增量 |
|---|---:|---:|---:|
| Fold 1 | 0.147566 | 0.147569 | +0.000003 |
| Fold 2 | 0.146958 | 0.147411 | +0.000453 |
| Fold 3 | 0.187348 | 0.190020 | +0.002672 |
| Fold 4 | 0.139834 | 0.142786 | +0.002952 |
| Overall | 0.153723 | 0.155539 | +0.001816 |

## 筛选记录

所有初筛均固定使用 seed 97，并与 P1.1 的相同 seed 比较；不通过的产物仍保留在本地 `artifacts/runs/`，但不进入测试集预测。

| 实验 | 特征/参数变化 | OOF | 结论 |
|---|---|---:|---|
| `lgbm-aa392a4fe694be4a` | 新增订单压力 48 列 | 0.149998 | 淘汰 |
| `lgbm-4bad11132d70afd2` | 新增短长窗口动态 77 列 | 0.150737 | 淘汰 |
| `lgbm-d91433e392595006` | 63 leaves、min leaf 1000、0.9 行/列采样、L2=2 | 0.152712 | 晋级 |

保守树的另外两个 seed 运行是 `lgbm-acf783606d9ec927`；三 seed 平均为
`ensemble-37d57df732a63940`。单独替换 P1.1 的提升不足以稳定超过原模型，
但与 P1.1 的预测相关性为 `0.9789`，存在可利用的互补信号。

## 最终构成

```text
P1.1 三 seed LightGBM                50%
P1.2 三 seed保守 LightGBM             50%
  -> 两路预测分别单位范数归一化后融合
  -> 测试向量自身去均值
  -> sign(x) * (abs(x) / std(x)) ^ 1.10
```

最终运行 ID：`postprocess-c4e1780b90d171bc`。

- OOF：`artifacts/runs/postprocess-c4e1780b90d171bc/oof.parquet`
- 测试预测：`artifacts/runs/postprocess-c4e1780b90d171bc/test_prediction.npy`
- 本地提交文件：`artifacts/submissions/postprocess-c4e1780b90d171bc.csv`

提交 CSV 已通过行数、连续唯一 ID、有限预测和非零范数校验。已通过 Kaggle API 提交：

- 提交编号：`55082138`
- 远端状态：`COMPLETE`
- Public Score：`0.132`

该分数与 P1.1 提交持平。离线 OOF 的提升没有在本次 Public LB 上转化为增益，因此不应
继续依据该单次榜单结果调整融合权重或 signed-power 指数；后续优先进行更稳健的时间分段
消融和深度模型 OOF 融合。

## 新增命令

`train-tabular` 现在可直接覆盖以下 LightGBM 参数：

```powershell
--learning-rate --num-leaves --min-data-in-leaf --feature-fraction `
--bagging-fraction --lambda-l2 --max-bin --max-rounds --early-stopping-rounds
```

增加三组确定性派生特征选择：

```powershell
--derived-feature-set order_category_ratios
--derived-feature-set order_pressure
--derived-feature-set temporal_dynamics
```

跨配置融合和后处理命令：

```powershell
python -m mscapital.cli --config configs/base.toml blend-runs `
  --run-id ensemble-5da978eaa0f118c1 ensemble-37d57df732a63940 `
  --weight 1 1

python -m mscapital.cli --config configs/base.toml postprocess-run `
  --run-id blend-20b67e12810e7bc8 --power 1.10 --center
```

`blend-runs` 与 `ensemble-runs` 的区别是：前者允许模型参数不同的运行参与融合，
但严格校验所有 OOF 的 `sample_id`、`month`、`target` 与 fold 完全对齐。
