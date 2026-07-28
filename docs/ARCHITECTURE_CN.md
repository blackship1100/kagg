# MSCapital 工程架构

## 设计原则

1. **原始数据只读**：`data/` 永远作为不可变输入，清洗结果写入带版本的缓存。
2. **I/O 与算法分离**：Feather 访问不进入特征公式，模型不直接读取原始文件。
3. **纯函数优先**：清洗、序列校验和特征公式尽量只依赖输入数组，便于单元测试。
4. **训练折隔离**：裁剪阈值、标准化参数和聚类器只在当前训练月份拟合。
5. **数据契约先行**：字段名、Arrow 类型、时间方向、分类编码和缺失语义显式声明。
6. **产物可追踪**：缓存、模型、OOF 和提交均记录配置指纹、代码版本和数据版本。

## 分层结构

```text
configs/                    实验参数，不包含业务代码
docs/                       规划、架构和实验规范
src/mscapital/
  config.py                 TOML 配置读取和路径解析
  contracts.py              跨模块公共枚举和数据结构
  data/
    schema.py               Feather 字段契约
    catalog.py              数据文件定位、读取和结构校验
    cleaning.py             缺失哨兵、修正记录和稳健变换
    canonical.py            按列、按 sample_id 对齐的 NPY 分片缓存
    sequence.py             sample_id 边界与时间顺序
  features/
    base.py                 无状态特征模块统一接口
    store.py                Zstd Parquet 特征缓存和完整性清单
  validation/
    splits.py               月份滚动切分
  models/
    base.py                 模型适配器接口
    lightgbm.py             CPU LightGBM 适配器
    preprocessing.py        只在当前训练折拟合的裁剪与缺失统计
  training/
    tabular.py              四折三种子 OOF 与全量训练
    submission.py           提交文件验证与原子写入
  cli.py                    可组合的命令行入口
tests/                      与 10 GB 原始数据解耦的快速测试
artifacts/                  本地生成，不进入版本控制
```

## 数据流

```text
只读 Feather
    -> DataCatalog / Schema Contract
    -> Sample spans
    -> Stateless cleaning + semantic masks
    -> Feature blocks / Sequence builders
    -> Versioned cache
    -> Rolling month folds
    -> Model adapters
    -> OOF predictions
    -> Ensemble optimizer
    -> Submission validator
```

## 模块边界

### DataCatalog

只负责文件路径、列投影、内存映射和字段契约。它不知道任何特征公式，也不负责填充缺失值。

### Cleaning

只负责解释原始值：

- `transaction_avgprice` 空值与无成交；
- 盘口 `price=0, volume=0` 缺失档位；
- 负成交量修正记录；
- signed-log、裁剪和异常标记。

清洗层不执行按月拟合，依赖训练分布的裁剪阈值由后续 Transformer/FeatureBlock 管理。

### FeatureBlock

每个特征族只实现无状态 `transform`。输出统一的 `FeatureMatrix`，包含 `sample_id`、二维数值矩阵和稳定的特征名。依赖训练分布的分位数裁剪与缺失统计由 fold-level preprocessor 负责，不能写入全局特征缓存。

Market、Order、Transaction 和跨表特征可以单独运行、缓存和消融。

### ModelAdapter

模型只接收已经构造好的矩阵或序列，不访问 Feather。所有模型统一暴露 `fit` 和 `predict`，从而让验证器和融合器不依赖 LightGBM、PyTorch 等具体框架。

## 缓存约定

计划采用以下结构：

```text
artifacts/
  cache/canonical/v1/<scope>/<split>/<table>/<source_fingerprint>/
  cache/features/v1/<scope>/<split>/<dataset_fingerprint>/<block>/<version>/
  runs/<run_id>/config.json
  runs/<run_id>/metrics.json
  runs/<run_id>/oof.parquet
  runs/<run_id>/models/
  submissions/<run_id>.csv
```

缓存键至少包含：输入文件大小和修改时间、源文件 SHA256、缓存/特征代码版本、窗口参数与清洗配置。训练 run 另外包含 folds、seed、模型参数和特征 content digest。禁止用模糊文件名覆盖已有结果。

## 单卡深度学习边界

P2 才新增 CUDA PyTorch。Market 保留最多 212 步并使用小型 TCN/1D-ResNet；Order 和
Transaction 默认最近 256 个事件加 60 个一秒聚合格。默认 hidden size 128、4 个轻量
attention 层，5070 Ti 从物理 batch 32 开始并通过梯度累积达到有效 batch 256。
不维护 PRO 6000、DDP、Slurm 或多节点分支。

## 测试策略

- **单元测试**：纯 NumPy 小数组覆盖所有特殊语义和边界。
- **契约测试**：真实 Feather 只读取元数据，检查文件、列和类型。
- **小样本集成测试**：固定少量 `sample_id`，跑通读取、特征、训练和预测。
- **回归测试**：固定样本的特征摘要、fold 行数和基线 cosine 不可无意变化。
- **提交测试**：行数、主键、有限值、顺序和预测范数必须合法。

单元测试不依赖 Kaggle 登录、GPU 或完整原始数据，确保修改任意模块后都能快速执行。
