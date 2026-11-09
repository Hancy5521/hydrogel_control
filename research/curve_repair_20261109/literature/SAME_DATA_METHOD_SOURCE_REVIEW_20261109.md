# 同数据公开仓库的方法源码补核

2026-11-09。对象是公开仓库 [ToughClimb/closure-bias-inheritance](https://github.com/ToughClimb/closure-bias-inheritance)，固定 commit **95a2312af82f0ea0107ea06a8695cfba3450a291**。本次只通过官方 GitHub 连接器读取该 commit 的 README、相关目录 tree 与下面两份方法源码；没有执行仓库代码，也没有打开 saved results、notebook 输出、模型权重、原始数据或任何保留配方的黏度。

## 1. 结论

**同一公开数据源不等于抄袭，也不能作为数据首次应用的证据。** 前一轮来源准入已发现此仓库使用相同树脂流变来源；本次针对其方法检查，不能继续只用“数据是公开的”跳过直接领域近邻。

在实际检查的两份源码中，未发现本项目目前的“六锚点物理 Student4 边际 + 冻结 GP/曲率后验相关 + Gaussian copula 联合尾段生成”。检查到的流变方法是温度/剪切坐标上的 log 黏度多项式、神经标量本构表示及确定性物理响应控制；另一份名字带 anchor 的代码处理 PDE 本构的边界先验干预，其 anchor 不是本项目六个实测剪切位置的同义方法。

这是有界源码结论，**不是对整个仓库、全部分支或关联论文作“没有 copula”的穷尽证明**。本轮没有读 `rheology_train.py`、`rheology_evaluate.py` 或所有 revision 源码，因为事先上限是两份直接方法文件。若论文审稿需要更全面对照，可另立范围；本次不以弱关键词反复扩搜。

## 2. 实际读取的文件、版本与范围

### README 与目录

- [README 固定版本](https://github.com/ToughClimb/closure-bias-inheritance/blob/95a2312af82f0ea0107ea06a8695cfba3450a291/README.md)，Git blob `048e66b9ffb43ba71981d5aff4371fa88ddb86eb`。
- 根目录 tree、`src/`、`src/closure_discovery/` 与 `experiments/` tree：仅文件/目录名、类型和 blob/tree SHA，用于定位方法，不进入 `results/`、`reproducibility/` 数据记录或 notebook。

README 描述的主要问题是已知反应扩散 PDE 结构下的未知本构关系发现，区分冻结 surrogate 的忠实压缩与重新拟合观测。它同时说明有流变控制及既有实验记录。README 中的聚合研究摘要按其声明读取，未作为本项目的新实验依据；其提到的逐种子文件、完整观测数组和响应结果链接均未点击。

README 声明 MIT，根 tree 中也存在 LICENSE（blob `9a9844d0149c9ab09d832eb088bc8989387bf0f7`）；本轮未重读许可证全文，未复制其代码进本项目，不把仓库源码许可自动等同于数据或论文许可。

### 方法一：experiments/rheology_core.py

[固定源码](https://github.com/ToughClimb/closure-bias-inheritance/blob/95a2312af82f0ea0107ea06a8695cfba3450a291/experiments/rheology_core.py)，Git blob `400dc94c95b6ddd580d6cc5ae932b24ea3a27c8d`。实际读完整模块，不是只看标题或 downloader。

实际可见内容包括：

- 以温度倒数与 log 剪切率为二维坐标并归一化；输入须满足物理取值范围。
- 总次数多项式 design 和 log 黏度最小二乘拟合；另有两隐层 tanh 的标量黏度网络类。
- 原重复实验组的整体留出工具，以及只在测量区间内的插值/阈值穿越。
- 检查单调应力后的狭缝流积分和未支持低剪切中心区的条件上下界；代码声明来源没有实测流量观测。

这些方法与本项目共享真实流变资料、log 黏度和物理本构建模的一般背景，需要在相同数据的既有使用中如实说明。但该模块没有 Student 后验边际、GP 后验 copula、CDF/逆 CDF 依赖搬运或八维概率密度。其条件物理 band 也不能直接解释成本项目的 90% 后验区间。

### 方法二：experiments/anchor_control_core.py

[固定源码](https://github.com/ToughClimb/closure-bias-inheritance/blob/95a2312af82f0ea0107ea06a8695cfba3450a291/experiments/anchor_control_core.py)，Git blob `044219c105709c592df5581071e3f6d3a6b0fdd5`。实际读完整模块。

这里的 unanchored / soft / hard 控制针对反应项在边界点的先验约束；使用观测 PDE 快照、前向解算与 least-squares 参数拟合，比较相同预算下边界约束的影响。源码提供的是通用求解/拟合函数，没有内嵌本项目未来配方的黏度数据。**不能因为都有 anchor、conditional、observations 等名字，就把它等同于六锚点材料曲线概率补齐。** 本模块也没有上述 Student-Gaussian-copula 联合生成机制。

## 3. 对当前贡献归因的影响

应保留三层出处：

1. 原树脂数据的作者、DOI/Zenodo 与许可。使用相同公开数据开展新任务，必须引用原来源；不能称本项目首次公开或生成了这些测量。
2. 该仓库已在同数据上开展本构/物理响应工作。本次检查没有发现相同完整概率生成方法，因此同数据不自动构成实质方法雷同；但可将其列为相关数据应用，不能用搜不到本项目名来证明创新。
3. Gaussian copula / copula process 与边际依赖解耦的数学先例，应引用 Wilson–Ghahramani、WGP、ECC 等已核来源。本仓库未发现相同方法，并不抵消那些非常直接的数学先例。

适当的材料贡献仍是具体任务与证据：以固定六个真实读数保留物理 Student 边际、借助原完整源条件相关生成整段缺失曲线，在同信息强控制和事先冻结的新配方上报告联合评分增益与取舍。不能扩写成首次概率流变建模、首次本构重建、首次使用该数据或新的 copula 理论。

## 4. 标签隔离与本次动作边界

本次未读取该仓库的逐种子 CSV/JSON/NPZ、同观测模型描述文件、保存权重、原始报告或表格结果。只读了 README 的公开摘要和两份不含具体测量数组的方法源码；没有通过外部结果提前看本项目四 final 配方/20runs 的数值，也没有执行任何 downloader、训练器、评分器或求解器。

前一轮仅在 Hancy5521/polymer_completion 名称范围内未检到记录，只能作为本项目消费身份背景，不能作为文献/源码原创性证明。本次在相同数据的公开仓库中做了实际方法阅读，按实际范围记录。若后续发现更接近的方法，仍须诚实补充对照和引用，而不是为了突出本项目忽略它。

所有本次 GitHub 读取成功；没有新的网络重试或材料实验。链接均绑定同一 commit，不以最新分支内容冒充当时版本。
