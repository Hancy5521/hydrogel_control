# 下一机制审查：从固定采点收益到可证伪的修复价值

实际检索日期：2026-11-09；本记录写作依据的最近工具时间为 UTC 03:30:41（北京时间 11:30:41）。角色：独立文献/方案审查。没有读取材料黏度、运行模型、拟合、评分、选择实验超参或改动冻结内核。本记录在材料 pilot 结果未知时提出判断分支，不是事后解释。

## 1. 结论与建议

保留 MLST 的完整高维参考分布与局部曲率 Gaussian 修正，以新任意 mask 接口为基础。当前最值得检验的增量是：**在相同读数预算下，把修复动作的潜在损害提前显现，并只在真实已知读数支持时采用修复；最后检验真实未读部分是否得到帮助。** 不急于将固定 Gaussian 的采点规划命名为自催化、强化学习或智能体。

“部分观察 → 概率恢复 → 选择新读数 → 再恢复”有直接、成熟的算法先例；这不否定针对材料任务的适配与证据贡献。应将本项目的可检验差异放在材料曲线的缺口几何、参考转移失配、预算内真实 sentinel 与完整联合概率输出上。当前有意义的目标可以是追平强基线、减少有害适配、或在较难缺口中获得同预算增益；不要求所有材料、所有指标领先。

Eigen 超循环可以解释“修复规则与测量反馈相互支持”的设计启发。只有真实外部读数或独立求解器输出才增加证据；模型生成值不能给同一模型正确性投票。没有实测的催化动力学，不把循环调度器描述成已实现的 Eigen 超循环。

## 2. 四项定向原始来源：实际阅读及边界

本轮只审查四项直接相关工作，没有重复昨天的完整检索表，也不声称完成广泛查重或全文复现。检索式依次为 `site.proceedings.mlr.press "Active Feature Acquisition" "Generative"`、`site.proceedings.mlr.press "EDDI" "Partial VAE"`、`site.proceedings.neurips.cc "Meta-Weight-Net"`。第四项来自检索结果中的直接近邻，再进入原作者 arXiv 正文核对。

### 2.1 EDDI：任意观测子集和目标信息采集已有直接先例

Chao Ma 等，ICML 2019，PMLR 97:4234–4243。[正式论文页](https://proceedings.mlr.press/v97/ma19c.html)、[主文](https://proceedings.mlr.press/v97/ma19c/ma19c.pdf)、[附录](https://proceedings.mlr.press/v97/ma19c/ma19c-supp.pdf)。正式版本固定为 ICML 2019；未把搜索引擎的相对“发布时间”误作论文年代。

实际读到：正式摘要/引文、附录 A.1 的 acquisition 期望及其潜变量近似、B.1–B.2 部分设置；主文 PDF 文本在一次 find 中未抽取成功，因此不声称通读主文。EDDI 将 Partial VAE 的任意条件补全与目标变量的信息增益相结合，并比较随机及单一固定排序。这直接覆盖“低维编码 + 看到局部后补全 + 再选读数”的一般组合。我们的 Gaussian CRPS 采点是不同损失与材料曲线实例，不因此成为首次主动补全。

官方 [microsoft/EDDI](https://github.com/microsoft/EDDI) 通过 GitHub 连接器读取了仓库元数据、根目录、README 和 LICENSE；未执行或复制代码。README blob `a1048afb9fcc26db3ab05b87e8d25497972d3c86`；LICENSE blob `dab456600062df5fe153273e06381caf3d2b5a40`；主分支 master，元数据 pushed_at 为 2023-06-12T18:56:30Z，仓库已归档。许可是 Microsoft Research License，而非 MIT；其研究使用与再分发条件应分别处理，本包不纳入该仓库源码。论文的公开访问不等于已核定论文再分发许可。

### 2.2 GSMRL：生成模型、主动补全和闭环决策不是空白领域

Yang Li、Junier Oliva，ICML 2021，PMLR 139:6450–6459。[正式页](https://proceedings.mlr.press/v139/li21p.html)、[主文](https://proceedings.mlr.press/v139/li21p/li21p.pdf)。实际读取主文方法 §2.1–2.3、Algorithm 1、AIR 任务及时间顺序约束，未读完所有实验/附录。

该工作让生成 surrogate 为主动特征采集提供条件分布、潜在信息收益和中间奖励；AIR 更直接研究通过部分特征恢复其余特征。因而“生成—验证—主动查询”本身不能作为本项目与所有 ML 工作的区别。它也提醒我们：在真实时间序列中不能查询过去已错过的时刻；流变剪切坐标是否能任意往返，需由实验协议决定，不能仅凭离线 CSV 的随机访问能力推断。

官方 [lupalab/GSMRL](https://github.com/lupalab/GSMRL) 连接器读了元数据、GSMRL 分支根目录与 README，README blob `61f95d75300f2dfa9699477fe7eb83293fba3821`。README 的训练流程是 ACFlow surrogate 后接 PPO；未执行。根目录未见 LICENSE，API license 为 null；不推断可复用许可，不复制代码。当前材料小样本试验不需要先完整移植 PPO。

### 2.3 Meta-Weight-Net：独立真实反馈指导更新也有清楚先例

Jun Shu 等，NeurIPS 2019，Advances in Neural Information Processing Systems 32。[正式页](https://proceedings.neurips.cc/paper/2019/hash/e58cc5ca94270acaceed13bc82dfedf7-Abstract.html)、[主文](https://proceedings.neurips.cc/paper/8467-meta-weight-net-learning-an-explicit-mapping-for-sample-weighting.pdf)。实际读取摘要、问题与元训练目标，以及官方代码中一次可微候选更新—meta loss—权重网络更新的相关段落；未验证全部收敛定理。

它从小的独立 meta 数据学习损失到样本权重的映射，表明“训练更新是否有利，应由另一组真实读数指导”是可引用的已有机制。本项目可以研究连续材料曲线上的修复损害预警和预算约束适配，但不将梯度对齐或单步 meta 更新本身宣称全新。

官方 [xjtushujun/meta-weight-net](https://github.com/xjtushujun/meta-weight-net) 读了 README、LICENSE 和 `MW-Net.py` 的相关训练段落，未执行。README blob `029a028b8ca5ead646c135f32a3d563cf03a6e0a`，源码 blob `bd54f2ab56b2b74483beb4fb47cce21703ae25cb`，LICENSE blob `537d6417a5a81d4fb1566365b016c0af5812b7f6`，许可 MIT；默认 master，元数据 pushed_at 2021-12-14T03:17:23Z。本项目当前未引入其源码依赖。

### 2.4 不完整训练数据下的主动采集：必须区分恢复值与可查询值

Reza Rezvan 等，*Active Feature Acquisition With Incomplete Training Data*，[arXiv:2609.32325v1](https://arxiv.org/abs/2609.32325v1)，2026-09-26；[作者正文](https://arxiv.org/html/2609.32325v1)。这是预印本，本轮未核得会议录用，不把它当作已发表顶会论文。HTML 明示 CC BY 4.0。

实际阅读摘要、§3.1–3.4、§4.1 和 §4.2 开头；未通读后续证明、全部实验。其分析在 MCAR 条件下，训练端不可用特征限制多步动作空间，单步价值不变并不意味着多步价值不受损。对我们最直接的提示是：人工遮挡已测曲线与从未测过的稀疏性质表不是同一训练环境。生成恢复可以帮助训练策略，却不会把其输出变成真实可验证测量；MCAR 结论也不能扩写成解决真实材料 MNAR。

作者给出的 `Linusaronsson/AFA-Benchmark/tree/missing-data` 连接器读取失败（`AttributeError: 'str' object has no attribute 'get'`）；仓库根读取返回 Moved Permanently，浏览器备用该分支又返回 restricted URL。此分支到此关闭，没有反复重试；尚未核定代码 commit/许可/训练细节，本项目没有下载或使用该代码。

## 3. 固定 Gaussian 效用为何不是值自适应

对固定参考参数和已知独立观测噪声，观测集合为 O、已知值为 y_O。条件协方差

\[
K^O=K-K_{:,O}(K_{O,O}+R_O)^{-1}K_{O,:}
\]

只依赖 O、K 和 R_O，不依赖 y_O。固定目标 panel T、候选读数 q 的 CRPS 模型期望收益为

\[
G(q;O,T)=\frac1{\sqrt\pi}\sum_{j\in T}w_j\left[
\sqrt{K^O_{jj}}-
\sqrt{K^O_{jj}-\frac{(K^O_{jq})^2}{K^O_{qq}+r_q}}
\right].
\]

因此在参数/噪声/成本不随读数更新时，整个 greedy 查询次序可以在读取黏度前预先算好。更新了后验均值并不改变这个事实。将其称为“顺序条件采点”准确；称为“根据当前材料状态实时学会修复”则需要另外证明读数确实改变决策。上式是标准 Gaussian 决策理论在当前 proper score 下的实例，不是新概率理论。

更重要的是，G 是模型正确时的期望，并不保证实际收益。新来源的均值、相关长度、噪声或跨位置相关性失配，均可让模型宣称有益、真实评分却变坏。昨日保留的错误相关结构反例具有实质价值，不应为突出新意删去。

## 4. 这次固定 tail 试验还需要最简单的几何对照

当前 T 固定为 native ranks 18–25，初始读数 ranks 1、2，四个新增点来自 ranks 3–17。若 expected-CRPS 只是在 tail 边界附近采点，它胜过 first-six 不足以说明复杂规划必要。

建议单独登记两个**诊断方法**，不改原十二方法或主终点：GP/curvature 各用 `nearest_target6`，读数 ranks **1、2、14、15、16、17**。索引为 `[0,1,13,14,15,16]`。选择规则只依赖目标位置，全部方法共享原八个评分点，六读数预算一致。

若能在任何目标揭示前完成登记和预测冻结，可报告它是新增的预 Y 诊断。若原 T 已经揭示，才增加它，只能标记该开发集上的 post-hoc 诊断，不能补称新确认，也不能用它替换失败的原主终点。是否实际加入由 root 按当前真实阶段决定；本文件没有执行。

还应报告 mask/坐标：expected-CRPS 是否与 nearest-target、spread 或 max-variance 相同；不同参考上的同信息交叉 mask 是否改变结论。相同 mask 不应包装成两个独立发现。材料实验读数的真实成本随扫速、往返/预剪切、稳态等待而变；当前等点预算只能称离线等数量比较，不能宣称已节约仪器时间。

## 5. 预设结果分支：不是看完结果后挑机制

| 当前 pilot 的可观察结果 | 合理下一步 | 不应作出的解释 |
| --- | --- | --- |
| 采点比 first-six 好，却不及 nearest-target/spread | 保留任意 mask 能力；简化布局；把主要投入转向转移失配或缺口形状 | 复杂采点策略已证明必要 |
| 相同六点下 GP 强于曲率修正 | 原曲率修正有来源适用边界；后续研究是否能识别/抑制有害修复 | 宣称需要立即更大网络或自动加分量 |
| 两参考均有较大误差、模型收益与真实收益失配 | 先检查单位/热历史/趋势与残差；若数据处理正确，再研究来源适配与简单物理控制 | 调最终目标位置、重切配方或搜索随机种子 |
| 效用比几何和 max-variance 有稳定收益 | 保留作为材料采点实证；再在未消费 expansion/final 检验 | Gaussian 效用由此成为新算法或已证普适 |
| 组间收益不同、只有个别 run 赢 | 解释 recipe 级异质性，保留负组；按预设宏平均判断 | 把十五 runs 当十五独立材料或改报最好组 |

## 6. 推荐的下一机制：预算内真实 sentinel 的有害修复预警

这是源代码/实验计划候选，不是已实施结果。先做一个小而明确的机制，暂不加入 RL 或非线性流。若现有资料不足支持训练/开发/确认分离，先扩充新来源，勿在已消费三配方上无限调试。

1. **候选更新从训练配方学习。** 原模型参数 θ₀ 冻结为基线。另存新适配器；在训练侧人工缺口任务得到一个有约束候选方向 d。不同配方之间共享 d，不给每条最终曲线另拟合完整模型。
2. **当前六读数预算内留 sentinel。** 部署/评估时将六个真实可用读数按事先固定规则分为 A 和 S；S 仍属于真实已知读数，但在诊断时先不用于条件化，计算 `L_S(θ | y_A)`。不得多读一组标签后仍声称六点预算。
3. **在同一候选上提前预警。** 计算 `h = ∇_θ L_S(θ₀)ᵀd`，局部预计损失变化为 `α h`。步长 α、警报阈值、方向范数/参数域都在独立开发资料固定。只把它当局部预警，不把负 h 当有限步保证。默认 current `early_influence` 的跨组训练/开发约束仍保留；如新增单条曲线的部署 sentinel 函数，须独立命名和审计，不绕过原接口身份规则。
4. **最终所有六读数用于概率条件化。** 依固定预警规则选择 θ₀ 或 θ₀+αd，再以 A∪S 产生未读 panel 的完整联合分布。这一决策真正依赖观测值，且所有对照得到相同的六个读数。S 可以在决策后用于最终条件化；这是正确计账的已知信息，不是新的独立验证。
5. **严格去模块对照。** 同一候选 θ₀+αd、同参数量、同步长和训练预算，取消预警；再加原 θ₀、GP 和同预算简单物理/趋势参考。若研究早期检查节省计算，需另测 wall time/完整训练调用数；五参数 Gaussian 的导数本来便宜，不能默认获得大加速。
6. **仅在未消费曲线上判断。** 先冻结全部预测和预警记录，再揭示目标。同时报告 CRPS、联合评分/覆盖、警报误报与漏报，以及预警排序对实际损害的作用。S 的反复规则选择归入开发；同一曲线内部 LOO 不是新批次确认。若没有可观测的危险方向，不能靠模型自一致性补上真实证据。

这一方案将用户“边恢复边制造伤口、尽早看出反向影响”的想法落到可检验位置：候选更新可以先由有限的真实已知读数进行损害检查。核心科学问题是**真实材料分布转移中，局部损害信号能否用很小的预算避免不必要的修复**，而不是是否能计算一个梯度。若该机制不能胜过同预算原参考/简单几何，就保留失败并采用较简单方法。

## 7. 输入版本、失败与恢复点

实际读取的本地输入 SHA256：

- `core/API.md`: `b10401595017c001e272172ac033e2d97a36d20b6529cb208a2e81b47510e157`
- `HYPOTHESES_AND_DEV_PILOT_PREPARED_ZH.txt`: `d777b2e0f7ff0c76023d1ff0d98182a67fdef9454dce2e24a7944b0e8d7204b4`
- 先前 `process_repair_20261108/RESEARCH_DECISION_AND_ALGORITHM_ZH.txt`: `6d4398d2b21c7a0276b6ee176fc1074edb57adbed736676d934ce8302c5f13af`
- 先前 `EXISTING_SYSTEM_INTEGRATION_ROADMAP_ZH.txt`: `e34df7c6da369504936856a7a2f1f05713111b6afc0ed7c5272f53c227314825`

接口失败保留：NeurIPS hash 路径 PDF 首次 timeout，标题路径首次成功、后续定位页 timeout；采用已取得正式方法段和官方源码对应段落，不把重试当阅读进展。AFA-ITD 代码分支失败如上。一次本地工具编排在对 connector 错误对象作 JSON.parse 时抛异常；随后读取已保存错误并纠正，只影响输出处理，没有材料动作。

明确恢复点：root 完成零拟合 resin pilot 后，先依据固定主终点、同信息参考比较、nearest-target 诊断和 recipe 级结果，选择上表唯一有依据的分支。只有发现可修复的 transfer 失配且有足够独立训练/开发资料，才登记第 6 节小机制。没有查看任何 pilot 目标或结果来写此方案；不将其保存或文献阅读本身计为新材料收益。
