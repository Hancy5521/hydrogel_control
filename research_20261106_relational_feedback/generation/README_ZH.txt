联合概率补齐样本接口

joint_sampler.py 接受已冻结 ensemble、N行已知 context_z、共享 response 变换、seed、n_draws，
返回 native_draws[n_draws,N] 和明确的生成来源元数据；return_auxiliary=True 可输出潜在z、
每draw的H/sigma²和分H保存的系数，以便独立数值核查。函数不读任何数据文件、不更新模型。
调用者负责核对被冻结的输入、源码hash和不同材料身份；pure sampler不替代这些来源检查。

每一个draw表示一个完整共同世界：只选一次关系模型H，只抽一次sigma²及一次回归系数beta，
然后让全部N行共用它们，各行再抽独立残差。不能把逐行独立抽H或beta的样本称为联合后验；
那会丢掉关系不确定性带来的跨材料依赖。相同context的两个不同材料仍保留独立残差。
最后按固定response逆YJ映回原单位，完整残差不会被低维编码删除。

这是一项经典后验预测采样能力，不是新增算法或校准结果。所有样本必须标为
statistical_generated，draw数不能增加材料N，也不能直接当新测量回填训练形成自证。
如未来做数据增强，应另立验证协议；此接口没有做数据增强或伪标签模型更新。

SAMPLING_GATE_PROPOSAL.json只复用已冻结人工fixture的q/t两行，以固定seed20261106和
2^17 draws核对混合边缘CDF、潜在均值/协方差及共同参数架构。native只检查CDF，
不以不稳定高阶native样本矩伪造精确验证。当前仅源码与方案，未生成任何draw。
主实验4032份预测的独立核验与96真实报告目标评分优先，本能力不延迟主实验。
