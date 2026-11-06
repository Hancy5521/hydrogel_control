混合测度与原始cell台账：有界数学机制检查

这一步的作用

这不是新的材料性能结果，也不是新算法或闭环智能体已经成功。本检查只解决往真实反馈环路推进前的两个一致性问题：
(1) G既可能精确为0、又可能是连续数时，如何在同一模型中使用正确的似然；
(2) 先看哪些性质、把一条记录拆成几张报告，是否会错误改变证据或费用。
程序只使用预先指定的两行合成数值和4个关系假设，不打开材料文件，不训练模型、不随机抽样。720种顺序只是同6个值的反事实计算顺序，不是720个独立材料实验。

固定模型

关系假设h=0,1,2,3，先验w=[0.1,0.2,0.3,0.4]；G为0的概率p_h=[0.1,0.3,0.6,0.8]；相关参数rho_h=[-0.6,-0.2,0.25,0.65]。
零点分支：G=0；(A,Y)服从二维Gaussian，均值(0.2h,-0.1h)，协方差[[1,rho],[rho,1]]。
连续分支：(A,G,Y)服从三维Gaussian，均值(0.2h,0.3-0.1h,-0.1h)，协方差Sigma_ij=rho^|i-j|。这只是合成机制，连续G允许负数，不是对真实带隙物理取值的断言。

两个分支独立定义；零点分支不是把连续Gaussian条件到G=0得到的分布。当G没观察到时，连续分支的(A,Y)边际相关是rho²，而零点分支是rho，因此必须按p_h和1-p_h把两个不同Gaussian边际相加，不能随便挑其中一个。
四个h同时改变均值、p_h和rho_h；因此这些固定案例中的后验权重改变，是在辨认整个联合机制，不能单独解释为“学会了相关方向”或证明新增观测只提供关系信息。

基准测度与似然

完整基准测度是Lebesgue(A) × [delta_0(G)+Lebesgue(G)] × Lebesgue(Y)。只观察一部分坐标时用它在相应坐标的投影测度。
若观察了G=0，Radon–Nikodym密度中只有p_h乘零点分支的已观测(A,Y)边际密度。连续slab在这个原子点的导数为0，不能把Gaussian在0处的连续密度加到p_h原子质量上。
若观察了G≠0，用(1-p_h)乘slab的已观测边际Gaussian密度。负G同样走连续分支。
若没观察G，用p_h*f_zero(observed A,Y)+(1-p_h)*f_slab_marginal(observed A,Y)。完全没观察时，似然为1、log似然为0。
A、Y只要是连续坐标，其Gaussian密度可能大于1；它不是“精确读到该实数的概率”。程序与记录明确保留mass/density区别。

后验与顺序

固定h下r0、r1条件独立，但同一行的A/G/Y相关。
  log L_h = sum_row log f_h(observed coordinates of that row)
  w_h(data) ∝ prior_h * exp(log L_h)
每新增一个cell，正确增量是log f_h(new row marginal)-log f_h(previous row marginal)，即条件似然的log；不是把新性质的一维边际与旧结果直接相乘。
程序用logsumexp形式归一化，保存每一步的4维完整关系后验，没有在这里近似、压缩或丢弃关系假设。
若动作只依赖已观察历史和独立随机性，而不偷偷依赖隐藏h，则同一行动历史的策略概率在关系后验分子分母中抵消。本程序没有实现或验证自适应策略，只验证固定顺序的似然与台账一致性。

物理cell、报告alias与费用

唯一键是(canonical pool row,field)，field为A/G/Y。r0、r1为可购买pool。
rowA与duplicate_report都映射r0，rowB映射r1；这只是同一原始报告身份的别名，不增加材料数或证据数。
held_out是不可购买的terminal row；terminal和terminal_report只是它的别名，不能绕过限制。
每个新物理cell费用1；panel只对真正新增的唯一cell收费。已知同值再次出现，增量费用0、log似然0、后验不变。数值按精确相等比较，不擅自把不同实测值揉为重复。正负0视作同一个0原子。
空panel明确表示合法STOP，费用和证据都为0。
若panel中出现冲突旧值、同panel互相矛盾的alias、未知行、未知field、非有限值或terminal，整笔事务原子拒绝：已经验证过的前几个合法cell也不写入，不收费、不更新后验。

固定计算案例

r0=(A=.7,G=0,Y=1.2)；r1=(A=-.4,G=-.6,Y=.1)。
主检查枚举6个唯一新cell的全部6!=720顺序；保存每种顺序全部6步，总4320步。每步都有完整before/after观察集、4个log似然、4个后验权重、费用、canonical request、新增cell以及条件log似然增量。每种终点与直接joint batch计算比较；完整披露费用应为6。
正值变体只把r1.G换成+.6，固定6种A/G/Y field顺序，每个field依次披露r0、r1，共36步。不看结果另挑数值或顺序。
17个事务案例覆盖同值重复、报告alias、panel内一致重复、整panel重复、部分已知panel、原子冲突拒绝、未知字段/行、NaN/正负无穷、terminal、正负0、STOP。
7个原子/边际控制覆盖空观察、G=0、G=±.6、缺失G的(A,Y)混合边际、完整零点/非零分支。
故意错误对照将6个一维cell边际相乘，保存它与正确同row联合似然的后验差异。这个对照只展示错误分解会改变后验，不用它冒充应被击败的强学习基线。
所有生产端容差预先固定1e-11。它们是float64模型一致性检查，不是严格有理数证明或向外舍入区间证书。

两模块分工

code/mixed_likelihood.py：固定4假设；已观察坐标的Gaussian边际；混合测度log似然；完整后验；CanonicalCellLedger原子事务。Gaussian密度用NumPy Cholesky/solve，不调用SciPy。
code/run_gate.py：固定配置、协议/SHA/授权校验、全部顺序与行为案例、结果/失败和冻结收据。独立审计另用SciPy构建观察子矩阵，不应import这里的似然函数或借用生产端输出做正确答案。
独立审计的数值是否已执行以及结论，必须看实际root批准、原始执行结果和独立审计收据；SOURCE_FREEZE只表示源码准备，不能当PASS。

手动复现入口

本门不加入主reproduce.py的6个材料开发重放模式。它是独立合成数学门，可单独手动重放。
先复制EXECUTION_PLAN_TEMPLATE.json为新的外部协议，保持source_sha256与configuration不变，填写status=REGISTERED及真实registered_actual_utc；输出目录必须是不存在的新目录。root正式研究执行需单独批准，不能用旧批准冒充新执行。
批准JSON字段为approved=true，scope=MIXED_MEASURE_CANONICAL_LEDGER_MATH_GATE，plan_sha256为新协议的真实SHA，allowed_phases=["math_gate"]。本地读者自行重放时应另标注“已消费合成门重复，非新材料证据”，保留新的真实时间。
实际命令：
  python -X utf8 -B mixed_measure_gate/code/run_gate.py --plan <新协议.json> --approval <实际批准.json> --output-dir <新的输出目录>
程序只有在检查精确源码、配置和批准后才执行固定案例。默认资源包校验、主reproduce.py prepare-only或打开本说明都不会触发这个门。

结果结构与诚实边界

execution_once/RESULTS.json中main_720_orders、positive_gap_variant_6_orders、behavior_cases、atom_and_marginal_controls、naive_product_control都保留，不只输出最终PASS。每条order的steps保留before/after完整状态；拒绝request中的NaN/Inf用repr字符串记录，避免生成非标准JSON。
EXECUTION_COMPLETE.json绑定实际输入协议、源码、开始与RESULTS的SHA；如失败则保留FAILED_STOP.json，不覆盖旧结果或为了通过而在同案例上改门槛。
本门不给真实材料模型校准、预测效果、关系参数估计、动作价值、自催化增益或实验节约背书。真实源中的G=0究竟是金属分支、报告舍入还是检测下限仍需来源证据；不能把这里假定的生成机制直接写成已验证的材料事实。
当前来源追溯发现2024版JARVIS API有clamp与unclamp路径，但尚不能将它们映射到2018年Matminer原op_gap字段生成过程。因此这个合成门不能使真实JARVIS的spike模型定型；主报告冻结结论与后续机制补充应分开保存。
