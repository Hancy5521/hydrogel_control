原单位风险场的单一固定机制门：程序和结果阅读说明

本门是什么
这里只有12个预先锁定的人工数值评价：8个机制例及4个使用相同原单位目标的Gaussian强参照。不读取材料文件、不训练预测器、不运行active campaign，也不在输出后修改context、prior去寻找想要的排序。源码准备不是PASS；实际执行和独立审计请看真实protocol、START、RESULTS、COMPLETE及独立收据。

两个core模块与薄driver
code/posterior_update.py：经典NIG状态检查、从完整NIW映射conditional NIG、真实已见标签的rank-one精确更新，以及不同材料未来响应的二元t参数。并未用粒子、MCMC或神经网络。相同context的query与target仍是不同物理材料，所以残差独立、共同回归参数不确定性相关，不能按同一cell重复测量处理。
code/activefield.py：正项CDF/SF差平方的Studentt风险核；Gaussian OwenT核及相消时的正Plackett角积分fallback；原单位YJ逆Jacobian、两类解析截断尾、Studentt affine闭式、rho耦合界。它读取当前后验参数，不压缩/丢弃完整posterior。
code/run_gate.py：固定fixture、实际协议/SHA/版本门、四级求积、全部评价与诊断、原始收据，不是新的方法模块。

单位约定必须一致
NIG的响应可以是在warm固定affine后的zY，也可以直接YJ坐标，但activefield的mu、scale明确是affine之前的YJ坐标U。若未来材料NIG在zY拟合，则必须先mu_u=centerY+SDY*mu_z，scale_u=SDY*scale_z，再以nativeY=s0*inverseYJhalf(U)计算风险。rho对同一正仿射转换不变。不能漏掉SD，不能将native Y直接传给在zY拟合的NIG更新。
当前人工NIG m=(0,1)，V=diag(.2,1)，alpha=3，b=1.5，nu=6；它的响应直接按U定义。terminal x=+1/-1，query x=+1/-1.1；所有原单位s0=2。前4例均由同一个合法NIG state产生，每个target scale=sqrt1.1。

为什么要评价风险而非只看相关大小
共享同NIG的两材料未来响应是二元Studentt。条件target的scale随query标准化结果的绝对值变化，有限df时rho=0仍可能有共享尺度信息。所有原单位动作价值用DeltaR=integral Var_Z[F_target|Z(y)]dy；它是模型内的Bayes期望，不保证实际反馈每一步提高材料CRPS。
标准化后K_nu(r,v)=E[(F_cond(v|Z)-F_marginal(v))²]只依nu、rho和v；query自身均值/scale消去。原单位权重s0*s*g'(mu+s*v)仍依target位置和scale。rho/v均有对称性，可为读出复用提供候选，但本门没有计时证实风险场压缩或实际active收益。

12例与公平参照
1–4为qplus/qminus×tplus/tminus四个native Studentt风险；5仅对第1例翻rho符号；6 Studentt nu6/rho0、mu1/scale sqrt1.1/s02的native值；7相同mu/scale/rho0的Gaussian，独立性使其值精确0（这里不把相同scale叫moment match）；8同参数Studentt rho0的affine目标g(U)=U，仍乘s0=2，可与独立闭式核对。
9–12为前4例的latent mean/covariance匹配Gaussian，sdG=st*sqrt(nu/(nu-2))，相同nativeYJ目标与rho。这是必要强参照，不能只比较native Studentt和错误的latent objective后声称新policy。
所有rankings都保留：native Studentt、native matchedGaussian、affine Studentt、affine matchedGaussian。单branch等cost的affine Studentt和matchedGaussian排序本就相同；rho0正值不构成选点创新。native排序即使翻转，也可能两个模型都翻；不能据此归因finite-df独有机制。排序不翻转、数值不充分、没有速度收益均是可接受的真实结论。

求积和误差分别记账
固定outer standardized v∈[-32,32]，在0与-mu/scale（若在范围内）分段，每段Gauss–Legendre32/64/128/256节点。Studentt query Z限制[-64,64]，用CDFprobability区间上的GL积分，保留原区间概率质量，不重新归一化。v>=0用SF差，v<0用CDF差再平方，避免尾部两个CDF都变1。
Gaussian有OwenT一维公式；若相消使raw核为负或小于64eps乘其两个正量之和，则用正Plackett角积分取代那个节点，保存raw负值、fallback与指数underflow计数。不用硬clamp把负值改0，也不把IEEE underflow显示0说成严格数学0。Gaussian query已解析积分，故无Studentt式query截断误差。
解析v/z截断上界与rho耦合界是模型数学量；它们不覆盖有限GL、CDF实现或IEEE舍入。末级差<=2e-5仅为convergence proxy，解析tail预算<=1e-4，独立不同坐标参考agreement<=2e-4+2e-5relative。数字未达到标准就保留未达标，不改prior/context寻胜。
耦合证明必须对每个同Z的conditional risk用W1后再取E_Z；无条件两个U_r边际相同，不能错误用无条件W1=0。inner截断不得renormalize。rho bound probes只比较本12例中同target marginal的既有例，不另扫参数网格。

结果文件
RESULTS.json的evaluations[12]各保留fixture、4个levels、每段积分与kernel诊断、tail及末级差；conditional_update_probes对原4 NIG pair、各z=-1/0/1检查“先真实label更新NIG再预测”与二元t条件化等价，均为人工数学例，不是新材料。hypothetical label仅用于这个数学核等价测试，不进入任何真实训练。
summary.rankings列qplus/qminus的平均值、signed差与数值proxy；resolved_on_declared_proxies_only不是全球严格证书。rho_coupling_probes保留各解析界与观测差。COMPLETE是程序完成，不自动代表排序反转、计算胜出、独立审计完成或投稿条件达标。
主4个真实反馈材料source及其224组结果保持历史冻结。这个新stage与它分开，防止将人工机制证据冒充真实材料收益。

运行入口（必须先真实注册与批准）
复制EXECUTION_PLAN_TEMPLATE.json为新协议，保持configuration/source/root_scope原样，填REGISTERED和真实registered_actual_utc。root批准scope=NATIVE_RISKFIELD_FIXED_MECHANISM_GATE，allowed_phases=["math_gate"]，plan_sha256为实际协议SHA。随后：
 python -X utf8 -B code/run_gate.py --plan <registered.json> --approval <actual_approval.json> --output-dir <不存在的新目录>
禁止用prepare-only或SOURCE_FREEZE冒充已执行。实际数学门完成后，才由结果决定是否开展新组的主动真实Y查询或是否需要另行登记数值改进；不因一次kernel评价完成就宣称研究已完成。
