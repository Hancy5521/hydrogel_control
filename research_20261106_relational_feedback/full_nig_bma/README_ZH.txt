完整关系后验与原单位信息价值：source-only 数学门

relation_posterior.py 是三种固定关系的经典 NIG 回归与 Bayesian model averaging：
[1,x1,x2]、加 x1*x2、再加 x1²,x2²。材料阶段若开展，后三项只按 warm 的
mean/sampleSD 固定表示，共享响应变换。初始 prior 为 m0=0,V0=I,alpha0=3,b0=1.5；
warm 数据顺序进入完整预测 evidence，Hprior=1/3，随后每个真实查询先保存 law、
再按旧状态 predictive density 更新模型 logweights 和各自完整 NIG。无 weight floor；
显示为零的概率不丢弃原 finite logweight。此处 fixture 直接给定三个后验，无材料 fit。

risk_field.py 积分的是完整条件混合 CDF 的方差。每个 query 值按所有模型的
预测密度更新 H 权重；各模型内部仍保留完整 bivariate Student-t 条件预测，
没有套用此前固定 Gaussian 的 C:D 特例，也没有只保留模型之间的不确定性。
原单位 Jacobian 是 s0*sd*g'(center+sd*z)。按生成 query 的各 H 分别以
z=sqrt(nu)*tan(theta) 积分，保留原截断密度且不重新归一化。

分解为 full = weighted within-H + Bbefore - E[Bafter]。
最后一项可以为负；特别是此唯一人工 fixture 的 target 初始边缘相同，Bbefore=0，
观测后各 H 的条件 target 可能分开。负号不能删去或当成程序 bug，完整 full 才是
可用于动作评分的非负信息价值。当前不把这些成熟机制改名冒称新算法。

run_gate.py 只执行固定 proposal 中的一个 fixture，四级 GL32/64/128/256、
Lv32/Lz64 和全部阈值不变。query Y=-1/+1 是人工代数 probe，不是新材料测量；
target CDF 点仅 -2/0/2。保留所有级别 full、within、signed between、closure、
解析理想算术 tail 与求积差值；求积收敛诊断不冒称严格 IEEE/CDF error certificate。
失败也完整保存，不扫描参数寻优。

执行入口由 root 登记实际UTC和精确source hash后运行一次：
D:/Anaconda/python.exe -X utf8 -B code/run_gate.py --plan EXECUTION_PLAN_REGISTERED.json --output execution_once
当前 TEMPLATE 不是执行许可。先独立静态审，再 root 执行 producer 与独立参考。
本门不读取任何材料性质、不选择新组、不拟合材料，不证明实际补全收益或顶会成熟度。
旧 152 组查询研究、MLST研究、V1失败/V2修复及完整本地资源包全部原样保留。
