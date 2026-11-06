V2：仅修复Studentt query积分坐标，模型与科学阈值均不改变

V1原code、SOURCE_FREEZE、12例结果和独立参考收据保留在上一级目录。V1的代数与截断tail检查通过，但Studentt末级求积差没有达到原2e-5；独立参考还出现native-query坐标的IntegrationWarning。不能把这些记录抹掉，也不能为通过而放宽tol或换context/相关参数。
本V2不改NIG、12个risk评价、Lv32/Lz64、四级32/64/128/256、目标原单位YJJac、Gaussian同目标强参照或排序问题。

唯一数学替换
旧inner query积分用t-CDF quantile区间的GL，端点PPF导数很大。新令z=sqrt(nu)*tan(theta)，theta在atan(-64/sqrt(nu))与atan(64/sqrt(nu))之间；cos(theta)>0。
  f_t(z) dz = cos(theta)^(nu-1)/B(1/2,nu/2) dtheta。
条件CDF参数等价简化为
  sqrt[(nu+1)/(nu*(1-rho²))] * [v*cos(theta)-rho*sqrt(nu)*sin(theta)]。
保持原t截断概率质量，不归一化成1，也不把theta扩到+-pi/2而偷偷改变Lz。原query尾界仍保留。diagnostics另列实际GL质量与CDFt(64)-CDFt(-64)的差，不能把质量差补归一化后掩盖误差。
V1两层nodes一起变化，旧结果不能严格因果识别slow点来自哪一层；V2只改变inner坐标，是数值一致性检验，不称新模型、material效益或policy创新。

版本边界
code/posterior_update.py逐字等于V1。activefield.py通过AST比对只有student_kernel函数改变；Gaussian/Owen+positive-angle、tail、affine闭式、rho coupling等函数AST原样。driver只更换config版本/坐标描述并要求绑定V1执行、V1独立核查、坐标变更说明三个收据。V1核查若失败仍如实绑定其原件，不伪称成功。

执行前
EXECUTION_PLAN_TEMPLATE.json尚是准备模板。root必须填写REGISTERED和真实UTC，并检查repair_inputs的全部原SHA后另行批准新plan。approval scope仍为NATIVE_RISKFIELD_FIXED_MECHANISM_GATE，allowed_phases=["math_gate"]。独立static review之后才可一次执行：
 python -X utf8 -B numerical_v2/code/run_gate.py --plan <V2registered.json> --approval <V2rootapproval.json> --output-dir <新的V2execution目录>
源码准备、AST核验和打开说明都不会运行积分；材料文件从未在本门读取。

之后
全部12例和4级结果继续保留；convergence仍未通过也不得扫prior/context寻想要的排序。独立querystandardZ×targetnative坐标参考要与此不同路线互核，不能再次调用producer θ核当独立证明。数学可靠后先按冻结真实context负载profile优化fullnative/vectorized/cache，实际有瓶颈才考虑压缩，不能把Gaussian近似或未优化循环当强reference。无论本门排序是否改变，它都不是已经完成真实主动材料查询的证据，后续新组的真实Y active loop仍应单独冻结执行。
