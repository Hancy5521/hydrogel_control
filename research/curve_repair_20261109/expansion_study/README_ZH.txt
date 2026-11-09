固定边际＋冻结依赖：保留配方扩展检验（执行前说明）

本目录当前仅SOURCE准备。源码存在不等于已读取保留配方、完成预测或通过扩展验证；以真实UTC登记和各phase实际收据为准。

1. 保留什么，检验什么

开发阶段的结果支持一个有限问题：保留六锚点幂律Student已经较强的逐点预测，仅把完整联合依赖替换为原冻结模型提供的条件相关结构，是否改善曲线联合概率密度？这不以牺牲边际分数换效果，不要求每条曲线低秩，也不把一维分数的恒等保持说成新的实证提升。

候选在开发标签上只按既有事前DECISION_GATE选择一次，当前固定为冻结曲率posterior correlation。不在这三条新配方上重选GP/曲率、混合权重、相关长度、尺度、核、目标区域或指标。若失败，保留负结果，不把该扩展悄悄改叫新开发集来追参。

2. 真正的样本与角色

同一公开树脂/HGM原ZIP、同一50运行的native输入身份、同一mlst-resin-role-v1|配方哈希分配。原3dev/3future_expansion/4final_confirm角色完全不变。本轮只准3future_expansion配方全部5运行，共15曲线；不把它们伪装成dev来复用旧reader。4final_confirm配方全部20运行数值仍禁止访问。所有温度块与同配方各运行仍随原配方分组。

这是同来源未参与开发的配方组检验，不是新实验室、认证新批次或新湿实验。三配方仍有限，不把15运行、120目标点或840分数项当作独立材料数量。

3. 所有方法严格同信息

每曲线25个原生剪切坐标，不插值/对齐。六个锚点固定零基索引[0,1,13,14,15,16]，目标八点固定17..24。source_reader.py显式要求authorized_role='future_expansion'并重新核原角色；每条曲线六锚点只读一次，所有七方法复用相同六个数。获准访问的无效/非正黏度立即停止，不替换配方/运行/点位。原cP/1000转Pa.s再ln，35°C实际输入块与全部Dy_auto状态保持原准入规则。完整所选原CSV字节会进入可信reader内存，隔离是软件契约而非加密。

每曲线七个联合分布：
physical_full_student：六锚点OLS幂律、常规参考先验产生的完整8维Student4。
copula_independence：相同八个Student4边际＋独立Gaussian copula。
copula_physical_student_shape：相同边际＋原Student shape的相关矩阵。
copula_frozen_gp_posterior：相同边际＋原冻结GP条件相关矩阵。
copula_frozen_curvature_posterior：相同边际＋原冻结曲率Gaussian条件相关矩阵，事前选定主候选。
gaussian_frozen_gp与gaussian_frozen_curvature：原冻结均值和完整协方差的六锚点条件预测，作为保留参考控制。

总105个八维联合分布；15次物理OLS，不是105次拟合；无协方差拟合。唯一anchor读数90，目标读数120；840逐点分数没有制造新测量。

4. 数学与创新口径

令物理Student提供边际F_j=t4((y_j−mu_j)/s_j)，保持其所有单点概率、分位数和CRPS。令z_j=Phi^-1(F_j(y_j))，冻结完整相关矩阵R给出联合密度：
log p(y)=sum_j log f_j(y_j) −0.5 log|R| −0.5 z'(R^-1−I)z。

用原Gaussian在同六观测后的完整Schur协方差归一化得到R；没有仅条件化两维编码而将高维残差留在未条件状态。R只作对角归一化，不作PSD修补、shrinkage、jitter或特征值裁剪。非SPD即失败。固定Student边际的scale不是标准差，Student4方差为2*scale²。log-Student取exp后没有有限的黏度算术期望，本试验不报告该期望。

Gaussian copula/分位数传输是经典机制，不能宣称本概率恒等式首创。材料增量应由“强物理边际的完整保留、来自冻结曲线模型的额外依赖价值，以及同信息保留配方检验”支撑。当前只保证固定观测集合O及其目标子集一致，不能声称不同O重新构造的模型自动是同一个全局联合分布的Bayes条件。

充分性须区分对象：对于原相关GP或通用克里金预测，两个beta系数与SSE并不替代原观测位置/值、完整K和残差条件项。本固定O/T的copula构造为保持强边际，有意只通过物理beta/SSE表达观测值依赖，R由观测位置/目标位置与固定K决定；给定这些坐标及K，保存的beta/SSE可以决定本构造的边际与联合law，但这只是该模型假设，不是beta/SSE对真实材料过程充分的证明，也不是恢复了所有被压缩丢弃的残差信息。

5. 执行前与执行后

root registration schema=mlst-resin-expansion-registration-v1，字段actual_utc、protocol_id/protocol_sha256、authorized_phases、source_pins、本轮开发选择收据development_selection_receipt及SHA。source_pins须包含本目录两个py及experiment.DEPENDENCIES中七个冻结依赖。选择收据须status PASS、expansion_gate_passed true、selected_candidate=frozen_curvature_posterior，并建议保存原dev scores和独审SHA链。

prepare：只读坐标/原角色及冻结源码参数，保存prepare/PREPARED.json的完整m0/K0/B、两reference m/K/theta、全部15成员、固定mask、原配方角色。0新黏度。
predict：每成员一次读取六anchor，完成105全预测并保存predict/PREDICTIONS.json和ACCESS.json。保存每法自己的共同六值；目标120值此时未读取。独审只读保存的六值和预测，不打开原CSV目标。
reveal：独立预Y报告PASS须绑定audited_expansion_predictions_sha256和audited_expansion_prepare_sha256；root release schema=mlst-resin-expansion-target-release-v1、authorization=REVEAL_ONLY_120_ORIGINAL_FUTURE_EXPANSION_TARGETS，绑定predictions_sha256/prepare_sha256/audit_sha256及真实UTC。通过后仅读15成员的八目标，保存reveal/TARGETS.json及ACCESS。
score：只用已冻结预测和已释放120值，保存score/SCORES.json。另由独立源码从准入的原行地址和公式复核；评分审计不打开final20的响应。

每phase创建独占目录与STARTED/COMPLETE收据；失败留FAILED及已有文件，拒绝覆盖与自动重跑。命令：
python -X utf8 -B experiment.py --registration REG --output OUT prepare
python -X utf8 -B experiment.py --registration REG --output OUT predict
python -X utf8 -B experiment.py --registration REG --output OUT --audit AUDIT --release RELEASE reveal
python -X utf8 -B experiment.py --registration REG --output OUT score

--checkpoint-root与--pins-path可指向字节一致的冻结16依赖副本（prepare）；predict的--checkpoint-root只搬原数学core。原ZIP/input路径由已登记协议固定，不自动寻找别的数据。

6. 必须原样报告的终点

主终点：曲率copula减原完整物理Student的8维NLL/8，先五运行/配方，再三配方等权；负值改善。
副终点：曲率copula减GPcopula的同一联合NLL/8。
全部七法同时报告逐点CRPS、logRMSE、90%覆盖/宽度和联合NLL，不能选副指标救主终点。四copula与原物理Student的边际分数相同是结构恒等，不是新边际改善。score包含105 curve_scores、840 point_terms、21 recipe_scores、7 macro_by_method；primary和secondary均有recipe_macro_joint_nll_delta，primary另逐配方保留差值。

结果即使过门，也只支持上述同来源配方范围的联合概率增量。不能据此称自催化智能体已建立、跨材料全面泛化、实际实验节约或顶会证据完成。若结果失败，保留一次事前设计的失败与全部强控制，final20仍保护；不在同扩展标签上追加相关搜索。
