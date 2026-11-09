物理趋势＋冻结完整相关结构：执行前开发诊断说明

这一步为什么必要

既有开发试验提示：改变锚点布局能显著帮助原冻结模型，但在相同六观测下，简单物理幂律Student对照仍可更好。一个可以证伪的解释是，跨材料时失配集中在原水凝胶的均值趋势与绝对尺度，而完整相关结构可能仍有用。本次只检验这个解释，不在同一开发集上搜索核参数、修正强度、先验或其它新旋钮。

这不是新的独立材料确认。目标八点已在前轮开发中消费，因此所有本次预测/评分均标POST_REVEAL_DEV_DIAGNOSTIC。先冻结新预测再由root释放同一目标，只提供可审查的开发执行次序，不能恢复被消费样本的独立性。其它35条future/final运行仍不读。

数学定义

给定一条曲线固定25坐标x=ln(gamma/(1s^-1))，已有六锚点O和固定八目标T。原main保存的完整25×25K来自零修正GP或原冻结曲率Gaussian。本次弃用原冻结mean，以X=[1,x]的每曲线物理趋势替代；保留K的全部高维结构与统计残差，不把K换成两维低秩矩阵。

y | beta,sigma²,K ~ Normal(X beta, sigma² K)
p(beta,sigma²) 正比 1/sigma²。

取Koo=L L'，Xw=L^-1 Xo，yw=L^-1 yo，在白化数据上用OLS得到betaGLS。r=yo−Xo beta；SSE=r'Koo^-1r；nu=6−2=4。定义V=L^-1Kot；H=Xt−V'Xw：

location = Xt beta + V'(L^-1r)
C = Ktt−V'V
U = H (Xw'Xw)^-1 H'
scale_matrix = (SSE/4)(C+U)
predictive_covariance = 4/(4−2) * scale_matrix。

重要的信息保留边界：两个beta系数及SSE本身不是对任意未来目标都充分的低维编码。预测仍依赖K_TO*K_OO^-1*(y_O−X_O*beta)这一观测残差信息，也依赖原观测位置/值和完整固定K。因此不能声称仅用二维beta完整编码25维曲线；保存完整posterior law及残差条件项，正是本实现没有丢弃这些信息的原因。

这是经典通用克里金/相关线性模型的Student预测，不把这一恒等式当新算法。C保留完整条件残差；U保留估计趋势的不确定性。计算只用Cholesky/solve/lstsq，不显式求逆，不添加jitter、方差floor或失败回退。

K=I时，V=0、C=I，精确退回此前强幂律Student。这是必须通过的退化一致性门。把K整体乘任意正常数时，SSE反向缩放，最终完整Student预测不变；这也是数值门。因而本次学习的每曲线尺度不应被解释为再次拟合相关核的整体振幅。

异常门与前一Student对照一致：rank不足/非SPD/非finite直接失败；数值残差分辨率固定为64*eps*(||Xw||₂||beta||₂+||yw||₂)，SSE不超过其平方时拒绝，不把roundoff变成极窄预测分布，不把阈值作为材料噪声。

源码边界

universal_student.py：纯GLS及完整Student预测。无文件/真实材料reader，无核调参器，无target标签参数。
transfer_experiment.py：只读main已保存PREPARED中的坐标/K，以及PREDICTIONS中每法自己的六锚点；每个main14mask分别用GP/curv两个固定K，总420个预测。重复mask保留以方便严格匹配，但不算新测量或独立模型选择。score复用已经冻结、已经独审的baselines/pure_student.py精确源码，SHA强校验，不再重写评分公式。
equivalence_gate.py：仅固定合成门，核K=I、整体K缩放不变、dense公式、观测/目标顺序及exactline拒绝。该门不是材料效果，不触碰原CSV或future标签。
TRANSFER_REFERENCE_PROTOCOL.json：实际root登记前的方案，无执行授权。

执行与文件

root登记schema mlst-resin-universal-student-registration-v1：actual_utc、protocol_id/hash、authorized_phases=['gate','predict','score']、source_pins（本目录全部3个py和旧pure_student.py）、main_predictions及SHA、main_prepare及SHA。

python -X utf8 -B equivalence_gate.py --registration REG --output GATE_JSON
python -X utf8 -B transfer_experiment.py --registration REG --output OUT predict
python -X utf8 -B transfer_experiment.py --registration REG --output OUT --audit AUDIT --release RELEASE --targets SAME_TARGETS score

predict/PREDICTIONS.json中forecast420每项保存：covariance_id、mask_method_id、member/recipe/run/history、六anchor索引与值、共同target索引、beta/SSE/df4、分辨率阈值、location、scale_matrix、predictive_covariance、未缩放的C/U分量。420次GLS与0相关核更新、0原始黏度新读取分别报告。

独立law审计报告需status PASS、audited_transfer_predictions_sha256及audited_main_predictions_sha256。root score release schema=mlst-resin-universal-student-score-release-v1，authorization=SCORE_DEV_DIAGNOSTIC_ON_SAME_CONSUMED_120_TARGETS，绑定transfer_predictions_sha256/main_predictions_sha256/audit_sha256/targets_sha256及实际UTC。

score/SCORES.json为420 curve_scores、3360 point_terms、84 recipe_scores、28 macro_by_method。聚合保持八点/run→五run/recipe→三recipe等权。对每个mask，必须与原Gaussian及原幂律Student在相同六值/目标比较，不从28组挑最佳并改写原主终点。

若失败意味着什么

如果这一步仍不能达到幂律强对照，不能据同dev目标再搜索相关长度或把某个结果最好布局改成主终点。应保留负结果，区分“冻结相关结构本身不适配”与“原mean/scale不适配”的可证伪解释。若有明确、固定新假说，另在受保护的future_expansion角色按新协议检验；最终确认角色保持到真正冻结后。程序可审查、完整数值门通过均不等于顶会论文证据已闭环。
