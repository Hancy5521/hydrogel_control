六锚点物理强对照：源码说明（执行前版）

这是补充对照，不是推翻原模型或更换原14方法主终点。本程序仅复用原predict冻结文件中各法自己的六个anchor_values，绝不调用ZIP reader或增加一个原始黏度读数。原目标八点仍须在420个对照预测冻结并独审之后，由root精确SHA释放才能评分。

物理模型只有两个：常数黏度假设在ln eta空间是截距模型；幂律eta=K*gamma^(n-1)在ln gamma/ln eta空间是截距+斜率模型。不预设其一定适配原材料，不凭结果在这两种模型中挑一个作为最终“新算法”。

统计假设：六个observed锚点满足y=X*beta+epsilon，epsilon独立同方差Gaussian；使用p(beta,sigma²)正比1/sigma²。这里明示“beta平坦先验＋通常参考尺度先验”，避免把它与所有参数联合Jeffreys定义混淆。

beta_hat=OLS；SSE=||y-X*beta_hat||²；ν=6-p；s²=SSE/ν；
新目标向量服从多元Student t：
location=X_T*beta_hat；scale_matrix=s²[I+X_T*(X_O'X_O)^(-1)*X_T']。
预测covariance=ν/(ν−2)*scale_matrix；单目标t_scale=sqrt(scale_matrix_tt)，这不是t标准差。常数p=1所以ν=5；幂律p=2所以ν=4。完整8维t共享回归系数和残差尺度，NLL不能用独立一维t的NLL简单相加。

SSE=0、exactconstant y、rank不足或数值非有限即停止并保留失败，不放方差floor、不删曲线、不替换mask、不改先验。数学上精确幂律也可能因浮点OLS得到正的roundoff SSE，因此执行前固定后向误差级拒绝门：r_resolution=64*machine_epsilon*(||X||₂*||beta_hat||₂+||y||₂)，SSE<=r_resolution²判定numerically_unresolved_residual。错误记录真实SSE、r_resolution和SSE阈值；成功时也保存阈值。该门只拒绝无法数值分辨的残差，不把SSE抬到阈值、不填补方差，也不把阈值当材料噪声。所有真实对照拟合只用本方法已有六值；420次OLS与“主模型零次重训”和“零额外原始η访问”分别报告，不能把OLS称为没有拟合。

纯数学在pure_student.py。CRPS实现采用scoringRules/Jordan等Appendix A中明确公式，用betaln稳定计算常数；实际应用前由另一源码直接积分CDF平方独立核对。此来源已定向阅读Student公式，不声称读完整包或全部论文：
https://scoringrules.readthedocs.io/en/latest/forecast_dists.html
多元t shape说明与density公式另核官方SciPy文档：
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html
该网页当前为1.18.0，实际本地SciPy1.15.3；公式手工实现，没有安装或调用1.18专属API。上述为已有公式使用，不是新概率理论或复制来源代码。

协议在ANCHOR_BASELINE_PROTOCOL.json。root实际UTC登记后，anchor_baseline_experiment.py只有predict和score两个命令：
python -X utf8 -B anchor_baseline_experiment.py --registration REG --output OUT predict
python -X utf8 -B anchor_baseline_experiment.py --registration REG --output OUT --audit AUDIT --release RELEASE --targets MAIN_TARGETS score

root registration schema=mlst-resin-anchor-baseline-registration-v1，需要actual_utc、protocol_id/protocol_sha256、authorized_phases、source_pins{绝对source_path:sha}、main_predictions/main_predictions_sha256、main_prepare/main_prepare_sha256。源码/原main文件任一改变均拒绝。每phase新建目录，不覆盖失败或已完成记录。

predict/PREDICTIONS.json顶层forecasts共420，每项保存member/recipe/run_id/thermal_history，mask_method_id对应原main方法，baseline_kind与method_id，以及本法六anchor索引/值、固定target_indices、beta/SSE/n/p/df/location/scale_matrix/predictive_covariance。源mask以原main210条为准，重复mask不会被当成独立新观测。

独审预Y至少须验证原own-six与row对应、OLS参数、df/shape/covariance、原配方角色、420完整预测。独审报告要求status=PASS、audited_baseline_predictions_sha256及audited_main_predictions_sha256。

root score release schema=mlst-resin-anchor-baseline-score-release-v1，authorization=SCORE_FROZEN_BASELINES_ON_SAME_120_DEV_TARGETS，精确绑定baseline_predictions_sha256、main_predictions_sha256、audit_sha256、targets_sha256和actual_utc。target文件必须与原main预测/prepare SHA一致。

score/SCORES.json包含curve_scores420、point_terms3360、recipe_scores84、macro_by_method28；metrics为CRPS、joint_nll_per_dimension、log_rmse、coverage90、width90。顺序仍八点/run、五run/recipe、三recipe等权；log_rmse是run级八点RMSE的平均。每point同时保存df/t_scale/t_sd以防尺度混淆。需要与原Gaussian在相同mask、相同目标、相同六锚点上比较。

所有分数在ln[eta/(1Pa.s)]空间。把Student t用于log eta不赋予有限的eta算术均值，因此本程序不输出Pa.s空间均值或把它误称lognormal；如需物理量区间可对有限分位数取exp，但本试验不新增该任务。

本补充是预目标的公平性改进，不改变原主对比，不增加新的独立材料数量，不自动等于水凝胶新批次确认。三recipe仍是开发，未来35运行保持未读。所有28个对照组合均报告，不事后重新选模型、mask或主指标来定义成功。
