强物理边际保持不变的完整依赖搬运
2026-11-09 SOURCE准备；尚未执行/验证，不预写联合收益。

一 为什么只做这个有界问题

已完成的UK开发诊断没有追平nearest6幂律Student的CRPS，故不开放expansion/final。它的jointNLL仍比纯幂律好，说明“逐点分布好不好”与“整段曲线的依赖结构好不好”值得解耦。新诊断固定目前强的六anchor幂律Student4边际，只改变联合copula。它不重调旧K/θ、尺度或趋势，不做mixture/gate/超参搜索。

十五dev runs×四个预定相关控制=60个完整8维law，固定nearest6读数，原八个tail目标不变。来源仍是已消费三recipe，绝不是新的材料确认。未来三expansion加四final共35runs的黏度不在本程序中读取。既有完整Student联合分布仍作为强对照：由它的shape相关构造Gaussian copula，不等于保留其原t-copula，不能将这两种law混写。

二 分布与代码

pure.py仅负责有限维分布：m_j、s_j取原强Student的location和sqrt(shape_jj)，df固定4。R分别为I、物理Student shape的相关矩阵、冻结GP nearest posterior相关矩阵、冻结curvature nearest posterior相关矩阵。

生成时ξ~N(0,I), z=chol(R)ξ, y_j=m_j+s_j t4^{-1}(Φ(z_j))。整条向量一起生成，而非把每个坐标独立抽样再叫联合恢复。密度为各Student边际密度乘Gaussian copula密度：
log q(y)=sum_j[log t4((y_j−m_j)/s_j)−log s_j]−.5logdetR−.5(z'R^{-1}z−z'z)。

实际实现以Cholesky solve计算二次型，不显式求逆。y→z按标准化y符号选择t.logcdf或t.logsf，再用ndtri_exp，避免CDF先变成0/1。z→y取Φ(−|z|)的较小尾概率，再配符号和Student逆CDF，避免1−p抵消。有限数值范围之外非finite就停止，不clip概率、不加jitter、不特征值裁剪或variancefloor。

correlation_from_covariance仅作定义性对称/对角标准化，使diag(R)=1；不改变任何特征值来修补非SPD。pure支持完整联合logpdf、相关采样和固定O下的target子集边际。不同O下重新构造的copula不自动是一套全局联合分布的Bayes条件；本诊断不作这种兼容性声明。

四种law的每个边际分布都精确等于原强baseline，因此CRPS、边际coverage、分位数不变是结构恒等式，不是新增实测效果。score阶段逐项校验目标/loc/scale/df并精确沿用原冻结边际score；新比较是60个完整law的jointNLL/维及相对原完整Student的差值。所有四个结果一起报告，不选最好后追加混合权重。

响应是ln(η/(1Pa.s))。对log-Student求exp后的η没有有限算术期望，所以程序只给logspace密度、黏度中位数/分位数或样本；不报告不存在的Pa.s arithmetic mean。这里的Gaussian latent相关也不等于变换后y的Pearson相关，不能直接等同解释。

三 实际执行顺序（root的真实UTC和hash注册后）

experiment.py是唯一runner，phase为gate/predict/score；源码本身不构成权限。

registration schema mlst-marginal-transport-registration-v1：actual_utc、authorized_phases、source_pins绝对路径→SHA（pure.py、experiment.py、本protocol）、input_pins已有素材路径→SHA，以及prepare/main_predictions/student_predictions三条路径。保留旧源原字节。

gate：固定3维fixture，检查±9 latent tail、标准化读数±1e12、SPD拒绝，保存20000个固定PCG64合成draw及预定0.1/0.5/0.9分位CDF计数。它不是材料标签。独审读取该NPZ核计数/尾部/density，不能用“重跑同函数”替代独审。

predict：独审gate PASS且绑定gate SHA后，只读已保存的main nearest和原Student nearest六anchor预测，冻结60个完整law。既无rawCSV reader，也无fit函数。物理母方法固定power_law__curv_nearest_target6，原GP/curv nearest masks必须完全同六点。

score：独审60law PASS、绑定预测SHA后，root另写authorization=SCORE_FROZEN_MARGINAL_TRANSPORT_ON_SAME120的release，精确绑定predictions_sha256/targets_sha256/student_scores_sha256/audit_sha256和实际UTC；才读原TARGETS与StudentSCORES。评分是同三recipe开发诊断，不成为fresh confirm。

若任何依赖或联合效益不成立，保留失败，不在本三recipe继续搜索copula参数/相关模板/目标位置。当前仍不开放未来35runs。该机制是否值得进一步工作，由四个固定对照及原t联合baseline的共同结果判断，不由恒等保留的CRPS证明。

四 最多三项直接原始先例与真实阅读范围

1 Schefzik, Thorarinsdottir, Gneiting (2013), Uncertainty Quantification in Complex Simulation Models Using Ensemble Copula Coupling. Statistical Science 28(4),616–640, DOI10.1214/13-STS443。原作者电子重印版arXiv1302.7149v2(2013-12-23): https://arxiv.org/pdf/1302.7149v2 。本轮实际读摘要、§3 copula/Sklar讨论与§4 ECC方法/与Schaake的关系，没有复现全部天气实验。先校正边际、再搬运依赖秩结构是直接先例。我们实现的是参数Gaussian copula密度，不声称就是ECC的有限ensemble经验copula，也不将借鉴改名成首创。

2 Clark, Gangopadhyay, Hay, Rajagopalan, Wilby (2004), The Schaake Shuffle: A Method for Reconstructing Space–Time Variability in Forecasted Precipitation and Temperature Fields. Journal of Hydrometeorology 5,243–262，DOI10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2。
https://journals.ametsoc.org/view/journals/hydr/5/1/1525-7541_2004_005_0243_tssamf_2_0_co_2.xml
搜索工具取得原期刊摘要及“ensemble reordering method”原文方法提取；随后直接open返回403，保留失败，没有冒称整篇全文读完。它通过历史观测rank重排边际ensemble来恢复空间/时间依赖，是相关思想直接来源。本诊断不复制其历史日期抽样算法。

3 Zhao & Udell (2020), Missing Value Imputation for Mixed Data via Gaussian Copula. KDD2020，DOI10.1145/3394486.3403106。
https://kdd.org/kdd2020/accepted-papers/view/missing-value-imputation-for-mixed-data-via-gaussian-copula.html
作者PDF https://web.stanford.edu/~udell/doc/zhao2019missing.pdf (标注arXiv1910.12845v2,2020-03-02)。本轮读正式摘要、作者PDF的Definition1和连续变量单调变换段，未复现其EM/混合类型实验。这是Gaussian copula用于缺失补全的直接先例；我们没有以新名绕开它，而是固定强物理边际、只检查已有fullrank相关在当前材料尾段任务上是否有用。

检索式包括Gaussian copula polymer rheology viscosity missing data imputation；该有界检索没有返回清楚的同一水凝胶/树脂六anchor任务近邻，但不能证明全领域无雷同。随后定向检索ECC/Schaake及上述KDD原始出处，总计最多三篇内容核查。没有引入任何第三方论文/仓库代码，没有核得这些论文再分发许可，不把公开可读误作允许打包整篇；本包只保留链接、阅读范围和独立实现。

创新定位：经典边际/依赖解耦的材料适配与可证伪证据。若有价值，贡献是让强物理边际不被高维相关修正破坏，同时仍给出可评分的fullrank联合曲线；不是发明Gaussian copula理论。阶段性成功不自动等于顶会录用，也不要求为引用经典机制再发明一个复杂网络。
