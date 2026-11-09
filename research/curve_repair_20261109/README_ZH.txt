曲线概率补齐：完整研究资源包A（2026-11-09）

这份包包含本轮已经完成的程序、原始公开来源、冻结权重/16项运行依赖、实际UTC协议、所有主/对照预测与评分、独立审计、失败原件、数学公式与程序说明。它是研发检查点，不是顶会达标或录用声明。旧MLST水凝胶有效版本和稿件仍单独封存，未被覆盖。

一、先看什么
READ_FIRST.html：总览和指标；EVIDENCE_SUMMARY.json：机器摘要。
scientific_review/RESULT_INTERPRETATION_20261109.md：主试验、强物理基线与早期预警的独立解释。
scientific_review/TRANSFER_REFERENCE_FAILURE_DIAGNOSIS_ZH.md：为什么趋势/尺度适配没有通过门槛，包含正负指标与数据来源限定。
material_stress/SOURCE_PHASES_ZH.txt、baselines/README_ZH.txt、transfer_reference/README_ZH.txt（若文件名略有不同请见目录）：模块、公式、命令。
core/API.md：任意缺口、完整残差、已知额外噪声、角色隔离及可用接口。
delivery/BUILD_NOTES_ZH.txt：移位核查说明。

二、真实结果
数据为未固化光敏树脂/HGM悬浮液，不是新水凝胶。只取预分组3配方×5原run的名义35°C块，每curve25原生剪切坐标。不插值，不因响应删除样本。其余3扩展配方/4最终配方共35run的黏度没有数值释放；原完整公开ZIP存在包中，不是加密隐藏。
主终点curvature expectedCRPS由first6的0.692146下降到0.065148，三配方均改善。但nearest简单布局已达0.062968，强幂律Student在同六锚点达0.044490，不能称新采点策略超越强基线。
原完整Gaussian联合NLL仍优于幂律，相关结构值得保留。一次跨配方早期梯度更新虽使已知sentinel NLL改善，却使其未读tail CRPS略差；负结果保存。另一经典趋势/尺度+冻结K Student诊断CRPS0.097995更差，未通过扩大资料门槛，没有消费35未来run或调参补救。所有数字均log黏度坐标，不能和旧水凝胶数据的分数直接相减。

三、算法是什么
完整Gaussian：mθ=m0+B a，Kθ=K0+B(S−I)Bᵀ=(K0−BBᵀ)+BSBᵀ。低维修正不替代高维残差。
任意可见O后采用完整Schur条件分布；不是只生成二维code再加一份未条件化残差。固定Gaussian的查询收益依赖协方差和位置，不依赖已观测值，不能冒称值驱动自催化。
强基线：lnη=β0+β1 lnγ+ε，六已知点拟合，β/σ²的reference prior积分给完整ν=4 Student预测；scale与variance严格区分。
趋势适配诊断：GLS物理均值+完整固定相关K+未知尺度，保留条件残差C与趋势不确定U。这是经典通用克里金，不是新的概率理论；本数据上失败也保留。
β和SSE并非完整相关GP预测的充分编码，还必须保留位置、残差响应K_TO K_OO^-1(y_O−X_Oβ)及原K。不能把二维表示当成整条高维曲线无损压缩。

四、程序分层
core只负责通用数学/角色，material_stress负责真实来源和一次实验阶段；baselines负责物理对照；influence_pilot单次参数更新；transfer_reference负责单一适配假说。verification是独立实现，scientific_review是解释。没有把实验协议混进旧部署源，没有引入扩散/RL大网络。
学到生成规则或绘制很多抽样不是增加测量。主210+420+45预测在120真目标释放前冻结；后420个适配预测是同一已消费dev的诊断，不是新确认。多方法共用120目标，不是数千独立数据。

五、如何复核
已验证环境Python3.13.5、numpy2.1.3、scipy1.15.3、mpmath1.3.0。安装依赖后在解压目录运行：
python -X utf8 -B delivery/reproduce_offline.py
该命令只独立审计保存的预测/分数，写新的offline_audits/UTC目录，不训练、不重跑生产实验，不访问35run未用响应。先校验每个成员SHA及原16pin。完整源和权重已放frozen_checkpoint/，不要求原H盘旧模型目录存在。原阶段生产CLI内保留当时绝对路径用于审计，若重建新试验必须另作路径映射和新协议，不能拿旧已消费集重新称确认。
H3第一次double审计FAIL原件保留；另登记70位精度审计通过，原容差未变，模型/step未改。它是审计数值稳定性修复，不是新材料效果。

六、包含与未包含
包含本阶段全部完成分支、原公开数据/元数据、源码/协议/数值证据/失败、16份必要冻结运行源码与参数。不嵌套这些天所有大历史ZIP或旧训练表；它们不参与本包移位数值核查。旧MLST完整手稿/水凝胶原包继续位于H:/PolymerReview/MLST_controls_20261104T154143Z_r2，本包不替代它。没有包含Obsidian私有笔记、下载目录审稿附件、账户认证或环境密码。
当前手动轮次仍在继续，run_logs是A封存时的真实日志快照；本轮最终结束收据与A封包后的Git/离线重放收据均另存，不能预填成已在A包内。
