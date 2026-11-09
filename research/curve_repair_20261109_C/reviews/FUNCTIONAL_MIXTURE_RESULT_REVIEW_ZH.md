# 教师分歧概率修复：2×2归因实验独立结论

2026-11-09。对象为已消费的18个 DaRUS functional pilot context；本轮为登记后的开发诊断，不是新增材料、独立确认或湿实验。原先预选的 half-Gaussian pilot 结论不改写。

## 独立核验

全程不 import 生产模块，不新增 fit，不读取原始材料 G 或其他161个条件，不接触 resin final20。

| 阶段 | 独立结果 | 收据 SHA256 |
|---|---|---|
| synthetic gate | PASS，2230 checks；3固定fixture、1344项CRPS及192个latent采样重建 | `f11470aaaabd645db69d7f14dab69b2732c906226f1cc13c436260947cc3d3e4` |
| 冻结 law | PASS，598 checks；18context×4law | `7bca6448b4a4f229bc444d9e7125859db8287ad4d436914d4333c631d8f9ec3e` |
| 保存结果评分 | PASS，2040 checks；72row、1008 CRPS、12macro、36配对、72 RMSE恒等、36density-mixture界 | `42d94ea8088aa44f86e4738ef22357e533d28293c9b60e666575b65881d69fa9` |

收据位于 `functional_mixture/INDEPENDENT_GATE.json` 与 `functional_mixture/development_once/INDEPENDENT_{LAWS,SCORES}.json`。独立源是 `verification/functional_mixture_independent_audit.py`，SHA `a5ff731a45f68312add8ff37450ec1f9bb95b207e5ba5df43eee75055a990d06`；数学源 SHA `3d03817adcc5395b0f703a605b5f0eb2a4e1392d4b0942d3f551f29a078254d2`。

生产 Student-mixture CRPS 使用两独立 Student 的 cross absolute-energy；独审用响应 CDF 的两侧平方积分，且围绕每个分量中心分段。没有共享生产积分函数。主材料评分最大逐点 CRPS 差1.69×10⁻⁹，预先固定容差为2×10⁻⁹绝对加2×10⁻⁸相对；最大NLL绝对差4.90×10⁻⁷发生于大NLL外推的病态同矩Gaussian，满足原相对容差。分位区间宽度差4.52×10⁻¹⁴，RMSE恒等差0。没有放宽门槛。

## 2×2 family 解释清楚了什么

令 `d=m_Maxwell-m_GP`。四个派生法均固定

\[
E[Y]=(m_{GP}+m_{Maxwell})/2,\qquad
\operatorname{Cov}(Y)=C_{GP}+dd^T/4.
\]

比较因子为 component tail（Gaussian / ν4Student）和 representation（整条曲线等权density-mixture / 同一二矩单分布）。Student分量scale为 `C_GP/2`，同矩Student的scale为总covariance的一半。不能把scale直接叫covariance。

这保留完整高维 residual C，仅给教师分歧增加方向性变异。新增总covariance一般有跨通道block；这来自共享的全曲线分支，不能写成学到了跨通道残差covariance。每条14维曲线只有一个分支；Student residual也只有一个14维共享radial变量，绝非逐列或逐通道独立拼接。

四法点预测与原half方法完全相同，因此RMSE恒等是结构属性，不是4份独立性能收益。density-mixture对同tail GP端点的每context NLL/维恶化上界是 `ln2/14`，该界不适用于同矩单分布；36项保存评分均满足。

## 内部缺口出现联合分布收益，但形状选择有取舍

预先保留的 `interior_holes` 六状态等权结果：

| 方法 | logCRPS | jointNLL/维 | logRMSE | 覆盖90% | log宽度90% |
|---|---:|---:|---:|---:|---:|
| 原完整GP Gaussian | 0.01681407 | -2.749985 | 0.03318027 | 1.000000 | 0.140386 |
| 原GP均值、同cov Student4 | 0.01632167 | -3.041432 | 0.03318027 | 1.000000 | 0.128658 |
| 原固定half-Gaussian | 0.01436128 | -2.574395 | 0.02360673 | 0.916667 | 0.140386 |
| 原固定half-Student4 | 0.01321836 | -2.850681 | 0.02360673 | 0.916667 | 0.128658 |
| 新全曲线Gaussian-mixture | 0.01444116 | -2.755326 | 0.02360673 | 1.000000 | 0.152392 |
| 新全曲线Student4-mixture | 0.01327787 | **-3.161595** | 0.02360673 | 0.988095 | 0.140563 |
| 新同矩Gaussian | 0.01447571 | -2.728797 | 0.02360673 | 1.000000 | 0.153007 |
| 新同矩Student4 | **0.01317962** | -3.135207 | 0.02360673 | 1.000000 | 0.140225 |

Student-mixture相对原α0Student强控，CRPS减少0.00304380，NLL减少0.120163/维，同时保留half的点误差。它解决了上一轮“点预测有用、联合NLL反而变差”的具体不足，足以保留为候选。

但同矩Student的CRPS更低，Student-mixture的联合NLL更低，差别分别约0.00009825与0.026388/维。成熟同矩控制已解释很大部分收益；不能把全部进步归于多峰混合，更不能声称只有mixture才能奏效。Gaussian和Student的tail差异带来的NLL收益，也不能与representation差异混在一起。

相对原half-Student，新同矩Student几乎不改变CRPS，却把NLL从-2.85068改善到-3.13521，并恢复本panel的覆盖。这与最初诊断一致：缺口主要是教师修复方向的不确定性没有被纳入概率法，而非必须再加一个点预测网络。

这些仍是同18个context上的开发比较。不能把本轮开发选择倒填为原pilot的预定主选，亦不能把6状态84个target-role标量称作84个独立材料。

## 外推尚未闭环

| 几何 | 方法 | logCRPS | jointNLL/维 | 覆盖90% |
|---|---|---:|---:|---:|
| 低频O→高频T | 原GP Gaussian | 0.200126 | 86.3758 | 0.047619 |
| 低频O→高频T | 新Student-mixture | 0.232977 | -0.041589 | 0.535714 |
| 低频O→高频T | 新同矩Student | 0.231940 | -0.500891 | 0.666667 |
| 高频O→低频T | 原GP Gaussian | 0.231125 | 858.419 | 0.023810 |
| 高频O→低频T | 新Student-mixture | 0.176609 | 0.574666 | 0.595238 |
| 高频O→低频T | 新同矩Student | 0.142065 | 0.407999 | 0.654762 |

宽度和覆盖改善明显，却仍不能称为可靠外推。低频O情景的点误差已经因错误教师方向恶化，四个新法均保留该half均值，所以不会凭不确定性建模把点误差修好。高频O情景的coverage仍远低于90%。

Gaussian-mixture在高频O情景的NLL约858.469，几乎等于GP加 `ln2/14`。这说明其确实受到端点下界保护，却不能用两个狭窄分量自动恢复缺失区域的真实联合形状。更宽中心区间可以提高覆盖，区间中间的density仍可能很低；只报coverage会掩盖这一点。

`ddᵀ/4`只有一个分歧方向。真实外推误差可以有其他形状方向，因此一条正确的概率恒等式不等于完整模型误差已识别。该失败不意味着物理教师在内部缺口没有价值，也不能扩大否定已有水凝胶结果。

## 下一阶段应检验的唯一问题

可继续一个有界的数字闭环问题：**原六个观测中，是否已有足够证据提前识别“教师修复有用还是有害”或至少识别明显失配？** 它比再次搜索α、ν、τ、分量数更直接对应外推失败。

开发阶段可以在已消费context上估计一个明确受限的oracle容量或同预算伪遮挡诊断，但oracle只能给出上限，不能作为可部署性能；伪遮挡真值只能来自原有实测O，不得把self-imputed伪值作为新测量。预先固定验证预算、分组、单一门或单一权重及停/转规则，再登记执行。若实际O不能辨别失配，应如实承认当前信息不足，将方法适用范围先收敛至内部缺口，并把未释放条件留给真正新确认。

现有结果足以支持“物理教师分歧应保留为概率结构”的领域适配主线，尚不支持“新自催化理论”“自主实验节约已测量”“跨材料普适生成智能体已成立”。经典mixture、Student和矩匹配机制应正常引用；创新要由具体材料问题、统一修复机制及后续可证伪证据共同支撑。

