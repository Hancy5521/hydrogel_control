"""Write the C checkpoint manual from saved, independently audited scores.

No training, new labels or scientific model selection occurs here. The target
must remain unsealed; historical source/registration bytes are never rewritten.
"""
from pathlib import Path
import datetime as dt
import html
import json

OUT=Path('H:/PolymerReview/Curve_Repair_20261109_C')


def main():
    state=json.loads((OUT/'BUILD_STATE.json').read_text('utf-8'))
    if state['status']!='BUILDING_NOT_SEALED_NOT_DELIVERED':
        raise ValueError('A sealed checkpoint cannot be rewritten')
    stage=OUT/'stage'
    a=json.loads((stage/'functional_repair/pilot_once/SCORES.json').read_text('utf-8'))
    b=json.loads((stage/'functional_mixture/development_once/SCORES.json').read_text('utf-8'))
    index={(r['mask'],r['method']):r for r in a['macro_by_mask']+b['macro_by_mask']}
    key='interior_holes'
    gp=index[key,'full_anchor_gaussian'];st=index[key,'full_anchor_student4']
    mix=index[key,'whole_curve_student4_density_mixture'];moment=index[key,'moment_matched_student4']
    physics=index[key,'channel_relative_Maxwell_NNLS_point']
    names=[('完整Gaussian GP','full_anchor_gaussian'),('完整同矩Student4 GP','full_anchor_student4'),
           ('原固定半程Gaussian','half_maxwell_mean_gaussian'),('整曲线Student4混合','whole_curve_student4_density_mixture'),
           ('同矩单Student4控制','moment_matched_student4'),('相对误差Maxwell点基线','channel_relative_Maxwell_NNLS_point')]
    table=['方法 | log CRPS↓ | log RMSE↓ | 联合NLL/维↓ | 90%区间覆盖']
    for label,method in names:
        r=index[key,method]
        def fmt(name):return '未定义（纯点模型）' if r.get(name) is None else f'{r[name]:.8f}'
        table.append(' | '.join([label,fmt('log_crps'),fmt('log_rmse'),fmt('joint_nll_per_dim'),fmt('coverage90')]))
    record_time=dt.datetime.now(dt.timezone.utc).isoformat()
    sections=[]
    def add(title,body):sections.append((title,body.strip()))
    add('一、这次保存了什么，以及当前结论',f'''
本包保存截至 {record_time} 的C研究检查点：当前可运行代码、真实公开数据、完整原协议与模型参数、全部预测和评分、失败记录、独立数值审计、合成数值门、生成示例及详细说明。它还完整保存原MLST发布目录的1711个payload（约68.5MB），放在legacy_mlst，避免仅留下一个不能恢复的旧路径。

当前最有价值的结果是：对一种商业聚氨酯的成对动态模量，在内部缺口任务中，从六个频率处的十二个标量观测恢复另外七个频率的十四个标量，物理教师与完整参考分布结合取得实测预测收益。整曲线Student4混合在已经消费的开发集上，log CRPS相对完整Gaussian GP下降{100*(gp['log_crps']-mix['log_crps'])/gp['log_crps']:.2f}%，log RMSE下降{100*(gp['log_rmse']-mix['log_rmse'])/gp['log_rmse']:.2f}%；相对更强的同矩Student4 GP，CRPS下降{100*(st['log_crps']-mix['log_crps'])/st['log_crps']:.2f}%，联合NLL每维改善{st['joint_nll_per_dim']-mix['joint_nll_per_dim']:.6f}。

这不是一次新的独立确认：原半程方法先冻结再揭示目标，而四种后续分布是看过原pilot结论后提出的开发诊断。它们不能继承原pilot的事前身份。当前六个湿度/历史状态仍属一种材料，不是六个独立配方；外推仍有严重不足。同矩单Student4在CRPS上还略优于混合，因此保留它作为强控制，不能说混合形状每项都必要。

这是普通研究检查点。新顶会论文、完整Eigen超循环和任意高维性质矩阵实证尚未完成；本包不命名为投稿达标或只待补实验候选。原MLST论文和研究结果按原字节独立保存，没有将其结论混入本轮新任务。
''')
    add('二、任务与数据：不是把所有性质随意排列成时间', '''
原始研究目标是材料稀疏性能数据的概率补全。本阶段先研究其中有明确坐标和物理关系的子问题：稀疏观测的聚合物函数，例如黏度随剪切率、储能模量和损耗模量随角频率。

动态模量来源为DaRUS DOI 10.18419/DARUS-2021，版本1.1，CC BY 4.0。源文件包含一种商业聚氨酯的dry、redry、30/50/70%湿度、wet历史块。每条条件曲线13个原生频率；原响应单位MPa，程序精确乘10^6转为Pa。输入频率Hz换为ω=2πf rad/s，不把ω当剪切率γ̇。建模响应为ln[G/(1Pa)]。

本次输入盲选每状态实际温度中位数最接近30°C的一条曲线，共六条、三种预定遮挡，18个context。实际逐点温度保留，dry曲线有约29.99–30.24°C漂移，不伪装为精确恒温。更大原文件虽有167条温度条件，也不能说167种材料或167个独立样本。

每context给六个频率上的成对G′/G″，预算=12个标量；目标是其余七个频率上的14个标量。源中复模量、tanδ、复黏度等派生列不算新性质标签。不同预设mask可以让同一原始格在另一个context中作为观测或目标，属于常规离线遮挡实验；每个预测器只拿到自己的12个值，不用其它context的隐藏响应。

本阶段仍没有证明任意几十种无坐标性质都能用同一个函数先验恢复，也没有证明新批次总体校准或实际湿实验节约。大规模原始多性质表的样品/配方/状态身份必须另行核实。
''')
    add('三、算法主线：物理低维代码保留，完整高维残差也保留', '''
整套方法保持三个清楚的模块：物理教师负责可解释关联，参考概率过程负责完整不确定性，生成分布负责保留两种解释的分歧。数据读取、拟合、分布运算、独立审计分别放在不同文件，避免把失败修成一层层隐藏规则。

1. 物理低维代码。广义Maxwell模型用固定六个松弛时间τk，加平衡模量G∞和溶剂黏度ηs，共八个非负系数。τk仅由完整原生频率输入范围预先确定，不看隐藏响应，也不搜索模式数。

G′(ω)=G∞+Σk gk(ωτk)^2/[1+(ωτk)^2]
G″(ω)=ηsω+Σk gkωτk/[1+(ωτk)^2]

将六个观测频率的两通道方程堆成12×8线性系统。主教师以每通道自身六个观测的RMS归一化残差做非负最小二乘；这让量纲相同但数量级不同的通道不会只由大模量主导。原raw-Pa等权NNLS仍单独保留作强点基线。拟合后得到目标Pa点值，取ln作为物理位置mM。

2. 完整参考。每通道只用自身六个真实观测，拟合OLS线性趋势和Matérn3/2+WhiteKernel GP。核参数边界、零重启、残差尺度规则事前固定；预测协方差包含显式white观测方差。两通道组成14维mG和完整正定C。这里C为块对角，没有宣称学会两通道噪声的经验互协方差；它的14维残差仍完整保留。

3. 不强行压成一个均值。原方法用mhalf=(mG+mM)/2、协方差仍C。它改善了点误差，却在联合评分上受罚。独立分解证明修复方向与C的尺度/方向不相容，问题不是实现算术错误。

新候选保留整条曲线的两种解释：
p(yT|O)=0.5·t4(yT;mG,C/2)+0.5·t4(yT;mM,C/2)。

t4是完整14维Student分布，不是十四个单独Student密度的乘积。因为ν=4，component covariance=2·shape=C。一个伯努利代码为整条配对曲线选择同一分支；Student还给整条曲线共享一个径向尺度。不会给每列独立挑一个模型而破坏曲线关系。

该分布的log空间均值和完整协方差为：
m=(mG+mM)/2，Σ=C+(mM−mG)(mM−mG)ᵀ/4。

新增项是教师分歧带来的低秩不确定性，原完整C没有被丢掉。因此不是把十四个目标硬塞进八个数，也没有把剩余方差设成零。低维物理参数和一个分支代码只解释结构，完整残差还负责未解释信息。
''')
    add('四、场、势和信息时间：哪些已经实现', '''
固定均值路径mα=(1−α)mG+αmM可以作为信息时间下的平移场，残差样本保持不变时dy/dα=mM−mG。这是经典位移解释，α不是材料实际反应时间，也没有训练一个加噪/去噪扩散网络。

当前改进保留整条曲线的分布分歧，而不是只取路径中点。“基因”对应Maxwell谱的非负系数；它有物理含义，但不是生物基因，也不是已经实现微观成网模拟或Eigen化学超循环。可研究的后续闭环是：由生成场提出有价值的观测位置，接收真实测量记录，再更新物理参数和分布置信。只有真实记录能作为验证反馈，生成值不能当新测量写回训练集。

负对数密度可以写成势U(y)=−log p(y)，但它的梯度一般不等于信息时间路径的速度。两者用途不同，数学说明与直接近邻文献保存在stage/literature和scientific_review。
''')
    add('五、四个严格控制，避免把成熟公式误算成新收益', '''
functional_mixture保留一个统一2×2归因设计：Gaussian/固定ν4Student × 整曲线density-mixture/同矩单分布。四法均有相同mhalf、相同Σ，因此它们的log RMSE相同是数学恒等，不能算四次新的准确性发现。

混合与同矩单分布的得分差反映高阶分布形状；原中点Gaussian与同矩Gaussian的差反映分歧协方差。Student控制用于区分重尾本身与物理均值的作用。ν=4、权重0.5都没有在这些目标上搜索。

对整曲线密度混合，p≥0.5pG，所以相对于同tail完整GP端点，每维NLL最坏增加不超过ln2/14≈0.04951。这个界只适用于两个density-mixture，不适用于同矩近似，也不保证RMSE、覆盖或新材料表现。

Student在log空间有有限均值/协方差，但ν4第四矩无限；exp(logG)没有有限算术期望。程序示例保存log样本，Pa可报告明确的分位数；不会把exp(log均值)误称Pa算术均值。四法等权对称时它也是各边际中位数，未来变权重时这个等式不能沿用。
''')
    add('六、真实结果与仍然失败的区域','\n'.join(table)+f'''

上表是原预定内部缺口任务的六状态宏平均，不附带虚构的独立样本置信区间。新整曲线混合相对强Student GP在CRPS和联合NLL上有开发收益；同矩Student的CRPS还略低{mix['log_crps']-moment['log_crps']:.8f}，而混合的jointNLL更好，所以保留两者，不说每个指标全面获胜。

图C_primary_scores.png将点模型的未定义概率指标留空，没有填0冒充好成绩。C_all_geometries.png完整保留两种外推：低频前缀向高频外推时新Student-mixture CRPS约0.233，反而差于GP Student约0.201；高频前缀向低频外推时虽有改善，覆盖仍仅约0.595。同矩Student在后者约0.655，也离可靠的90%校准很远。

因此现阶段可主张“在该聚合物配对频谱的内部缺口任务中具有竞争力的条件生成”，不能主张任意区间外推、所有聚合物、全属性矩阵或可靠总体校准。
''')
    add('七、之前的失败为什么也完整保留', '''
stage/quantile_midpoint：树脂30条已消费曲线的固定Gaussian–Student分位中点改善了边际CRPS，但没有胜过完整GP的联合评分。解析CRPS独立核验通过；原192点积分尾部误差FAIL和修正前源码也保留，没通过放宽门槛蒙混过关。

stage/generalization/xanthan_once：黄原胶—盐水11制备条件、7盐度、1原曲线/条件，两个坐标窗口、三种mask，同预算六观测八目标。实际source/input门和数值审计通过，但预选copula方法的泛用性主门失败：物理边际欠覆盖，换依赖不能修复它。完整冻结Gaussian反而更稳。没有把357个坐标或66个context算成独立材料。

mean_capacity_once：仅沿“幂律物理均值−完整GP均值”方向，保C不变的精确全局oracle改善仅0.002637 NLL/维。它是已消费数据的容量上限，不是新确认；据此关闭这一单参数方向，没有再堆门控、调噪声或释放final20追结果。

moment_student_once：完整同一二矩Student4强控制有时改善联合NLL，却会缩窄中央90%区间；不能把“尾更重”直接解释成更可靠。这些控制与失败帮助本轮把问题收敛到真正遗漏的教师分歧信息。
''')
    add('八、程序结构与最短使用方法', '''
当前候选核心：
• stage/functional_repair/pure_laws.py：仅真实观测→两通道GP、Maxwell参数和完整C；数据读取不在这里。
• stage/functional_mixture/pure.py：build_laws、logpdf、marginal_quantiles、marginal_crps、score、sample；不读数据、不拟合、不搜索权重。
• stage/generalization/anchor_gp.py：当前六观测的常规GP强参照。
• stage/generalization/generalized_maxwell_nnls.py：固定谱raw-Pa物理点基线。
• stage/functional_repair/experiment.py、functional_mixture/study.py：冻结研究的I/O、角色和分阶段执行。
• stage/verification及各independent_audit.py：独立公式/原字段/评分核验，不调用生产评分器作为真值。

对自己的合法数据，先显式给定13个正频率Hz、六个O、七个T和2×6的真实观测Pa：

from functional_repair.pure_laws import build_context
from functional_mixture.pure import build_laws, sample
import numpy as np

context = build_context(frequency_Hz, O, T, observed_moduli_Pa)
if not context['method_availability']['half_maxwell_mean_gaussian']['available']:
    raise ValueError(context['components'])
laws = build_laws(context['gp_mean'], context['maxwell_log_location'],
                  context['full_residual_covariance'])
draws = sample(laws['whole_curve_student4_density_mixture'],
               32, np.random.default_rng(7))

这段调用会拟合自己的数据；本包保存结果的只读重放和生成示例不会重新拟合旧材料。O/T必须由调用者按真实原行定义，不能输入伪造的预测标签。上述实验版本固定13/6/7预算；通用n≥4参考模块虽已通过合成数学门，尚不能据此宣称任意维新材料都验证完成。

模型未定义/秩不足/非正模量等情况保留失败，不自动补一个σ地板、换种子或删除case。GP、weighted Maxwell和raw Maxwell的失败相互隔离：一个物理教师失败不抹掉可用GP，反过来也一样。18个context完整分母和所有方法配对可用率均在评分文件中。
''')
    add('九、复现、检查点、数据与历史在哪里', '''
解压目录的stage是当前研究，legacy_mlst是完整旧MLST封存。旧论文只供历史恢复，不是本轮新稿。requirements-current.txt给出本轮实际CPU依赖：Python3.13.5、NumPy2.1.3、SciPy1.15.3、scikit-learn1.6.1、mpmath1.3.0。绘图为可选Matplotlib3.10.0。

独立便携重放入口（完成移位验证后记录在DELIVERY_VALIDATION.json）：
python -B delivery/replay_checkpoint_C.py --package-root .

此入口重放D原PRE/POST和后续mixture gate/laws/scores五项，只核保存证据，不重拟合、不读取未消费条件。旧REG、RELEASE、原UTC与原SHA不重写；原H路径通过明确的同SHA只读映射定位包内文件，没有原机目录fallback。输出另存offline_replays/实际UTC/。较早capacity、同矩Student、quantile及B的完整审计源和原收据仍在包内；不能因为存在收据就声称它们也由这个新入口全部重跑。

所有必要XG与DaRUS原件、本轮冻结依赖、协议、预测、目标和失败均内含。EC/pyRheo/RheoGP等未准入新来源保留门报告、URL和hash；未使用、许可或身份未闭合的大原件不作为当前程序依赖混入。这不影响当前实验恢复；若未来启用这些来源，须按其记录先完成适配。

ROOT封包清单记录每个成员SHA256与相对路径，ZIP另记录CRC、Windows.NET读回和最长路径。不得将新ZIP说成旧B的更新文件；A/B和legacy原件都保持原样。GitHub研究分支保存的是源代码与审核后的汇总，不上传本地整个原始数据包或用户私人资料。
''')
    add('十、创新性、泛用性与下一步如何判断', '''
本项目接受经典方法在具体材料任务中的适配、组合和新证据作为贡献。Maxwell、Gaussian/Student、概率混合和law of total variance均有先例。文献目录明确列出Copula Processes、聚酯intrinsic-viscosity Copula-JITL、区间流变模型、pyRheo、物理特征GP等直接近邻及真实阅读范围，不声称读完未读全文或“所有论文都没有相似机制”。

当前可形成的贡献是：让可解释的低维物理谱参与多通道缺口生成，同时保留完整残差与整曲线模型分歧；用可追溯测量身份、同预算强参照和完整失败证明适用范围。当前新方法的四种形状没有新增训练网络，也没有为了追分搜α/ν/权重。

进一步泛用性按层级推进：同一曲线不同空洞→同材料不同测量状态→不同材料/来源的相应物理适配→更大无序多性质矩阵。现有XG失败表明不能用同一个不合适的幂律教师覆盖所有任务；SMP新函数收益也不能抹掉这次失败。

已授权下一有界工作是测量反馈闭环的开发检验：先从已消费六条30°C曲线模拟严格预算的4→5→6频率观测，真实记录oracle只返回选中位置，分布预测先冻结，再验证并更新。目标面板不参与选点或拟合，生成样本永不写回observed。这是离线实测记录实验，不是实际新湿实验，也不等于Eigen化学超循环已经实现。

原树脂final20运行和DaRUS其它161温度条件仍未数值释放。只有开发结果和协议足够明确才另行冻结新确认；不能因公开文件已经在硬盘里就拿它们当调参池。需要作者最终补的是可追溯的新材料/批次资料、实际用途误差和必要仪器/服务器实验；尚能在本机完成的代码、数学、源追溯与评估仍继续推进，不把未成立的核心问题直接推给作者。
''')
    text='C阶段完整资源说明书与算法讲解\n记录时间UTC：'+record_time+'\n\n'+'\n\n'.join(t+'\n'+s for t,s in sections)+'\n'
    (OUT/'01_ALGORITHM_AND_RESOURCES_ZH.txt').write_text(text,'utf-8')
    body=[]
    for i,(title,content) in enumerate(sections):
        body.append(f'<section id="s{i}"><h2>{html.escape(title)}</h2>')
        for paragraph in content.split('\n\n'):
            if paragraph.startswith('方法 |'):
                lines=paragraph.splitlines()
                cells=lambda line:''.join('<td>'+html.escape(v.strip())+'</td>' for v in line.split('|'))
                body.append('<div style="overflow:auto"><table><thead><tr>'+cells(lines[0])+
                            '</tr></thead><tbody>'+''.join('<tr>'+cells(line)+'</tr>' for line in lines[1:])+
                            '</tbody></table></div>')
            else:
                body.append('<p style="white-space:pre-wrap">'+html.escape(paragraph)+'</p>')
        if i==5:
            body.append('<img src="stage/figures/C_primary_scores.png" alt="内部缺口全部主要控制结果">'
                        '<img src="stage/figures/C_all_geometries.png" alt="全部几何及失败">')
        body.append('</section>')
    nav=''.join(f'<a href="#s{i}">{html.escape(t)}</a>' for i,(t,_) in enumerate(sections))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>C阶段研究资源与算法说明</title>
<style>body{max-width:1120px;margin:35px auto;padding:0 25px;font:17px/1.8 "Microsoft YaHei",Arial,sans-serif;color:#182535;background:#f8fafc}h1,h2{line-height:1.4}section{background:white;padding:22px 30px;margin:25px 0;border:1px solid #dbe3ee;border-radius:10px}nav{display:grid;gap:5px}a{color:#075ea7}img{max-width:100%;height:auto;margin:20px 0}p{overflow-wrap:anywhere}small{color:#516273}table{width:100%;border-collapse:collapse;font-size:14px}td{padding:9px;border:1px solid #cbd5e1;text-align:right}td:first-child{text-align:left}thead{background:#e0edf8;font-weight:bold}</style>
<h1>低维物理谱与完整概率残差：C阶段完整资源</h1>
<p>最新可审阅候选：整曲线物理教师–参考分布混合。原MLST完整封存保留在legacy_mlst；本页介绍当前新研究。</p>
<p><a href="01_ALGORITHM_AND_RESOURCES_ZH.txt">下载纯文本完整说明</a> · <a href="stage/functional_mixture/development_once/SCORES.json">最新开发评分</a> · <a href="stage/scientific_review/FUNCTIONAL_MIXTURE_RESULT_REVIEW_ZH.md">独立结果解释</a></p>
<small>本检查点有内部缺口效果支持，外推/泛化仍有不足；不是投稿认证，也未重写旧论文。</small><nav>'''+nav+'</nav>'+''.join(body)+'</html>'
    (OUT/'00_READ_FIRST_C.html').write_text(page,'utf-8')
    summary={'actual_utc':record_time,'scope':'ordinary C research checkpoint, not submission certification',
        'current_candidate':'whole_curve_student4_density_mixture','role':'reused-development after original fixed-pilot reveal',
        'primary_task':'paired dynamic-modulus internal gaps,one commercial polyurethane,six humidity/history states near30C',
        'budget_observed_scalars':12,'target_scalars':14,'original_pilot_cases':18,
        'main_results':{'GP_gaussian':gp,'GP_student4':st,'student_mixture':mix,'same_moment_student4':moment,'Maxwell_point':physics},
        'unreleased_resin_final_runs':20,'unreleased_DaRUS_other_conditions':161,'new_wet_experiments':0,
        'new_loop_material_execution_included':False,'original_paper_registry_unchanged':True}
    (OUT/'EVIDENCE_SUMMARY_C.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    (OUT/'requirements-figures-current.txt').write_text('matplotlib==3.10.0\n','utf-8')
    print(json.dumps({'manual':str(OUT/'01_ALGORITHM_AND_RESOURCES_ZH.txt'),'characters':len(text),'HTML':str(OUT/'00_READ_FIRST_C.html')}))


if __name__=='__main__':main()
