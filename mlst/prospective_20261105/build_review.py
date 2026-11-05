"""Build the bounded prospective-preparation review; never read material labels."""
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(value)


def main():
    status = json.loads((ROOT / 'STAGE_STATUS.json').read_text('utf-8'))
    if status['independent_synthetic_audit_status'] != 'PASS':
        raise ValueError('Do not build a qualified handoff before independent synthetic audit passes')
    table_rows = [
        ('独立同域确认', '未完成', '本轮4个不同查询没有新增准入单位；一个Mendeley候选经历史核对为旧来源。不是全网不存在数据的证明。', '先取得真正未消费、状态相容、有原始单位身份的新曲线。'),
        ('独立校准', '工具与数学核查完成；材料实证未完成', '新增完整单位最大残差、精确顺序统计量与预测带工具；仅在人工数值上验证。', '独立校准单位和确认单位分开；同一独立单位的相关曲线、mask不得跨角色（新配方按配方，已知配方新制备按制备组）。'),
        ('表示稳健性', '已有受限开发证据', '8个新增固定随机表示、先验KL与45°旋转已完成；不可再当本轮新增实验。', 'KL更好的开发CRPS及不利密度/局部表现同时保留。'),
        ('FPCA／CNP通用函数基线', '未执行', '先验KL是参考核特征方向，不是用训练曲线估计的经验FPCA。现有conditional Gaussian也不能改名冒充FPCA。', '若加入，另定小容量、折内训练、固定选择规则及资源预算；在新确认揭示前冻结。'),
        ('适用域／拒绝回退', '有失效分析；没有确认过的策略', 'P0和跨来源欠覆盖已入稿；不能根据这些已知目标挑最有利阈值。', '仅用开发资料确定风险规则，再与实际新单位一起评价；本包不自动拒绝材料。'),
        ('锚点数量／位置', '六锚点主任务已固定', '本包不扫描3／4／8／10点，也不把不同锚点跨度解释成单纯数量效应。', '如改变问题，先登记坐标几何、评价范围和方法，不能回写旧主终点。'),
        ('真实实验资源节约', '未测量', '6／39是观测比例；不等于时间、样品或仪器节约。', '需要人工记录加载、平衡、预剪切、扫描、清洗与失败重测。'),
        ('第二类响应曲线', '未执行；不是当前域内论文的必需主张', '四个外部来源是已消费迁移压力测试，包含材料状态差异。', '只有能提出清楚科学问题且有合法新单位时再扩展。'),
        ('公开复现与DOI', 'Git与本地资源已有；发表访问／权利待定', 'MLST政策推荐可信仓库DOI，DOI并非唯一硬性载体；原数据及第三方资产再用不能由软件代授权。', '保留代码与原资产权利表，作者落实发表时可验证／复现访问；本轮不创建DOI、不发布原数据。'),
        ('主文与附信', '17页主文／27页SI／64参考已封存', '本轮没有新材料结果，不把未来校准写成已完成；原论文和模型哈希不变。', '作者完成权利、机构、基金、贡献、冲突、ORCID及实质AI使用声明。'),
    ]
    rows = ''.join('<tr>' + ''.join('<td>' + escape(x) + '</td>' for x in row) + '</tr>' for row in table_rows)
    source_status = status['new_admitted_independent_units']
    audit_families = status['independent_synthetic_audit_families']
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>MLST：独立校准与确认准备增量</title>
<style>body{{font:17px/1.8 "Microsoft YaHei",sans-serif;max-width:1150px;margin:36px auto;padding:0 25px;color:#18303e}}h1{{font-size:30px}}h2{{font-size:23px;border-top:1px solid #ccd;padding-top:18px}}.note{{background:#eef4fa;border-left:5px solid #39749c;padding:15px 20px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd;padding:10px;vertical-align:top}}th{{background:#eaf0f4}}code,pre{{font-family:Consolas,monospace}}pre{{white-space:pre-wrap;background:#f3f5f7;padding:14px}}a{{color:#135b8b}}.small{{font-size:14px;color:#50606b}}@media print{{body{{margin:0;max-width:none;font-size:11pt}}table{{font-size:9pt}}h2{{break-after:avoid}}tr{{break-inside:avoid}}}}</style>
<h1>MLST：独立校准与确认准备增量</h1>
<p class="small">实际构建UTC：{escape(datetime.now(timezone.utc).isoformat())}。研究轮次20261105T020511590098Z。手动启动；自动化与持续Goal保持暂停。</p>
<div class="note"><b>本轮完成的是未来独立校准／确认的数学、程序和交接准备。</b>新增准入独立材料单位 {source_status}；真实材料拟合0、预测0、目标评分0、校准0。独立人工数值核查通过 {audit_families} 个测试族。软件核查不是新材料的覆盖率或新算法性能。</div>
<h2>1．保留的主线与当前证据</h2>
<p>主任务仍是水凝胶前驱墨水流变曲线：给定六个剪切率—黏度锚点，续接较高剪切率区域。现有部署模型仍为“固定局部曲率表示＋共享Gaussian五参数修正＋完整高维GP残差”。保留整个高维随机残差，避免把全曲线强行压成秩二分布；不新增扩散或门控网络。</p>
<pre>Y = m₀ + C z + ε,  z ~ N(a,S),  ε ~ N(0,K₀−CCᵀ)
E[Y] = m₀ + Ca;  Cov[Y] = K₀ + C(S−I)Cᵀ</pre>
<p>25个配方及其三种布局已反复用于开发。原曲率Gaussian宏CRPS为0.0432770516；新增的经典先验KL表示在同一开发研究中为0.0371389984，低14.18%，19／25配方更好。KL并非所有指标均胜出：联合密度更差、中间缺口仅8／25胜出，P0仍欠覆盖。八个新增固定随机表示均比原曲率更差，但这不是从随机方向分布推断“优于95%所有随机表示”的充分证据。</p>
<p>这些结果已在上一资源包和论文中保存，不在本轮重拟合。46条外部曲线仍是来源已检查的迁移压力测试；其中状态、化学体系和协议同时变化，不能直接认证原任务的新批次泛化。当前创新继续定位为具体水凝胶续接问题中的表示与训练适配和受控证据，不宣称Gaussian修正、KL或conformal为首创。</p>
<h2>2．十项建议与实际进度对照</h2>
<table><thead><tr><th>建议</th><th>当前状态</th><th>事实与边界</th><th>真正下一步</th></tr></thead><tbody>{rows}</tbody></table>
<h2>3．单位级预测带：能证明什么</h2>
<p>先选择一个科学主张：“新配方”或“已知配方的新制备批次”。对前者，同一规范化配方的全部相关曲线／mask归为一个单位；对后者，以真正独立制备组为单位，已有配方可以重用，但旧制备、旧原始曲线及其别名不能改名充新。程序只能检查声明过的身份账本，不能证明元数据真实、物理独立、盲态、许可或交换性。</p>
<p>每个方法在校准单位g的所有预定曲线、mask及有限目标网格上，取最大标准化残差。不同方法用相同校准单位、坐标和alpha，各自求阈值；主方法在校准标签揭示前已固定，不按校准q或宽度选择胜者；确认集不得参与阈值或方法选择。每个固定方法的保证不自动成为全部方法同时保证或事后选优保证。</p>
<pre>s_g = max_{{r属于g, j属于H_r}} |y_rj − μ_rj| / σ_rj
k = ceil((n_cal + 1)(1 − α))
q = 第k个排序后的s_g，当k ≤ n_cal；否则q = +∞
B_rj = [μ_rj − qσ_rj, μ_rj + qσ_rj]</pre>
<p>在固定预测规则、完整单位对象交换、同样的取样及查询政策下，有限样本秩论证给出<b>新单位所有声明网格同时被带覆盖的边际保证</b>。曲线内点不需要独立；校准单位也不能由这些点扩充。它不是对每种化学体系、每个配方、每个批次或网格之间连续曲线的条件保证，更不自动适用于换域。</p>
<p>alpha使用字符串精确有理数，例如<code>"0.1"</code>，不由浮点误差下调顺序索引。90%水平至少9个校准单位才可能得到有限阈值，95%至少19个；这只是代数下限，不是建议样本量、精度保证或确认样本量。代码允许数学核心返回无限带，真实登记模板不替作者填N。数值溢出形成的无限分数与样本不足分开标记，不夹断为有限数。</p>
<p>预测带只是集合，<b>没有人为构造CRPS或联合密度</b>。原始Gaussian的CRPS和联合密度继续评分原分布；另外报告带的有限网格覆盖、宽度及同alpha区间评分。无限带以严格JSON中的null与明确标志记录；它的覆盖是空泛的，宽度无界，不输出一个虚假的有限区间评分。</p>
<h2>4．未来主终点：原生对数剪切率的积分权重</h2>
<p>所有目标必须是最高锚点以上的已登记原生坐标，不能插值或生成“测量”。令隐藏坐标x₁&lt;…&lt;x_m为ln(γ)，m至少2。采用规范化梯形权重：</p>
<pre>w₁ = (x₂−x₁)/(2(x_m−x₁))
w_m = (x_m−x_{{m−1}})/(2(x_m−x₁))
w_j = (x_{{j+1}}−x_{{j−1}})/(2(x_m−x₁)), 1&lt;j&lt;m
CRPS_curve = Σ_j w_j CRPS_j</pre>
<p>积分域准确地是<b>第一个隐藏原生点至最后一个隐藏原生点</b>。最高锚点到第一个隐藏点之间的空隙单独报告，不能假装已被积分。单个目标点和数值上重合的对数坐标拒绝作为此主终点；不加epsilon凑跨度。它与旧39坐标、每mask33目标的开发宏指标是不同的未来终点，旧表数值不被重算或覆盖。</p>
<p>同一原始曲线的预定mask先等权，单位内曲线再等权，最后独立单位等权。相同原始曲线的重复mask必须绑定相同测量文件哈希，重叠坐标的真值也必须一致。程序保留逐单位配对CRPS差值，但不编造样本量依据、实用容差、置信区间、非劣等或显著优越结论。</p>
<h2>5．四模块边界与执行顺序</h2>
<table><tr><th>文件</th><th>单一职责</th></tr><tr><td>code/contract.py</td><td>严格元数据、规范身份、物理组与角色、原生坐标、输入／源码绑定；拒绝登记元数据包络中的未知字段／目标数组。</td></tr><tr><td>code/calibration.py</td><td>单位最大残差、精确顺序统计量与独立预测集合对象。</td></tr><tr><td>code/evaluation.py</td><td>原Gaussian评分与独立的预测带诊断；曲线／单位权重。</td></tr><tr><td>code/cli.py</td><td>只写新文件的登记、冻结、揭示与阶段收据；不训练、不运行预测器。</td></tr></table>
<pre>① 元数据与设计登记
② 所有校准单位、所有固定方法预测冻结
③ 校准标签揭示，冻结每个方法的q
④ 所有确认预测与预测带冻结
⑤ 确认标签揭示并一次性评价</pre>
<p>阶段完成收据记录SHA256；后续阶段先核前序文件，防止单个冻结文件被无意改动。尝试失败、即将打开标签、读到字节、成功解析标签分别记录。源文件变化、身份交叉、方法缺失、网格改变、不正定协方差、非正标准差、缺失成员、按已声明账本可识别的旧目标复用均停止；不回退到“成功子集”。本地收据链不能阻止恶意同时改写全部文件，也不能证明人没有在别处预读标签。</p>
<p>此工具接收原冻结预测器输出的均值和完整协方差，不新造一个预测模型。其高层包络需引用真实模型、runtime、锚点输入、预测源码、原测量文件的哈希。它不会替作者查明物理身份或自动把任意文件宣布为独立材料。</p>
<h2>6．独立数值核查与失败保留</h2>
<p>审计者与实现者分开。在执行前记录实际UTC、四模块和审计脚本哈希；人工夹具在脚本中明确标注SYNTHETIC。核查精确有理秩索引与穷举留一排名、并列／零N／无穷、非法尺度／权重、Gaussian CRPS的独立CDF积分、非均匀梯形积分、物理单位交叉、真实空模板拒绝、冻结后单文件变化拒绝、9个人工校准组＋3个人工确认组＋3个方法的完整CLI。9和3是软件夹具，不是未来材料样本量。</p>
<p>完整原始计划、命令、返回值、人工输出与审计结果位于<a href="audit/">audit/</a>。这轮的实际执行摘要见<a href="STAGE_STATUS.json">STAGE_STATUS.json</a>。第一次语法版本未执行SUT；静态缺陷修正前的v1及其收据保存在<a href="history/">history/</a>，不覆盖失败，也不把修复次数当科研增量。</p>
<h2>7．作者接手所需的最少事实</h2>
<ol><li>先决定要证明新配方还是新制备重复性，并核对现有开发及外部来源的规范身份、原曲线／批次别名。</li><li>给出未消费、状态相容的新单位；选择固定温度、材料状态、扫描／预剪切／平衡协议及原生坐标准入。</li><li>冻结实际六锚点政策。原开发[0,3,6,9,12,15]、外部“最低六个原生点”和旧物理比例提案不等同。</li><li>结合实际容差、最小有意义效应及区间精度规划，决定校准／确认各自N和alpha。当前空模板故意不能登记。</li><li>审查模型／基线面板与真实哈希。加入KL／FPCA需在新标签前另做合法开发及冻结，不能把本轮工具当已完成新模型。</li><li>落实MLST发表访问和第三方资产再用权利、作者机构／基金／贡献／冲突／ORCID与实际AI使用。DOI、许可、署名不由本程序虚构。</li></ol>
<p>进一步的材料校准与确认在这些事实确定前停止。本包已经把能离线完成的数学、程序边界和复现入口落实；它不替代湿实验，也不把“需要新证据”说成论文一定不能投稿或保证补齐后录用。</p>
<h2>8．入口与原件关系</h2>
<ul><li><a href="code/README.txt">程序格式、命令及限制</a>；<a href="code/DESIGN_TEMPLATE.json">待定设计模板</a>；<a href="code/COHORT_TEMPLATE.json">待定单位账本</a></li><li><a href="author_handoff/AUTHOR_NEXT_STEPS_ZH.txt">作者操作单</a>；<a href="author_handoff/PRIORITY_AND_AUTHOR_FACTS.json">十项建议对照／未定事实</a></li><li><a href="theory/mathZH.txt">数学细节</a>；<a href="theory/THEORY_AND_PROTOCOL_REVIEW.json">原论文、版本、实际阅读深度与许可记录</a></li><li><a href="sources/SOURCE_ADMISSION_FACTS_ZH.txt">来源准入说明</a>；<a href="sources/SOURCE_ADMISSION_REVIEW.json">实际查询与旧来源判定</a></li><li><a href="PRIOR_ARCHIVES.json">原17／27页论文、完整科学包与哈希</a>。本包是轻量增量，不重复包含610MB原包、原始测量或Python环境。</li></ul>
<p class="small">主要理论来源：<a href="https://arxiv.org/abs/2106.01792">Diquigiovanni等，conformal functional band</a>；<a href="https://arxiv.org/abs/1604.04173">Lei等，split conformal</a>；<a href="https://arxiv.org/abs/2107.07511">Angelopoulos与Bates，conformal教程</a>；<a href="https://arxiv.org/abs/2306.06342">Lee等，分层数据的预测推断</a>。本轮实际阅读是账本所列选节，不声称全文或各100篇已读。IOP：<a href="https://publishingsupport.iopscience.iop.org/iop-publishing-open-data-policy/">MLST适用开放数据政策</a>、<a href="https://publishingsupport.iopscience.iop.org/journals/machine-learning-science-and-technology/about-machine-learning-science-technology/">期刊范围与作者说明</a>。当前工具是完整单位标量秩校准，不冒称实现Hierarchical Conformal Prediction。</p>
</html>'''
    write_new(ROOT / 'RESEARCH_REVIEW_ZH.html', content)
    write_new(ROOT / 'READ_FIRST.html', '<!doctype html><html lang="zh"><meta charset="utf-8"><title>研究交接入口</title><body style="font:18px/1.8 sans-serif;max-width:900px;margin:40px auto"><h1>独立校准与确认准备</h1><p>没有新增独立材料结果；本包补齐可执行流程和独立人工数值核查。</p><p><a href="RESEARCH_REVIEW_ZH.html">打开详细中文说明、数学原理与下一步</a></p><p><a href="code/README.txt">程序使用说明</a> · <a href="author_handoff/AUTHOR_NEXT_STEPS_ZH.txt">作者接手操作单</a> · <a href="STAGE_STATUS.json">本轮事实摘要</a></p><p>原17页主文与27页SI不变；大包未重复内置，位置与哈希见<a href="PRIOR_ARCHIVES.json">原件索引</a>。</p></body></html>')
    bindings = {name: sha(ROOT / name) for name in ('RESEARCH_REVIEW_ZH.html', 'READ_FIRST.html', 'STAGE_STATUS.json', 'PRIOR_ARCHIVES.json')}
    write_new(ROOT / 'REVIEW_BUILD_RECEIPT.json', json.dumps({'actual_utc': datetime.now(timezone.utc).isoformat(), 'builder_sha256': sha(__file__), 'outputs_sha256': bindings, 'material_data_read': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
