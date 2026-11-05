"""Read frozen results and build a descriptive report; no experiment execution."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'
OUT.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / 'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources = [
        ('initial_12', ROOT / 'math_design/moment_followup/execution_once/RESULTS.json',
         'cc48fe6581545569760d544da6d5a1342ce36a2ab0855a1fb3fad80787c37893'),
        ('posthoc_16', ROOT / 'math_design/moment_followup/posthoc_scaling/execution_once/RESULTS.json',
         '1ece313e34b5b4586fa7bedf0d1fbea079f14b17fcac0057395f4cbb18fdd927')]
    rows = []
    for phase, p, pin in sources:
        assert sha(p) == pin
        for c in json.loads(p.read_text('utf-8'))['cases']:
            s = c['stats']; deg = [d for d in s['moments']['degrees'] if d is not None]
            total = c['reported_total_wall_seconds']; common = c['data']['common_numeric_input_bytes']
            stream_workspace = s['streamed_exact']['algorithm_workspace_numeric_bytes']
            moment_workspace = s['moments']['algorithm_workspace_numeric_bytes']
            exact, approx = c['scores']['streamed_exact'], c['scores']['moment_or_fallback']
            rows.append({'phase': phase, 'H': c['H'], 'rmax': c['rmax'],
                'degree_min': min(deg) if deg else None, 'degree_max': max(deg) if deg else None,
                'fallback_proxies': s['moments']['fallback_proxies'],
                'selected_policy_regret_float': c['restricted_model_regret_float'],
                'strict_adaptive_gain_first_actions': c['strict_positive_gap_count_over_2e-12'],
                'max_score_truncation_abs': max(abs(a-b) for x,y in zip(exact,approx) for a,b in zip(x,y)),
                'stream_seconds_common_setup_plus_readout': total['streamed_exact'],
                'cache_seconds_common_setup_plus_readout': total['cached_exact'],
                'moment_seconds_common_setup_plus_readout': total['moment_with_fallback'],
                'stream_over_moment_time_ratio': total['streamed_exact']/total['moment_with_fallback'],
                'stream_principal_workspace_bytes': stream_workspace,
                'moment_principal_workspace_bytes': moment_workspace,
                'retained_common_numeric_input_bytes': common,
                'estimated_principal_total_memory_ratio': (stream_workspace+common)/(moment_workspace+common),
                'managed_peak_stream_bytes': c['costs']['streamed_exact']['managed_tracemalloc_peak_bytes'],
                'managed_peak_moment_bytes': c['costs']['moment_with_fallback']['managed_tracemalloc_peak_bytes']})
    assert len(rows) == 28 and len({(r['H'],r['rmax']) for r in rows}) == 28
    summary = {'actual_utc': datetime.now(timezone.utc).isoformat(), 'source_result_pins':
        [{'phase':phase,'path':str(p),'sha256':pin} for phase,p,pin in sources],
        'interpretation': '12 initially registered cases plus16 separately preregistered post-hoc scaling points; no new independent material evidence',
        'all_selected_policy_regret_float_zero': all(r['selected_policy_regret_float'] == 0 for r in rows),
        'initial12_faster_than_streamed_points': sum(r['stream_over_moment_time_ratio'] > 1 for r in rows if r['phase']=='initial_12'),
        'posthoc16_faster_than_streamed_points': sum(r['stream_over_moment_time_ratio'] > 1 for r in rows if r['phase']=='posthoc_16'),
        'timing_limit': 'One ordered timing perpoint, common input setup plus score readout only; later adaptive argmax/bound/policy arithmetic excluded; not end-to-end or robust speed ranking',
        'memory_limit': 'Principal-array estimates plus retained numeric inputs shown; managed peak is not process RSS; complete H posterior is retained',
        'material_fits_predictions_scores_new_label_reads': 0, 'rows': rows}
    (OUT/'SCALING_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (OUT/'ALL_28_COMPUTE_POINTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    fig, axes = plt.subplots(1,2,figsize=(12.6,4.9),layout='constrained')
    for rho, color in zip((.25,.6,.9,.99),('#176c9c','#29834b','#bf792a','#9b4e88')):
        a=sorted([r for r in rows if r['rmax']==rho],key=lambda r:r['H'])
        for ax,key in zip(axes,('stream_over_moment_time_ratio','estimated_principal_total_memory_ratio')):
            ax.plot([r['H'] for r in a],[r[key] for r in a],marker='o',lw=1.7,color=color,label=f'max |rho|={rho}')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([8,64,128,256,512,1024,2048])
        ax.set_xticklabels(['8','64','128','256','512','1024','2048'])
        ax.axhline(1,color='#555',lw=1,ls='--');ax.axvspan(181,2300,color='#ddd',alpha=.22)
        ax.set_xlabel('Number of relation hypotheses H');ax.grid(alpha=.2)
    axes[0].set_ylabel('Exact streamed time / moment time')
    axes[0].set_title('One ordered timing per point; >1 favors moments')
    axes[1].set_yscale('log');axes[1].set_ylabel('Exact / moment principal numeric memory')
    axes[1].set_title('Includes retained inputs; estimates, not process RSS')
    axes[0].legend(fontsize=9);axes[1].text(.03,.96,'Shaded: exploratory scaling extension',transform=axes[1].transAxes,va='top',fontsize=9)
    fig.savefig(OUT/'compute_scaling.png',dpi=170);fig.savefig(OUT/'compute_scaling.svg');plt.close(fig)

    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in (
        '原12' if r['phase']=='initial_12' else '扩展16',r['H'],r['rmax'],
        '精确回退' if r['degree_min'] is None else f"{r['degree_min']}–{r['degree_max']}",
        f"{r['stream_over_moment_time_ratio']:.3f}",f"{r['estimated_principal_total_memory_ratio']:.2f}",
        f"{r['selected_policy_regret_float']:.3g}"))+'</tr>' for r in rows)
    selected=next(r for r in rows if r['H']==2048 and r['rmax']==.6)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>关系矩编码：数学、计算结果与材料证据门</title>
<style>body{{font:17px/1.85 system-ui,sans-serif;max-width:1120px;margin:35px auto;padding:0 24px;color:#172b38}}h1,h2{{line-height:1.4}}a{{color:#126e9d}}pre{{white-space:pre-wrap;background:#f2f6f8;padding:16px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{border:1px solid #ccd7df;padding:7px;text-align:right}}th{{background:#edf3f7}}img{{max-width:100%}}.notice{{background:#fff4df;padding:16px}}</style>
<h1>关系后验与低维未来信息编码：研究检查点</h1>
<p>本轮沿顶会目标推进，得到一个可解释、带数学截断界的计算原语。它保留完整关系后验和观测分支，用少量相关矩近似后续补齐的信息价值。当前完成的是数学与合成计算证据；<b>尚未在新材料数据上拟合、生成或评分，不能称已达到材料强基线或顶会投稿标准。</b></p>
<p><a href="../root_review/ALGORITHM_AND_ORIGINAL_GOAL_ZH.txt">易懂算法解释与最初目标的对应</a> · <a href="../math_design/MATH_AND_MODEL_ZH.txt">完整模型与推导</a> · <a href="../math_design/moment_followup/IMPLEMENTATION_README.txt">矩编码程序说明</a> · <a href="SCALING_SUMMARY.json">28点完整结果</a></p>
<h2>1. 创新候选收敛到了什么</h2>
<p>不是给旧固定Gaussian关系再加权。假设有H种仍可能成立的性质依赖，真实共测通过似然更新其后验。旧单位的关系观察A，能影响新候选proxy B对缺失目标的预测价值。我们检验怎样保留这项未来信息，避免每次构造H×H的条件CDF核。</p>
<p>自催化、Bayes规划、CRPS信息价值、Taylor特征都已有先例。候选贡献是材料身份一致的共测任务、保留分支的低维价值表示、误差与失败边界，以及实证成立的空间／计算权衡。有限检索没有找到完全同题，但不保证全网唯一。见<a href="../prior_art/CONTRIBUTION_GATE_ZH.txt">直接近邻</a>与<a href="../prior_art/moment_followup/MOMENT_MATH_AND_GATE_ZH.txt">矩表示近邻</a>。</p>
<h2>2. 数学与编码</h2>
<pre>m_n(w,b) = sum_h w_h rho_(b,h)^n
a_n = choose(2n,n) / [4^n(2n−1)]
g_D(w,b) = sigma_b/sqrt(pi) * sum_(n=1..D) a_n m_n(w,b)^2
0 ≤ g−g_D ≤ sigma_b/sqrt(pi) * choose(2D,D)/4^D * r_b^(2D+2)</pre>
<p>这是指定Gaussian-copula模型内的信息价值编码，不是完整材料分布或原始矩阵已被压成D维。各候选需要自己的矩；完整H权重继续保留以便未来Bayes更新。每个第一次观察结果的概率与后验都保留，先对第二动作取max，再对第一次结果平均。</p>
<p>独立的有限联合表、CDF积分、Fraction／整数平方根区间审计已通过。关键反例同时保留：同一个后验协方差摘要可以产生不同自适应收益；D=1可以把真正更好的候选排在后面；相关接近1时低阶近似不足。数学截断界不包含浮点、模型失配或真实测量误差，不是IEEE机器严格区间。</p>
<h2>3. 实际结果：先保留不利结果，再看规模</h2>
<p>第一组12个预登记中小规模点全部比优化流式精确参考更慢。随后另行登记16个大H点，保持原源码、epsilon=10⁻⁶、Dcap=64、所有4个相关上限和全部结果；它是<b>后验探索性计算扩展</b>，不是独立材料确认。扩展中11/16点比流式参考快，接近1时回退并无稳定速度优势。</p>
<p>代表点H=2048、r=0.6：共同输入准备＋分数读出为{selected['stream_seconds_common_setup_plus_readout']:.3f}→{selected['moment_seconds_common_setup_plus_readout']:.3f}秒；算法主数组估计为{selected['stream_principal_workspace_bytes']:,}→{selected['moment_principal_workspace_bytes']:,}字节。加上两者仍保留的共同输入后，主数值数组估计比约{selected['estimated_principal_total_memory_ratio']:.1f}。这些不是完整RSS、端到端智能体延迟或真实实验节约。</p>
<p>28点的所选受限R-first策略，相对精确同家族参考的保存浮点价值regret均为0。它不代表任意输入／完整POMDP／真实材料上零误差；预定值误差允许10⁻⁶截断加单列数值容差。一次有序计时不构成普遍加速排名。</p>
<p>独立保存数值核算覆盖原12点：36个全核位置、所有3072项保存包络、1792个非回退moment重建及12个实际策略。16点扩展另做了逐例字节／来源／结果绑定的只读核查，并复用相同未改源码；不能称全部28点都做过另一套全核独立重算。</p>
<img src="compute_scaling.png" alt="全部28点计算和主数值数组权衡，含所有失败速度点与近1回退">
<table><tr><th>阶段</th><th>H</th><th>r</th><th>D</th><th>流式时间/矩时间</th><th>主数组总量比</th><th>策略regret</th></tr>{table}</table>
<h2>4. 必须保留的失败与限定</h2>
<p>此前6个经典缓存计时案例中，240个首动作都没有严格adaptive相对best-fixed收益；13处proxy索引变化只是等值。它们没有被改写，见<a href="../independent_implementation/SAVED_RESULTS_INTERPRETATION_ZH.txt">独立解释</a>。新12点是不同的已声明合成统计族，不是对旧案例的独立确认。</p>
<p>对称±10⁻⁸相关的分布形状价值约7.052×10⁻³⁴；直接二次型浮点读出为0，D2正项表示与稳定H2强参考均保留正值。但最终值超出浮点范围仍须拒绝；旧原型的广义极端数值边界没有被冒充已修复。同rho可分离时简单解析排序就是强解，不能用慢循环制造创新。</p>
<p>当前只研究纯关系观察后接一个proxy的子家族；B的整个边际必须与关系假设无关，旧／新单位给定关系后独立。混合panel、共享样品残差、直接取得当前目标、E先R后和后续付费V要进入完整联合规划。纯B本身不会学习关系，尚未实现完整生成—验证—自迭代智能体。</p>
<h2>5. 下一步：进入真正材料任务，停止在合成上堆故事</h2>
<p>元数据上仍有JARVIS8802个未消费身份组；它们是同一DFT表内的新组，不能称外部湿实验。UCSB仅余22个来源组。下一步先按<a href="../data_feasibility/MATERIAL_TASK_CONTRACT_DRAFT_ZH.txt">材料执行门草案</a>落实原始字段、共测身份、Gaussian边际／双proxy星形假设、真实动作／情景成本，再登记新组和源码。强参考必须拥有同后验、同动作、V／STOP／不同查询顺序和相同预算。</p>
<p>这轮不需要作者现在补湿实验，也不能说只差湿实验就可投稿：实际材料有效性和任务适配仍须研究。MLST原稿、原模型与全部已有材料结果独立封存，不混入本轮结论。</p>
<h2>来源与复现入口</h2>
<p><a href="../root_review/ROOT_READING_LEDGER.json">root实际阅读</a> · <a href="../prior_art/DIRECT_NEIGHBOR_LEDGER.json">主近邻选读</a> · <a href="../prior_art/moment_followup/MOMENT_PRIOR_ART_LEDGER.json">矩编码近邻</a> · <a href="../independent_math/POST_EXECUTION_READ_ONLY_REVIEW.json">CDF／联合表审计</a> · <a href="../independent_moment_math/POST_EXECUTION_REVIEW.json">有理区间审计</a></p>
<p>基础出处包括<a href="https://proceedings.mlr.press/v22/kar12.html">Kar–Karnick 2012</a>、<a href="https://arxiv.org/html/2503.11250v2">化学空间CRPS序贯设计</a>、<a href="https://proceedings.mlr.press/v235/buathong24a.html">p-KGFN</a>。只声称账本中实际选读范围，没有声称各100篇全文读完。</p></html>'''
    (OUT/'RESULTS_AND_NEXT_STEPS_ZH.html').write_text(page,encoding='utf-8')
    index='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>关系矩编码研究资源</title><body style="font:19px/1.9 system-ui;max-width:950px;margin:40px auto;padding:0 22px"><h1>关系矩编码：新研究检查点</h1><p>核心是保留后验分支、用受控相关矩计算未来补齐价值。已完成数学审计、12个预登记计算点及16个单独登记的探索性规模点；尚无新材料效果或顶会达标结论。</p><ol><li><a href="report/RESULTS_AND_NEXT_STEPS_ZH.html">完整中文报告：算法、全部正负结果与下一步</a></li><li><a href="root_review/ALGORITHM_AND_ORIGINAL_GOAL_ZH.txt">易懂算法说明</a></li><li><a href="math_design/moment_followup/IMPLEMENTATION_README.txt">程序与运行边界</a></li><li><a href="data_feasibility/MATERIAL_TASK_CONTRACT_DRAFT_ZH.txt">真实材料任务草案</a></li></ol><p>新旧源码、实际UTC协议、结果、来源与独立审计均保留。完整资源范围与复现说明见同目录RESOURCE_GUIDE_ZH.txt，文件哈希见PACKAGE_MANIFEST.json。旧47.4MB与MLST归档为外部历史资源，均不覆盖。</p></body></html>'''
    (ROOT/'INDEX.html').write_text(index,encoding='utf-8')
    print(json.dumps({'report_built':True,'rows':len(rows),'initial_wins':summary['initial12_faster_than_streamed_points'],
                      'posthoc_wins':summary['posthoc16_faster_than_streamed_points'],'new_experiment_calls':0}))


if __name__=='__main__':
    main()
