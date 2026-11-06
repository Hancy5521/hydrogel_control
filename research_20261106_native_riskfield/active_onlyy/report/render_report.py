"""Aggregate completed frozen results and plot; no scientific-model execution."""
import argparse,csv,hashlib,html,json,os
from datetime import datetime,timezone
from pathlib import Path
POLICIES=('native_student','native_gaussian','latent_ivr','random','gp_variance','warm_openloop')
BUDGETS=(0,4,8)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--results',required=True);a.add_argument('--output-dir',required=True);a.add_argument('--independent-score-audit',required=True);a.add_argument('--independent-score-plan',required=True);a=a.parse_args()
 stage=Path(__file__).resolve().parents[1];source=Path(a.results).resolve();out=Path(a.output_dir).resolve();out.relative_to(stage)
 r=json.loads(source.read_text(encoding='utf-8'))
 assert r['status']=='COMPLETE_ACTIVE_ONLYY_FROZEN_DEVELOPMENT_COMPARISON'
 audit_path=Path(a.independent_score_audit).resolve();audit=json.loads(audit_path.read_text(encoding='utf-8'))
 assert str(audit['status']).startswith('PASS')
 audit_plan_path=Path(a.independent_score_plan).resolve();audit_plan=json.loads(audit_plan_path.read_text(encoding='utf-8'))
 assert sha(audit_plan_path)==audit['plan_sha256'], 'Audit must bind actual independent registered plan'
 audit_result=audit_plan['inputs']['score_result']
 assert Path(audit_result['file']).resolve()==source and audit_result['sha256']==sha(source), 'Audit plan must bind actual result'
 assert r['roles']=={'warm':64,'shared_query_pool':24,'terminal_development':64}
 assert r['actual_sklearn_fits']==18 and r['fixed_perpolicy_onlyY_cost']==8 and r['common_setup_native_cells']==368
 assert [len(r[k]) for k in ('primary_NIG_forecast_scores','strong_reference_forecast_scores','secondary_prequential_scores','per_terminal_paired_differences')]==[1152,960,48,64]
 plan=stage/'EXECUTION_PLAN_REGISTERED.json';assert sha(plan)==r['plan_sha256']
 assert json.loads(plan.read_text(encoding='utf-8'))['source_sha256']==r['source_sha256']
 assert all(sha(stage/'code'/k)==v for k,v in r['source_sha256'].items())
 m=r['summary_NIG_prediction_by_policy'];refs=r['summary_strong_native_prediction_references']
 assert set(m)==set(POLICIES) and set(refs)=={'native_student','random','gp_variance'}
 rows=[]
 def add(policy,predictor,n,v):
  assert v['formula_groups']==64
  rows.append({'query_policy':policy,'predictor':predictor,'new_onlyY_cost':n,'common_cells':368,'terminal_groups':64,
   'CRPS_native':v['mean_CRPS_native'],'RMSE_predictive_median_native':v['RMSE_predictive_median_native'],
   'coverage80':v['coverage']['0.8'],'coverage90':v['coverage']['0.9'],'coverage95':v['coverage']['0.95'],
   'width90_native':v['mean_interval_width_native']['0.9']})
 for policy in POLICIES:
  assert set(m[policy])=={'0','4','8'}
  for n in BUDGETS:add(policy,'NIG',n,m[policy][str(n)])
 for policy,models in refs.items():
  assert set(models)==({'GP'} if policy=='gp_variance' else {'GP','RF'})
  for method,bs in models.items():
   for n in BUDGETS:add(policy,method,n,bs[str(n)])
 assert len(rows)==33
 out.mkdir(parents=True,exist_ok=False)
 with (out/'SUMMARY_BY_POLICY.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 candidate=m['native_student']['8']['mean_CRPS_native'];warm=m['native_student']['0']['mean_CRPS_native']
 sequences=r['actual_selected_pool_sequences'];assert set(sequences)==set(POLICIES)
 assert all(len(v)==8 and len(set(v))==8 and all(type(i) is int and 0<=i<24 for i in v) for v in sequences.values())
 same_three=sequences['native_student']==sequences['native_gaussian']==sequences['warm_openloop']
 ivr_same_final=set(sequences['latent_ivr'])==set(sequences['native_student'])
 ivr_same_first4=sequences['latent_ivr'][:4]==sequences['native_student'][:4]
 comparisons=[{'reference':x['query_policy']+'/'+x['predictor'],'CRPS':x['CRPS_native'],'candidate_minus_reference':candidate-x['CRPS_native']} for x in rows if x['new_onlyY_cost']==8 and (x['query_policy'],x['predictor'])!=('native_student','NIG')]
 summary={'status':'READ_ONLY_COMPLETED_RESULT_AGGREGATE','actual_utc':datetime.now(timezone.utc).isoformat(),
  'results_sha256':sha(source),'independent_score_audit_sha256':sha(audit_path),'independent_score_plan_sha256':sha(audit_plan_path),'independent_score_status':audit['status'],'plan_sha256':r['plan_sha256'],'source_sha256':r['source_sha256'],'report_script_sha256':sha(__file__),
  'primary_comparison':r['primary_comparison'],'candidate_final_minus_warm':candidate-warm,'all33summaryrows':rows,
  'final_comparisons_descriptive_only':comparisons,'policy_path_comparisons':r['policy_path_comparisons'],
  'actual_selected_pool_sequences':sequences,'native_Gaussian_openloop_same_sequence':same_three,'IVR_same_final_set':ivr_same_final,'IVR_same_first4':ivr_same_first4,
  'counts':{'components_development':152,'warm':64,'shared_pool':24,'terminal':64,'logical_terminal_laws':2112,'prequential_laws':48,'GPfits':13,'RFfits':5,'common_setup_native_cells':368,'newY_perpolicy':8},
  'no_new_fit_query_CDF_scoring_bootstrap_or_inferential_test':True}
 write(out/'REPORT_AGGREGATE.json',summary)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(15,6),constrained_layout=True)
 colors=('#1261a0','#e78524','#26856b','#767676','#a0509e','#a64c4c')
 for policy,color in zip(POLICIES,colors):axes[0].plot(BUDGETS,[m[policy][str(n)]['mean_CRPS_native'] for n in BUDGETS],marker='o',label=policy,color=color)
 axes[0].set_title('Query policies, common actual-updated NIG predictor')
 if same_three and ivr_same_final:axes[0].text(.02,.03,'Student / Gaussian / IVR / open-loop final laws coincide',transform=axes[0].transAxes,fontsize=8,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
 axes[1].plot(BUDGETS,[m['native_student'][str(n)]['mean_CRPS_native'] for n in BUDGETS],marker='o',linewidth=2.3,label='native_student / NIG',color=colors[0])
 for policy,models in refs.items():
  for method,bs in models.items():axes[1].plot(BUDGETS,[bs[str(n)]['mean_CRPS_native'] for n in BUDGETS],marker='s',linestyle='--',label=policy+' / '+method)
 axes[1].set_title('Strong predictors on declared same-cost trajectories')
 for ax in axes:
  ax.set_xticks(BUDGETS);ax.set_xlabel('Only-Y query cost; common setup = 368 cells');ax.set_ylabel('Mean native CRPS (lower is better)');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
 fig.suptitle('64 terminal development groups; 24 shared pool; no independent confirmation',fontsize=12)
 fig.savefig(out/'learning_curves.png',dpi=180,facecolor='white');plt.close(fig)
 def fmt(x):return '\u2014' if x is None else f'{x:.6f}'
 table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (x['query_policy'],x['predictor'],x['new_onlyY_cost'],fmt(x['CRPS_native']),fmt(x['RMSE_predictive_median_native']),f"{round(x['coverage90']*64)}/64 ({x['coverage90']:.1%})",fmt(x['width90_native'])))+'</tr>' for x in rows)
 cmp=''.join('<tr><td>'+html.escape(x['reference'])+'</td><td>'+fmt(x['CRPS'])+'</td><td>'+fmt(x['candidate_minus_reference'])+'</td></tr>' for x in comparisons)
 links=[]
 for label,path in [('注册协议',plan),('6文件冻结',stage/'SOURCE_FREEZE.json'),('IO审校',stage/'independent_io_review/STATIC_IO_FAIRNESS_CLEARANCE.json'),('保存IO核查',stage/'independent_io_review/SAVED_FORECAST_IO_AUDIT.json'),('独立评分核查',audit_path),('主执行源码',stage/'code/run_active.py'),('原始结果',source)]:
  links.append('<a href="'+html.escape(os.path.relpath(path,out).replace('\\','/'),quote=True)+'">'+label+'</a>')
 pc=r['primary_comparison']
 path_table=''.join('<tr><td>'+html.escape(policy)+'</td><td>'+html.escape(str(sequences[policy]))+'</td></tr>' for policy in POLICIES)
 path_note=('本轮native_student、native_gaussian与warm_openloop的8次选点序列完全相同。真实Y更新确实执行，但未改变这些策略的动作；不能把预测改善归因于native或有限df特有的闭环选点。' if same_three else '策略路径是否不同及效果方向按冻结结果列出；存在真实更新不自动证明动作增量。')
 if ivr_same_final and ivr_same_first4:path_note+=' 潜变量IVR在预算4具有相同前缀、预算8具有相同最终集合，因此这几个检查点的相同NIG结果不能当独立机制优势。'
 page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生风险主动反馈开发结果</title><style>body{font:16px/1.7 'Microsoft YaHei',sans-serif;color:#203040;max-width:1250px;margin:35px auto;padding:0 22px}h1{font-size:29px}h2{font-size:22px;margin-top:30px}.note{background:#f1f5f8;border-left:5px solid #1261a0;padding:14px 18px}.warn{background:#fff5e8;padding:14px 18px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px 9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f6}img{width:100%;height:auto}code{word-break:break-all;font-size:12px}.scroll{overflow:auto}a{color:#1261a0}</style><h1>原生单位风险驱动的材料性质主动反馈</h1>
<div class="note">152个开发组件：64 warm、24共用池、64终端。六策略各8次only-Y查询，共同信息368个原生cell。2112个终端预测坐标加48个前序预测坐标，不是2160份独立材料。</div>
<h2>实际效果</h2><p>native_student最终平均原生CRPS <b>__CANDIDATE__</b>，相对共同warm起点变化 <b>__CHANGE__</b>。候选/最强已列参照之比 <b>__RATIO__</b>，是否进入预设2%描述带：<b>__WITHIN__</b>。差值负为候选更好；该带不是非劣效检验或统计显著性结论。</p><img src="learning_curves.png" alt="所有策略与强参照冻结学习曲线"><div class="scroll"><table><tr><th>最终参照</th><th>参照CRPS</th><th>候选−参照</th></tr>__CMP__</table></div>
<h2>必须保留的动作路径证据</h2><div class="warn">__PATHNOTE__</div><table><tr><th>策略</th><th>实际池索引顺序（原始元数据顺序）</th></tr>__PATHTABLE__</table><h2>完整比较与概率区间</h2><p>同一真Y更新NIG比较查询策略；GP/RF在native_student和random轨迹获得同样真实训练集；GP主动策略另保留自己的GP预测。18次sklearn拟合为13 GP和5 RF，共享warm及GP step4复用均明确记账。下表同时列90%区间覆盖与宽度，数值正确不等于概率校准；没有在这64个终端标签上选择新缩放或参数。区间单位沿用源bulk字段GPa归属，仍受历史聚合口径限制。</p><div class="scroll"><table><tr><th>查询策略</th><th>预测器</th><th>新Y成本</th><th>原生CRPS</th><th>中位数RMSE</th><th>90%覆盖</th><th>90%宽度</th></tr>__TABLE__</table></div>
<h2>创新归因和限制</h2><p>保留完整NIG不确定性，在warm固定变换下以原生单位跨目标proper-score期望风险选点，并与潜变量IVR、潜矩匹配Gaussian原生风险、随机、GP方差与标签盲开放环比较。开放环选择用warm均值、设计V和已知查询计数，实际预测仍用真Y更新，不生成伪测量。proper-score主动学习、NIW/NIG、Student-t共享尺度均有先例，任务适配的价值必须由这些公平对照实证，不能重命名为新原理。</p><div class="warn">全部152是开发角色；反事实复用24池不增加材料N。标签为既有发布计算报告，不是新湿实验或新DFT。本轮是连续NIG，无spike机制、自催化智能体、跨材料域泛化或实际实验节约的保证。完整gzip由可信解析器解析，API控制模型可见字段，不是密码学盲测。不得在已消费终端上调参/重切。</div>
<h2>复核与资源</h2><p>__LINKS__</p><p><a href="SUMMARY_BY_POLICY.csv">全部摘要CSV</a> · <a href="REPORT_AGGREGATE.json">聚合结果与SHA</a> · <a href="REPORT_FILES.json">报告文件清单</a></p><p>只读聚合与绘图，无重拟合、查询、原生CDF评分、bootstrap或推断检验。报告UTC __UTC__；结果SHA <code>__SHA__</code>。</p></html>"""
 for k,v in {'__CANDIDATE__':fmt(candidate),'__CHANGE__':fmt(candidate-warm),'__RATIO__':fmt(pc['candidate_to_best_reference_ratio']),'__WITHIN__':str(pc['within2percent_descriptive']),'__PATHNOTE__':path_note,'__PATHTABLE__':path_table,'__CMP__':cmp,'__TABLE__':table,'__LINKS__':' · '.join(links),'__UTC__':summary['actual_utc'],'__SHA__':sha(source)}.items():page=page.replace(k,v)
 (out/'RESULTS_AND_LIMITS_ZH.html').write_text(page,encoding='utf-8')
 files={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(out.iterdir()) if f.is_file()}
 write(out/'REPORT_FILES.json',{'status':'READ_ONLY_REPORT_RENDERED','actual_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'results_sha256':sha(source),'files':files})
 print(json.dumps({'status':'READ_ONLY_REPORT_RENDERED','summary_rows':33,'output':str(out)},ensure_ascii=False))
if __name__=='__main__':main()
