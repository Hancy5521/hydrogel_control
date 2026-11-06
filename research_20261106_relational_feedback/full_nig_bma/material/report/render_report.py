"""Render accepted frozen results only; no scientific model import/query/score/fit."""
import argparse,csv,hashlib,html,json,math,os
from datetime import datetime,timezone
from pathlib import Path
POLICIES=('full_native','warm_openloop','random','marginal_native','model_MI','gp_variance')
H=('affine','interaction','fullquad');B=(0,4,8)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvwrite(p,rows):
 with p.open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 a=argparse.ArgumentParser(description=__doc__)
 for n in ('results','forecast-dir','independent-score-audit','independent-score-plan','output-dir'):a.add_argument('--'+n,required=True)
 a=a.parse_args();stage=Path(__file__).resolve().parents[1];resultfile=Path(a.results).resolve();forecast=Path(a.forecast_dir).resolve();out=Path(a.output_dir).resolve();out.relative_to(stage)
 auditpath=Path(a.independent_score_audit).resolve();auditplanpath=Path(a.independent_score_plan).resolve();audit=read(auditpath);auditplan=read(auditplanpath)
 assert str(audit['status']).startswith('PASS') and audit['plan_sha256']==sha(auditplanpath)
 assert any(Path(x['file']).resolve()==resultfile and x['sha256']==sha(resultfile) for x in auditplan['inputs'].values()),'Independent plan must pin actual completed results'
 r=read(resultfile);assert r['status']=='COMPLETE_FRESH160_RELATIONAL_FEEDBACK_DEVELOPMENT'
 assert r['roles']=={'warm':32,'shared_query_pool':32,'terminal_development':96}
 assert [len(r[k]) for k in ('main_forecast_scores','reference_forecast_scores','secondary_prequery_scores')]==[1728,2304,48]
 planpath=stage/'EXECUTION_PLAN_REGISTERED.json';assert sha(planpath)==r['plan_sha256'];plan=read(planpath)
 assert r['source_sha256']==plan['source_sha256'] and all(sha(stage/'code'/n)==h for n,h in r['source_sha256'].items())
 freeze=read(forecast/'FORECAST_FREEZE.json');assert sha(forecast/'FORECAST_FREEZE.json')==r['forecast_freeze_sha256']
 assert freeze['status']=='ALL4032_TERMINAL_AND48_PREQUERY_MIXTURE_LAWS_FROZEN' and len(freeze['files'])==123
 assert freeze['source_sha256']==r['source_sha256'] and freeze['plan_sha256']==r['plan_sha256']
 for name,pin in freeze['files'].items():file=(forecast/name).resolve();file.relative_to(forecast);assert sha(file)==pin
 summary=r['summary_BMA_prediction_by_policy'];refs=r['summary_sameinformation_references'];assert set(summary)==set(POLICIES)
 rows=[]
 def add(policy,method,n,v):
  assert v['formula_groups']==96
  rows.append({'policy':policy,'predictor':method,'newY_cost':n,'common_learning_cells':352,'terminal_validationY_separate':96,'groups':96,
   'mean_CRPS_native':v['mean_CRPS_native'],'RMSE_predictive_median_native':v['RMSE_predictive_median_native'],
   'coverage80':v['coverage']['0.8'],'coverage90':v['coverage']['0.9'],'coverage95':v['coverage']['0.95'],
   'mean_width80_native':v['mean_interval_width_native']['0.8'],'mean_width90_native':v['mean_interval_width_native']['0.9'],'mean_width95_native':v['mean_interval_width_native']['0.95']})
 for policy in POLICIES:
  for n in B:add(policy,'BMA',n,summary[policy][str(n)])
 assert set(refs)=={'full_native','random','gp_variance'}
 for policy,models in refs.items():
  assert set(models)==({'GP'} if policy=='gp_variance' else {'GP','RF','affine_NIG','frozen_weights_BMA'} if policy=='full_native' else {'GP','RF','affine_NIG'})
  for method,bs in models.items():
   for n in B:add(policy,method,n,bs[str(n)])
 assert len(rows)==42
 sequences=r['actual_selected_pool_sequences'];assert set(sequences)==set(POLICIES)
 assert all(len(v)==len(set(v))==8 and all(type(i)is int and 0<=i<32 for i in v) for v in sequences.values())
 states=read(forecast/'MODEL_STATES.json');weightrows=[]
 def weightrows_add(label,ensemble):
  assert [x['id'] for x in ensemble['models']]==list(H)
  logs=ensemble['log_weights'];assert len(logs)==3 and all(math.isfinite(x) for x in logs)
  for h,log in zip(H,logs):weightrows.append({'state':label,'hypothesis':h,'log_weight':log,'weight_display_exp_log':math.exp(log),'display_underflow_possible_no_floor':True})
 weightrows_add('warm',states['warm_ensemble'])
 for policy in POLICIES:
  ensemble=next(x['ensemble'] for x in states['checkpoint_states'] if x['policy']==policy and x['budget']==8);weightrows_add(policy,ensemble)
 ledger=read(forecast/'QUERY_LEDGER.json');assert len(ledger['events'])==48 and all(x==8 for x in ledger['cost_per_policy'].values())
 assert ledger['common_setup_native_cells']==352 and ledger['terminalY_model_access']==0
 candidate=summary['full_native']['8']['mean_CRPS_native'];warm=summary['full_native']['0']['mean_CRPS_native']
 path_same=sequences['full_native']==sequences['warm_openloop'];set_same=set(sequences['full_native'])==set(sequences['warm_openloop'])
 if path_same:path_note='本轮full_native与warm_openloop的实际8次序列相同：存在真Y后验更新，仍不能宣称标签反馈带来额外动作选择。'
 elif set_same:path_note='full_native与warm_openloop顺序不同，但最终集合相同。顺序差本身不保证预算8收益，必须结合检查点评分和机器精度差别解释。'
 else:path_note='full_native与warm_openloop实际选择集合不同；这只建立策略差别，效用方向仍以同预算终端评分为准，不能仅凭路径变化声称机制有效。'
 comparison=r['primary_comparison'];paired=r['paired_final_summaries_descriptive']
 assert len(paired)==13 and all(x['groups']==96 for x in paired.values())
 aggregate={'status':'READ_ONLY_ACCEPTED_FRESH160_RESULT_REPORT','actual_report_utc':datetime.now(timezone.utc).isoformat(),
  'results_sha256':sha(resultfile),'forecast_freeze_sha256':sha(forecast/'FORECAST_FREEZE.json'),'registered_plan_sha256':sha(planpath),
  'independent_score_audit_sha256':sha(auditpath),'independent_score_plan_sha256':sha(auditplanpath),'report_script_sha256':sha(__file__),
  'all42summaryrows':rows,'posterior_modelweight_rows':weightrows,'actual_sequences':sequences,
  'candidate_openloop_same_sequence':path_same,'candidate_openloop_same_set':set_same,
  'primary_comparison':comparison,'candidate_final_minus_warm':candidate-warm,'paired_summaries_descriptive_96groups_onecampaign':paired,
  'counts':{'material_components_development':160,'warm':32,'shared_pool':32,'terminal':96,'logical_terminal_coordinates':4032,'process_coordinates':48,'sklearn_fits':18,'GPfits':13,'RFfits':5,'warm_NIG_initializations':3,'warm_response_uses':96,'feedback_component_NIG_updates':144},
  'cost':{'common_learning_cells':352,'perpolicy_newY':8,'validationY_separate':96,'singlepolicy_including_validation':456,'counterfactual_events':48,'unique_pool_Y_components':ledger['union_pool_components_with_Y_extracted']},
  'native_model_or_scoring_execution_by_report':False,'limits':['All160aredevelopment; samepoolreused, not independentconfirmation.','SEisdescriptive96targetgroupsconditionalononecampaign, notcampaignorformalinference.','Hypothesisweightchangesnotphysicalrelationidentification; classicalBMA/MI/properdesignpriorart.','Noactualwet/DFTcostsaving orcalibrationguarantee; allfailureandstrongcontrolsretained.']}
 out.mkdir(parents=True,exist_ok=False);csvwrite(out/'ALL42_SUMMARIES.csv',rows);csvwrite(out/'MODEL_WEIGHTS.csv',weightrows);write(out/'REPORT_AGGREGATE.json',aggregate)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 colors=('#1261a0','#a64c4c','#777777','#26856b','#e78524','#a0509e')
 fig,axes=plt.subplots(1,2,figsize=(16,6.5),constrained_layout=True)
 for policy,color in zip(POLICIES,colors):axes[0].plot(B,[summary[policy][str(n)]['mean_CRPS_native'] for n in B],marker='o',label=policy,color=color)
 axes[0].set_title('Query policies, common actual-updated BMA predictor')
 axes[1].plot(B,[summary['full_native'][str(n)]['mean_CRPS_native'] for n in B],marker='o',linewidth=2.3,label='full_native / BMA',color=colors[0])
 for policy,models in refs.items():
  for method,bs in models.items():axes[1].plot(B,[bs[str(n)]['mean_CRPS_native'] for n in B],marker='s',linestyle='--',label=policy+' / '+method)
 axes[1].set_title('Declared same-information and own-GP references')
 for ax in axes:
  ax.set_xticks(B);ax.set_xlabel('Only-Y cost per policy; common learning setup=352');ax.set_ylabel('Mean native CRPS (lower is better)');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
 fig.suptitle('Fresh160 development components: 32warm /32shared pool /96terminal; no independent confirmation',fontsize=11)
 fig.savefig(out/'learning_curves.png',dpi=180,facecolor='white');plt.close(fig)
 labels=['warm',*POLICIES];fig,ax=plt.subplots(figsize=(13,5.7),constrained_layout=True);bottom=[0.]*len(labels)
 for h,color in zip(H,('#1261a0','#e78524','#26856b')):
  values=[next(x['weight_display_exp_log'] for x in weightrows if x['state']==label and x['hypothesis']==h) for label in labels]
  ax.bar(labels,values,bottom=bottom,label=h,color=color);bottom=[a+b for a,b in zip(bottom,values)]
 ax.set_ylim(0,1.05);ax.set_ylabel('Posterior model probability (display only; no floor)');ax.tick_params(axis='x',rotation=20);ax.legend(loc='upper right');ax.set_title('Warm evidence and final-budget model weights; concentration is an allowed result')
 fig.savefig(out/'model_weights.png',dpi=180,facecolor='white');plt.close(fig)
 def fmt(x):return '\u2014' if x is None else f'{x:.6f}'
 table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (x['policy'],x['predictor'],x['newY_cost'],fmt(x['mean_CRPS_native']),fmt(x['RMSE_predictive_median_native']),f"{round(x['coverage90']*96)}/96 ({x['coverage90']:.1%})",fmt(x['mean_width90_native'])))+'</tr>' for x in rows)
 pairs=''.join('<tr><td>'+html.escape(k)+'</td>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (fmt(x['mean_signed_CRPS_difference']),fmt(x['standard_error_descriptive']),x['candidate_better_groups'],x['candidate_worse_groups'],x['exact_score_ties']))+'</tr>' for k,x in paired.items())
 paths=''.join('<tr><td>'+html.escape(p)+'</td><td>'+html.escape(str(sequences[p]))+'</td></tr>' for p in POLICIES)
 weights=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (x['state'],x['hypothesis'],f"{x['weight_display_exp_log']:.9g}",f"{x['log_weight']:.9g}"))+'</tr>' for x in weightrows)
 links=[]
 for label,path in [('实际注册协议',planpath),('源与数学证据冻结',stage/'SOURCE_FREEZE.json'),('源前IO对接',stage/'FINAL_CROSS_OWNER_IO_CLEARANCE.json'),('独立评分审计',auditpath),('冻结预测',forecast/'FORECAST_FREEZE.json'),('实际结果',resultfile)]:links.append('<a href="'+html.escape(os.path.relpath(path,out).replace('\\','/'),quote=True)+'">'+label+'</a>')
 page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新160材料：条件关联反馈结果</title><style>body{font:16px/1.7 'Microsoft YaHei',sans-serif;max-width:1300px;color:#203040;margin:30px auto;padding:0 22px}h1{font-size:29px}h2{font-size:22px;margin-top:30px}.note{padding:14px 18px;background:#eff4f8;border-left:5px solid #1261a0}.warn{padding:14px 18px;background:#fff3e0}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:7px 9px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f6}.scroll{overflow:auto}a{color:#1261a0}code{font-size:12px;word-break:break-all}</style><h1>新160材料：条件关联假设反馈与缺失性质恢复</h1>
<div class="note">32warm、32共用池、96终端，先排除旧928完整组件。六策略各8次only-Y，学习共同setup352cell，终端验证Y另计96；单策略含验证总披露456cell。4032终端预测坐标＋48过程坐标不是4080份材料；48个反事实查询来自最多32个池组件。全部160为开发角色。</div>
<h2>同预算实际效果</h2><p>full_native预算8原生CRPS <b>__LOSS__</b>，从warm起点变化 <b>__CHANGE__</b>。候选/最强已列参照比 <b>__RATIO__</b>，预设2%描述带：<b>__WITHIN__</b>。差值负为候选更好，不据此声称统计显著、正式非劣效或普适泛化。</p><img src="learning_curves.png" alt="六策略与全部同信息参照真实学习曲线">
<h2>动作路径与关系权重必须分别看</h2><div class="warn">__PATHNOTE__</div><table><tr><th>查询策略</th><th>池索引顺序</th></tr>__PATHS__</table><img src="model_weights.png" alt="warm与六策略最终假设权重"><p>模型先验等概率，warm权重由实际条件边际证据决定；未用floor、温度或保留熵强行维持不确定性。日志权重始终保留，显示概率可能机器下溢。权重改变是诊断，不能等同识别真实材料机制。frozen_weights_BMA只在同一候选查询链冻结warm预测权重；不是另一个非Bayes选点算法。</p><div class="scroll"><table><tr><th>状态</th><th>假设</th><th>概率（显示）</th><th>log权重</th></tr>__WEIGHTS__</table></div>
<h2>预设的配对效果</h2><p>以下均为同96终端组的预算8候选减参照；SE是单次campaign条件下组间差异的描述量，不覆盖查询campaign变化，不是置信声明。正负结果全部保留。</p><div class="scroll"><table><tr><th>参照</th><th>平均CRPS差</th><th>描述SE</th><th>候选更好组</th><th>更差组</th><th>精确评分tie</th></tr>__PAIRS__</table></div>
<h2>全部42行比较与区间</h2><p>六策略统一用真Y更新后的BMA预测；full_native/random各用同一真训练集比较affine_NIG、固定GP和RF。GP方差策略保留自己的GP预测。候选再与同component状态、固定warm模型权重预测比较。18次sklearn拟合=13GP+5RF；3warmNIG初始化、96warm响应使用及144反馈分量更新不算额外材料。90%覆盖与宽度独立列出，CRPS接近强参照不等于校准已成立。native单位沿发布bulk字段GPa归属，仍受来源聚合语义限制。</p><div class="scroll"><table><tr><th>策略</th><th>预测器</th><th>新Y成本</th><th>原生CRPS</th><th>中位数RMSE</th><th>90%覆盖</th><th>90%宽度</th></tr>__TABLE__</table></div>
<h2>方法、创新边界与保留结论</h2><p>关联形式候选只有affine、interaction、fullquad。每次先用旧predictive density更新模型权重，再用同一真实Y更新全部NIG分量。完整跨目标VOI保留共享H及H内条件依赖，按原生单位proper score评价。对照包括标签盲开放环、随机、逐点原生CRPS不确定性、经典模型辨识MI和GP方差；开放环的一步lookahead仍从合法joint做Bayes条件化。</p><div class="warn">BMA、NIG、proper-score主动学习、MI与自适应设计都有直接先例；本轮可能的贡献在材料缺失目标、报告身份/成本和可证伪关系反馈，不宣称新统计原理、Eigen超循环或LLM智能体已实现。已发布计算标签不是新湿实验/新DFT；控制隐藏完整案例Y不是MNAR识别。模型内期望风险非负不保证实际每一步更好。不得在这96个终端标签上改先验、选假设、重切或校准救分；不把反事实路径、重抽样与合成积分样本当新测量。</div>
<h2>复核与资源入口</h2><p>__LINKS__</p><p><a href="ALL42_SUMMARIES.csv">完整摘要CSV</a> · <a href="MODEL_WEIGHTS.csv">全部权重与log权重</a> · <a href="REPORT_AGGREGATE.json">聚合结果与SHA</a> · <a href="REPORT_FILES.json">文件清单</a></p><p>只读聚合/绘图，没有重新计算nativeCDF评分、拟合、query或bootstrap。报告UTC __UTC__；RESULTS SHA <code>__SHA__</code>。</p></html>"""
 for k,v in {'__LOSS__':fmt(candidate),'__CHANGE__':fmt(candidate-warm),'__RATIO__':fmt(comparison['candidate_to_best_reference_ratio']),'__WITHIN__':str(comparison['within2percent_descriptive_only']),'__PATHNOTE__':path_note,'__PATHS__':paths,'__WEIGHTS__':weights,'__PAIRS__':pairs,'__TABLE__':table,'__LINKS__':' · '.join(links),'__UTC__':aggregate['actual_report_utc'],'__SHA__':sha(resultfile)}.items():page=page.replace(k,v)
 (out/'RESULTS_AND_LIMITS_ZH.html').write_text(page,encoding='utf-8');files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()};write(out/'REPORT_FILES.json',{'status':'READ_ONLY_RESULT_REPORT_RENDERED','actual_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'result_sha256':sha(resultfile),'files':files});print(json.dumps({'status':'READ_ONLY_RESULT_REPORT_RENDERED','output':str(out),'summary_rows':42,'modelweight_rows':21},ensure_ascii=False))
if __name__=='__main__':main()
