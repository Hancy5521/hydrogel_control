"""Frozen phase1 descriptive diagnosis only; no gzip, source model or refit imports."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,collections,csv,hashlib,json,math,statistics,traceback

MODELS=('niw_joint_t','gaussian_momentmatched','independent_target_marginal')
MASKS=('none','e_form','gap_opt','both')
LEVELS=('0.8','0.9','0.95')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def put(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def require(x,m):
    if not x:raise AssertionError(m)
def avg(x):return math.fsum(x)/len(x) if x else None
def descriptive(x):
    if not x:return {'n':0,'mean':None,'median':None,'sample_SD':None,'skewness_population_moment':None,'negative_count':0,'zero_count':0,'minimum':None,'maximum':None}
    require(all(math.isfinite(v) for v in x),'Nonfinite frozen value')
    mu=avg(x);m2=avg([(v-mu)**2 for v in x]);m3=avg([(v-mu)**3 for v in x])
    return {'n':len(x),'mean':mu,'median':statistics.median(x),'sample_SD':math.sqrt(math.fsum((v-mu)**2 for v in x)/(len(x)-1)) if len(x)>1 else None,'skewness_population_moment':m3/m2**1.5 if m2 else None,'negative_count':sum(v<0 for v in x),'zero_count':sum(v==0 for v in x),'minimum':min(x),'maximum':max(x)}

def pit_shape(rows):
    if not rows:return {'n':0,'counts_ten_bins':[0]*10,'mean':None,'median':None,'max_ECDF_deviation':None,'max_deviation_locations':[]}
    ordered=sorted(rows,key=lambda r:(r['score']['PIT'],r['component_id']));n=len(ordered);bins=[0]*10;locations=[]
    for i,r in enumerate(ordered,1):
        u=r['score']['PIT'];require(0<=u<=1,'Saved PIT outside0..1');bins[min(9,int(u*10))]+=1
        for side,d in [('ECDF_upper_minus_PIT',i/n-u),('PIT_minus_ECDF_lower',u-(i-1)/n)]:
            locations.append({'component_id':r['component_id'],'jid':r['jid'],'PIT':u,'rank_one_based':i,'ECDF_lower':(i-1)/n,'ECDF_upper':i/n,'side':side,'deviation':d})
    peak=max(x['deviation'] for x in locations)
    return {'n':n,'bin_rule':'[j/10,(j+1)/10),lastincludes1;no jitter','counts_ten_bins':bins,'proportions_ten_bins':[v/n for v in bins],'mean':avg([r['score']['PIT'] for r in rows]),'median':statistics.median(r['score']['PIT'] for r in rows),'zero_PIT_count':sum(r['score']['PIT']==0 for r in rows),'one_PIT_count':sum(r['score']['PIT']==1 for r in rows),'max_ECDF_deviation':peak,'max_deviation_locations':[x for x in locations if abs(x['deviation']-peak)<=1e-15],'not_IID_KS_test_or_calibration_guarantee':True}

def score_summary(rows):
    return {'groups':len(rows),'equal_group_mean_CRPS_native':avg([r['score']['CRPS_native'] for r in rows]),'RMSE_native':math.sqrt(avg([r['score']['squared_error_native'] for r in rows])) if rows else None,'coverage':{l:avg([int(r['score']['intervals'][l]['covered']) for r in rows]) for l in LEVELS},'mean_interval_width_native':{l:avg([r['score']['intervals'][l]['width'] for r in rows]) for l in LEVELS},'PIT':pit_shape(rows)}

def signs(values):
    return {'n':len(values),'wins_positive':sum(x>0 for x in values),'losses_negative':sum(x<0 for x in values),'ties_exact_saved_float_zero':sum(x==0 for x in values),'mean_gain_native_CRPS':avg(values),'median_gain_native_CRPS':statistics.median(values) if values else None,'no_tolerance_or_extreme_deletion':True}

def main():
    a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--approval',required=True);a.add_argument('--output-dir',required=True);args=a.parse_args()
    plan=read(args.plan);approval=read(args.approval);out=Path(args.output_dir)
    require(plan['status']=='REGISTERED' and plan['source_sha256']==sha(__file__),'Actual source-pinned plan required')
    require(approval.get('approved') is True and approval['scope']=='FROZEN_PHASE1_POSTHOC_DIAGNOSIS_ONLY','Root diagnosis-only approval required')
    require(approval['plan_sha256']==sha(args.plan) and approval['source_sha256']==sha(__file__),'Approval binding differs')
    require(set(plan['inputs'])=={'VISIBLE_INPUTS','MODEL','FORECASTS','DIAGNOSTIC_TRUTH','RESULTS'},'Exactly5frozen inputs allowed')
    for item in plan['inputs'].values():require(not item['file'].lower().endswith(('.gz','.npz','.npy')) and sha(item['file'])==item['sha256'],'Frozen JSON input changed')
    require(not out.exists(),'No repeat or overwriting diagnosis');out.mkdir(parents=True)
    start={'actual_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),'plan_sha256':sha(args.plan),'approval_sha256':sha(args.approval),'inputs':plan['inputs'],'posthoc_descriptive_only':True,'gzip_or_new_group_access':False}
    put(out/'EXECUTION_START.json',start)
    try:
        data={k:read(v['file']) for k,v in plan['inputs'].items()};visible=data['VISIBLE_INPUTS'];fitted=data['MODEL'];result=data['RESULTS']
        require(result['admission']['status']=='NO_GO_STOP_FULL_CAMPAIGN_NO_RETUNE','Preservedphase1NO_GO required')
        require(fitted['fit_rows']==64 and fitted['diagnostic_rows_used_in_fit']==0,'Unexpected fit scope')
        fit=visible['fit'];diag=visible['diagnostic'];truth=data['DIAGNOSTIC_TRUTH']['truth'];forecasts=data['FORECASTS']['forecasts'];scores=result['per_forecast_scores']
        require(len(fit)==64 and len(diag)==len(truth)==192 and len(forecasts)==len(scores)==2304,'Frozen cohort shape differs')
        fitgroups={x['identity']['component_id'] for x in fit};dg={x['identity']['component_id']:x for x in diag};tg={x['identity']['component_id']:x for x in truth}
        require(len(fitgroups)==64 and len(dg)==len(tg)==192 and set(dg)==set(tg) and not fitgroups&set(dg),'Group identity scope differs')
        keys={(g,m,k) for g in dg for m in MODELS for k in MASKS};pred={(x['component_id'],x['model'],x['mask']):x for x in forecasts};sc={(x['component_id'],x['model'],x['mask']):x for x in scores}
        require(set(pred)==set(sc)==keys,'Incomplete frozen model/mask/group map')
        for key,r in sc.items():
            require(r['native_target']==tg[key[0]]['native_target'] and r['jid']==dg[key[0]]['identity']['jid']==pred[key]['jid'],'Frozen target orJID mismatch')
        summaries={m:{k:score_summary([sc[(g,m,k)] for g in sorted(dg)]) for k in MASKS} for m in MODELS}
        fit_bulk=[r['native_values'][2] for r in fit];diag_bulk=[tg[g]['native_target'] for g in sorted(dg)]
        overall={'fit64_bulk':descriptive(fit_bulk),'diagnostic192_bulk':descriptive(diag_bulk),'frozen_model_target_affine_center':fitted['affine_center'][2],'frozen_model_target_affine_sample_SD':fitted['affine_scale_sample_SD'][2]}
        strata={};per_group=[]
        conditions={'gap_exact_zero':lambda v:v==0,'gap_positive':lambda v:v>0,'gap_negative':lambda v:v<0}
        for name,condition in conditions.items():
            ids=sorted(g for g in dg if condition(dg[g]['native_proxies'][1]));fb=[r['native_values'][2] for r in fit if condition(r['native_values'][1])]
            strata[name]={'definition':'nativegapopt '+{'gap_exact_zero':'==0','gap_positive':'>0','gap_negative':'<0'}[name],
                'fit_bulk':descriptive(fb),'diagnostic_bulk':descriptive([tg[g]['native_target'] for g in ids]),
                'all_12_saved_model_mask_metrics':{m:{k:score_summary([sc[(g,m,k)] for g in ids]) for k in MASKS} for m in MODELS},
                'posthoc_descriptive_not_newvalidation_or_selected_submodel':True}
        wins={m:{k:signs([sc[(g,m,'none')]['score']['CRPS_native']-sc[(g,m,k)]['score']['CRPS_native'] for g in sorted(dg)]) for k in MASKS} for m in MODELS}
        t_vs_g={k:signs([sc[(g,'gaussian_momentmatched',k)]['score']['CRPS_native']-sc[(g,'niw_joint_t',k)]['score']['CRPS_native'] for g in sorted(dg)]) for k in MASKS}
        for g in sorted(dg):
            gap=dg[g]['native_proxies'][1];row={'component_id':g,'jid':dg[g]['identity']['jid'],'native_gap_opt':gap,'native_bulk_target':tg[g]['native_target'],'gap_class':next(n for n,f in conditions.items() if f(gap))}
            for m in MODELS:
                for k in MASKS:
                    row[m+'__'+k+'__conditional_gain_vs_own_none']=sc[(g,m,'none')]['score']['CRPS_native']-sc[(g,m,k)]['score']['CRPS_native']
            per_group.append(row)
        with (out/'ALL_GROUP_CONDITIONAL_GAINS.csv').open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(per_group[0]));w.writeheader();w.writerows(per_group)
        diagnosis={'actual_utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETE_POSTHOC_PHASE1_NO_GO_DIAGNOSIS','preserved_original_admission':result['admission'],'all12_model_mask_summaries':summaries,'bulk_fit_vs_diagnostic':overall,'predefined_gap_strata':strata,'per_group_conditional_gain_signs_vs_own_none':wins,'paired_NIW_gain_vs_matched_Gaussian':t_vs_g,'interpretation_limits':['No refit/rescore/PIT recomputation from probability laws; only frozen individual scores/intervals used.','All192diagnosticgroups remain consumed development; subgroup descriptions are not independent validation.','No thresholds/prior/transform/role/seed changed;phase1fullcampaignNO_GO stands.','Gapstrata fixed to exact0/positive/negative; no Yselection, clipping or deletion of extremevalues.','Statistical association can suggest a newhypothesis, not prove itscause or validate a newmodel. Any futuretransform needs separate finite-original-unit-moment justification/newprotocol.'],'new_gzip_groups_models_fits_or_score_law_calls':0}
        put(out/'DIAGNOSIS.json',diagnosis)
        for item in plan['inputs'].values():require(sha(item['file'])==item['sha256'],'Frozeninput mutated during diagnosis')
        lines=['phase1 NO_GO：冻结数据的事后描述诊断','', '原门限与失败状态完全保留；没有新拟合、概率评分或新材料。',
            'fit64 bulk：'+json.dumps(overall['fit64_bulk'],ensure_ascii=False),
            'diagnostic192 bulk：'+json.dumps(overall['diagnostic192_bulk'],ensure_ascii=False),
            '', 'model | mask | CRPS | 90coverage | PITmean/median | PITD']
        for m in MODELS:
            for k in MASKS:
                v=summaries[m][k];lines.append('%s | %s | %.8g | %.6f | %.6f / %.6f | %.6f'%(m,k,v['equal_group_mean_CRPS_native'],v['coverage']['0.9'],v['PIT']['mean'],v['PIT']['median'],v['PIT']['max_ECDF_deviation']))
        lines+=['','所有12组合的80/90/95coverage、宽度、PIT十bin及最大偏差位置见DIAGNOSIS.json。',
                'gap==0、gap>0、gap<0（即使为空也保留）分别给fit/diag bulk与全部12组合描述，未按Y筛选。所有组收益与正负胜负保留于JSON和CSV。',
                '这些只解释当前失败可能的统计表现，不恢复准入资格，不证明分层或非线性新模型有效；当前256组与siblings不能变成未来独立确认。']
        (out/'DIAGNOSIS_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        done={'status':'COMPLETE_POSTHOC_DIAGNOSIS_NO_GO_UNCHANGED','actual_utc':datetime.now(timezone.utc).isoformat(),'execution':start,'files_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},'refits_probability_scores_or_new_groups':0}
        put(out/'DIAGNOSIS_COMPLETE.json',done);print(json.dumps({'status':done['status'],'receipt_sha256':sha(out/'DIAGNOSIS_COMPLETE.json')},ensure_ascii=True))
    except Exception:
        put(out/'FAILED_STOP.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),'status':'FAIL_PRESERVED_NO_AUTOMATIC_RETRY','traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
