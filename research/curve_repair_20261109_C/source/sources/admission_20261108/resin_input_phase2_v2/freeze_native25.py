import pathlib,json,datetime,hashlib,collections
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'NATIVE_RANK25_INPUT_MASKS.json'
if OUT.exists():raise FileExistsError(OUT)
d=json.loads((ROOT/'WHITELIST_INPUTS.json').read_text(encoding='utf-8'));runs=d['runs'];records=[];feasible=True
for run in runs:
 bl=run['intervals'][run['nominal35_interval_indices'][0]-1]
 rows=sorted(bl['input_rows'],key=lambda x:x['gamma_per_s_numeric_input']);n=len(rows);k=n//3;nt=n-k
 nominal=[round(sum(b['temperatures_C'])/len(b['temperatures_C'])) for b in run['intervals']]
 history='ascending_35_first' if nominal==list(range(35,126,10)) else 'descending_35_last' if nominal==list(range(125,34,-10)) else 'unresolved'
 ranks=list(range(1,n+1));T=ranks[-k:];low=ranks[:-k];anchors=low[:2];pool=low[2:];ix=[round(i*(len(pool)-1)/3) for i in range(4)];fixed=pool[:4];uniform=[pool[i] for i in ix]
 feasible=feasible and n==25 and len(set(r['gamma_per_s_numeric_input'] for r in rows))==25 and history!='unresolved' and len(run['nominal35_interval_indices'])==1
 records.append({'recipe':run['recipe'],'run_id':run['run_id'],'member':run['member'],'source_member_sha256':run['sha256'],'nominal35_interval_index_1based':bl['interval_index_1based'],'thermal_history':history,'nominal_temperature_sequence_C':nominal,'actual35_temperature_min_C':min(bl['temperatures_C']),'actual35_temperature_max_C':max(bl['temperatures_C']),
  'native_gamma_sorted_per_s':[r['gamma_per_s_numeric_input'] for r in rows],'native_gamma_raw_tokens_sorted':[r['gamma_raw'] for r in rows],'source_line_by_rank_1based':[r['source_line_1based'] for r in rows],'status_raw_counts':bl['status_raw_counts'],
  'panelT_ranks_1based':T,'panelT_source_lines':[rows[i-1]['source_line_1based'] for i in T],'anchor_ranks_1based':anchors,'anchor_source_lines':[rows[i-1]['source_line_1based'] for i in anchors],'extra_query_candidate_ranks_1based':pool,'fixedfirst6_extra_ranks_1based':fixed,'uniform4_extra_ranks_1based':uniform})
recipe_summary=[]
for recipe in sorted({r['recipe'] for r in records}):
 rr=[r for r in records if r['recipe']==recipe]
 recipe_summary.append({'recipe':recipe,'run_count':len(rr),'run_ids':sorted(r['run_id'] for r in rr),'history_counts':dict(collections.Counter(r['thermal_history'] for r in rr)),'35_interval_counts':dict(collections.Counter(str(r['nominal35_interval_index_1based']) for r in rr)),'actual35_T_range_C':[min(r['actual35_temperature_min_C'] for r in rr),max(r['actual35_temperature_max_C'] for r in rr)]})
o={'actual_utc_before_any_eta_read':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metadata_only_not_final_adopted_protocol':True,'source_zip_sha256':d['source_sha256'],'whitelist_inputs_sha256':hashlib.sha256((ROOT/'WHITELIST_INPUTS.json').read_bytes()).hexdigest(),'input_feasible':feasible and len(records)==50 and len(recipe_summary)==10,
 'scheme':'per-curve native gamma ascending rank; no common-grid intersection, rounding or interpolation','all_methods_same_curve_same_targets':True,'nominal35_rule':'each measured T in [34.5,35.5), nearest-integer block assignment; retain each actual T; no within-block row filtering','target_panelT_ranks_1based':list(range(18,26)),'anchor_ranks_1based':[1,2],'extra_query_candidate_ranks_1based':list(range(3,18)),
 'fixedfirst6_extra_ranks_1based':[3,4,5,6],'uniform4_extra_ranks_1based':[3,8,12,17],'uniform4_rule':'4 evenly spaced ordinal positions of sorted permitted pool ranks3..17; indices round(i*14/3), i=0..3 -> [0,5,9,14]',
 'max_variance_and_expected_CRPS':'only choose four still-unqueried ranks3..17; model/objective/tie/integration rules must be frozen before eta. No computation in this metadata phase. Expected CRPS effect must not inspect panelT labels.',
 'labels_per_run':6,'total_budget_50runs_per_method_labels':300,'grouping':'recipe10 x exact-filename run5, paired-method within curve; macro recipe mean of 5 run effects; all temperature blocks linked; full50 and predefined ascending/descending strata; no same-state equality or causal recipe/history claim',
 'status_policy':'all Dy_auto retained; semantics/QC unknown; no removal rule inferred','eta_unit_conversion':'cP/1000 -> Pa.s; eta not read/converted','reference_support':'parent reports frozen reference input support 0.0128..754.5 s^-1, source coordinates checked below; no model inspected/fit here','native35_gamma_global_range':[min(min(r['native_gamma_sorted_per_s']) for r in records),max(max(r['native_gamma_sorted_per_s']) for r in records)],'in_parent_reported_reference_support':all(0.0128<=g<=754.5 for r in records for g in r['native_gamma_sorted_per_s']),
 'recipe_history_summary':recipe_summary,'per_run':records,'eta_numeric_read_count':0,'eta_statistics':0,'fits':0,'scores':0}
OUT.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in o.items() if k!='per_run'},ensure_ascii=False,indent=2))
