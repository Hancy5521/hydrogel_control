"""Read only whitelisted coordinates/status; never convert response fields."""
import pathlib, zipfile, hashlib, json, datetime, re, collections, math
ROOT=pathlib.Path(__file__).resolve().parent
PHASE=ROOT/'resin_input_phase2_v2'
PHASE.mkdir(exist_ok=True)
ZIP=ROOT/'19699588'/'Rheological_Data_For_Syntactic_Foam.zip'
EXPECTED='a2a7a84d92bf84a69e6f6c49e0ae976d916cc63fa46ff28bccd08a7ebdd4e218'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,d):
 p=PHASE/n
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
script_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
save('PLAN.json',{'actual_utc_before_input_read':now(),'fixed_source_sha256':EXPECTED,'script_sha256':script_hash,
 'allowed_numeric_fields':['Point No. for row detection/address only','Temperature','Shear Rate'],
 'allowed_text_fields':['exact member name as run identity','recipe key from member name','header/units labels','interval occurrence','Status raw token'],
 'prohibited':['Viscosity numeric conversion or statistics','Rotational Speed and other response fields','fit','score','plot raw curves','eta based QC deletion','Y based split or coordinate choice'],
 'proposed_mask_before_eta':'Nominal35C unique block per run: every measured T in [34.5,35.5), a nearest-integer nominal assignment fixed before eta. Also separately report exact35 eligibility. Common sorted gamma axis; last floor(m/3) target fixedtail; first two gamma anchors; four extra queries in other lower-axis coordinates only; never query target tail',
 'status_policy_before_semantics':'No filtering; preserve every original token and report QC. Do not infer status semantics from observed outcome.',
 'fit_budget':0,'eta_read_count':0})
b=ZIP.read_bytes();actual=hashlib.sha256(b).hexdigest()
if actual!=EXPECTED:raise ValueError('source hash changed')
save('SOURCE_CHECK.json',{'utc':now(),'sha256':actual,'expected':EXPECTED,'pass':True,'bytes':len(b)})
z=zipfile.ZipFile(ZIP.open('rb')); runs=[]
NUMBER=re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
for member in z.namelist():
 if not member.lower().endswith('.csv'):continue
 raw=z.read(member);txt=raw.decode('utf-16');run={'member':member,'sha256':hashlib.sha256(raw).hexdigest(),'intervals':[]}
 stem=pathlib.PurePosixPath(member).name
 m=re.fullmatch(r'(0_\d+gcc_\d+%|Neat)_Resin Viscosity\s*(\d*)\.csv',stem)
 if not m:raise ValueError('filename identity not understood '+stem)
 run['recipe']=m.group(1);run['run_suffix_raw']=m.group(2);run['run_id']=int(m.group(2)) if m.group(2) else 0
 current=None;fields=None
 for line_n,line in enumerate(txt.splitlines(),start=1):
  parts=line.split('\t')
  if 'Interval data:' in parts and 'Temperature' in parts and 'Viscosity' in parts and 'Shear Rate' in parts:
   fields={name:parts.index(name) for name in ['Point No.','Temperature','Shear Rate','Status']}
   current={'interval_index_1based':len(run['intervals'])+1,'header_line_1based':line_n,'input_rows':[],'units':[]}
   run['intervals'].append(current);continue
  if current is None or fields is None:continue
  if any(token.startswith('[') for token in parts):
   current['units'].append({'line':line_n,'temperature_unit_raw':parts[fields['Temperature']] if len(parts)>fields['Temperature'] else None,'shear_rate_unit_raw':parts[fields['Shear Rate']] if len(parts)>fields['Shear Rate'] else None})
  if len(parts)<=max(fields.values()):continue
  point=parts[fields['Point No.']].strip();temp=parts[fields['Temperature']].strip();gamma=parts[fields['Shear Rate']].strip()
  if not NUMBER.fullmatch(point):
   if current['input_rows']:current=None;fields=None
   continue
  if not NUMBER.fullmatch(temp) or not NUMBER.fullmatch(gamma):raise ValueError('nonstandard whitelisted numeric input '+member+':'+str(line_n))
  current['input_rows'].append({'source_line_1based':line_n,'point_no_raw':point,'temperature_raw':temp,'temperature_C_numeric_input':float(temp),'gamma_raw':gamma,'gamma_per_s_numeric_input':float(gamma),'status_raw':parts[fields['Status']]})
 for block in run['intervals']:
  rows=block['input_rows'];temps=sorted(set(r['temperature_C_numeric_input'] for r in rows));gammas=sorted(set(r['gamma_per_s_numeric_input'] for r in rows));status=collections.Counter(r['status_raw'] for r in rows)
  block.update({'row_count':len(rows),'temperatures_C':temps,'gamma_count':len(gammas),'gamma_values_per_s':gammas,'status_raw_counts':dict(status),'duplicate_gamma_count':len(rows)-len(gammas)})
 run['exact35_interval_indices']=[v['interval_index_1based'] for v in run['intervals'] if v['temperatures_C']==[35.0]]
 run['nominal35_interval_indices']=[v['interval_index_1based'] for v in run['intervals'] if v['temperatures_C'] and all(34.5<=t<35.5 for t in v['temperatures_C'])]
 run['is35_unique']=len(run['nominal35_interval_indices'])==1
 runs.append(run)
save('WHITELIST_INPUTS.json',{'utc_completed':now(),'source_sha256':EXPECTED,'runs':runs,'response_fields_numeric_converted':0,'eta_statistics':0,'fits':0})
summ=[];axes=[]
for r in runs:
 summ.append({'recipe':r['recipe'],'run_id':r['run_id'],'member':r['member'],'block_count':len(r['intervals']),'exact35_interval_indices':r['exact35_interval_indices'],'nominal35_interval_indices':r['nominal35_interval_indices'],'blocks':[{k:v for k,v in bl.items() if k not in ['input_rows','units']} for bl in r['intervals']]})
 if r['is35_unique']:axes.append(set(r['intervals'][r['nominal35_interval_indices'][0]-1]['gamma_values_per_s']))
common=sorted(set.intersection(*axes)) if axes and len(axes)==len(runs) else []
tail_count=len(common)//3;lower=common[:-tail_count] if tail_count else []
tail=common[-tail_count:] if tail_count else []
anchors=lower[:2];pool=lower[2:]
uniform_idx=[]
if len(pool)>=4:
 # Four spread positions, deterministic endpoints included. No response-based selection.
 uniform_idx=[round(i*(len(pool)-1)/3) for i in range(4)]
uniform_queries=[pool[i] for i in uniform_idx]
fixed_queries=pool[:4]
feasible=len(runs)==50 and len(axes)==50 and all(a==set(common) for a in axes) and len(anchors)==2 and len(pool)>=4 and len(set(uniform_queries))==4 and len(tail)>0
policy={'source_sha256':EXPECTED,'utc_before_any_eta_read':now(),'input_feasible':feasible,'nominal35_rule':'all actual T in [34.5,35.5), input-only nearest integer setpoint; exact equality separately reported; no row deletion','all_50_runs_unique35':len(axes)==50,'all_35_coordinate_axes_identical':len(axes)==50 and all(a==set(common) for a in axes),'common_gamma_count':len(common),'common_gamma_axis':common,
 'tail_rule':'largest floor(m/3) elements of sorted common gamma axis','panelT_gamma':tail,'lower2third_gamma':lower,'first2_anchor_gamma':anchors,'additional_query_pool_gamma':pool,
 'label_budget':{'initial_anchors':2,'additional_queries':4,'total':6},
 'fixedfirst6_additional_gamma':fixed_queries,'uniform4_rule':'four evenly spaced indices round(i*(L-1)/3), i=0..3, over sorted non-anchor lower pool; include endpoints; Python round ties-to-even; here chosen indices explicitly fixed','uniform4_pool_indices_0based':uniform_idx,'uniform4_additional_gamma':uniform_queries,
 'adaptive_policies':'max posterior variance / expected CRPS effect: no posterior and no Y accessed here; query only still-unqueried members of allowed lower pool. Score/effect law and deterministic ties require protocol specification before reading eta. Exact future queries depend only on then-permitted observed prefix labels and frozen model.',
 'grouping':'10 recipes x 5 exact-filename runs. All temperature blocks from same run linked; no train/test split by points. Macro per recipe after equal run averaging. No material fit in transfer mode.',
 'units':'Viscosity cP -> Pa.s by division by 1000, not executed; gamma 1/s; T Celsius by method plus file label',
 'status':'All raw status tokens kept; no exclusions made; semantics needs authoritative mapping. Full-range all-token QC reporting permitted without inventing deletion rules.',
 'fits':0,'eta_read_count':0,'eta_statistics':0}
save('FROZEN_INPUT_MASKS.json',policy)
save('RUN_BLOCK_SUMMARY.json',{'utc':now(),'runs':summ})
print(json.dumps({'input_policy':policy,'status_tokens_all_blocks':dict(collections.Counter(v['status_raw'] for r in runs for bl in r['intervals'] for v in bl['input_rows'])),'block_counts':dict(collections.Counter(len(r['intervals']) for r in runs)),'temperature_sets_counts':dict(collections.Counter(str(tuple(tuple(bl['temperatures_C']) for bl in r['intervals'])) for r in runs)),'first_run_blocks':summ[0]},ensure_ascii=False,indent=2))
