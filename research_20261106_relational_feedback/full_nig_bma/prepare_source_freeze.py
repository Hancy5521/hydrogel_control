from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,ast
p=Path(r'H:\polymer_completion_public\research\relational_feedback_20261106T061957384774Z\full_nig_bma')
source={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((p/'code').glob('*.py'))}
for f in (p/'code').glob('*.py'):ast.parse(f.read_text('utf-8-sig'))
proposal=p/'FIXED_DATAFREE_GATE_PROPOSAL.json';proposal_sha=hashlib.sha256(proposal.read_bytes()).hexdigest();spec=json.loads(proposal.read_text('utf-8'))
freeze={'status':'SOURCE_FROZEN_AST_ONLY_NO_NUMERICAL_IMPORT_OR_EXECUTION','actual_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source,'proposal_sha256':proposal_sha,'scientific_source_modules':2,'thin_synthetic_driver':1,'material_label_reads_or_fits':0,'single_fixture_no_parameter_scan':True,'original_prior_stage_and_archives_unchanged':True}
plan={'status':'TEMPLATE_NOT_AUTHORIZATION','approved_for_execution':False,'actual_registered_utc':None,'scope':'ONE_FIXED_SYNTHETIC_FULL_NIG_BMA_FIXTURE_NO_MATERIAL','proposal_sha256':proposal_sha,'source_sha256':source,'criteria':spec['criteria'],'levels':spec['quadrature_levels'],'Lv':32,'Lz':64,'notes':['Root must approve actual UTC and source/fixture hashes before the unique execution.','The raw signed between term is diagnostic and is not clipped.','Main full-mixture information value includes within-H parameter learning.','Independent reference is separate; producer PASS does not imply reference PASS.']}
for name,value in [('SOURCE_FREEZE.json',freeze),('EXECUTION_PLAN_TEMPLATE.json',plan)]:
 with (p/name).open('x',encoding='utf-8') as s:json.dump(value,s,ensure_ascii=False,indent=2);s.write('\n')
print(json.dumps({'freeze_sha256':hashlib.sha256((p/'SOURCE_FREEZE.json').read_bytes()).hexdigest(),'template_sha256':hashlib.sha256((p/'EXECUTION_PLAN_TEMPLATE.json').read_bytes()).hexdigest(),'source_sha256':source},indent=2))
