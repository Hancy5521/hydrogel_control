from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
p=Path(r'H:\polymer_completion_public\research\relational_feedback_20261106T061957384774Z\generation')
source={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ('joint_sampler.py','sampling_gate.py')}
for name in source:ast.parse((p/name).read_text('utf-8-sig'))
proposal=json.loads((p/'SAMPLING_GATE_PROPOSAL.json').read_text('utf-8'))
plan={'status':'TEMPLATE_NOT_AUTHORIZATION','approved_for_execution':False,'actual_registered_utc':None,
 'source_sha256':source,'proposal_sha256':'7c9a99e2ab34a2b64e3ad579f02357791a7df4b09eb3e4ed6035fc77f62a4ca8',
 'fixture':proposal['input_fixture'],'seed':20261106,'n_draws':131072,'numpy_version':'2.1.3',
 'output':'one unique new directory; draws_and_aux.npz plus metadata/inputbinding/primitive summaries only',
 'scope':'single frozen artificial two-row fixture; no material reads/fits/model updates; independent audit decides validation'}
freeze={'status':'SOURCE_FROZEN_AST_ONLY_NO_DRAWS','actual_utc':datetime.now(timezone.utc).isoformat(),
 'source_sha256':source,'proposal_sha256':plan['proposal_sha256'],'fixture_sha256':proposal['input_fixture']['sha256'],
 'driver_primitive_summary_not_independent_validation':True,'primary_material_7sources_and_2core_unchanged':True,
 'sampler_original_bytes_preserved':source['joint_sampler.py']=='5b77cde6f6356eb977596ea89564da04e95a0e57688414d142d40442cd44bfae'}
for name,obj in [('SAMPLING_EXECUTION_PLAN_TEMPLATE.json',plan),('SAMPLING_SOURCE_FREEZE.json',freeze)]:
 with (p/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'source_sha256':source,'template_sha256':hashlib.sha256((p/'SAMPLING_EXECUTION_PLAN_TEMPLATE.json').read_bytes()).hexdigest(),'freeze_sha256':hashlib.sha256((p/'SAMPLING_SOURCE_FREEZE.json').read_bytes()).hexdigest()},indent=2))
