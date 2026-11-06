from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast
p=Path(r'H:\polymer_completion_public\research\relational_feedback_20261106T061957384774Z\full_nig_bma\material')
keys=['material_model.py','acquisition.py','reference_models.py','run_material.py'];files={}
for name in keys:
 f=p/'code'/name;ast.parse(f.read_text('utf-8-sig'));files[name]=hashlib.sha256(f.read_bytes()).hexdigest()
value={'actual_utc':datetime.now(timezone.utc).isoformat(),'status':'RUNNER_AND_OWNED_FUNCTIONAL_SOURCES_READY_AST_ONLY_NO_NUMERICAL_IMPORT','source_sha256':files,'source_mutations_after_this_receipt':'none planned; root all7 freeze after IO/scorer owners finish','forecasts':{'main_BMA':1728,'GP_RF_and_affine_on_full_random':1728,'GP_policy_own':288,'full_frozenwarmweights_BMA':288,'all_terminal':4032,'prequery':48,'frozen_payload_files':123},'fits':{'GP':13,'RF':5,'all_sklearn':18,'warm_conjugate_initializations':3,'feedback_component_conjugate_updates':144},'queries':{'per_policy':8,'policies':6,'counterfactual_events':48,'common_setup_cells':352,'new_numeric_or_oracle_execution_by_source_preparation':0},'numerical_rule':{'levels':[32,64,128,256],'native_max_absolute_consecutive_level_difference_across_available_query_means':2e-5,'MI_nats_max_absolute_query_difference':2e-8,'fail_before_next_Y_if_unmet_at256':True,'all_levels_preserved':True,'certificate':False},'fixed_foundation_source_sha256':{'relation_posterior.py':'1d74cde91519ce282c6ebee54e53807a2a79be0cebf19e06221eb4f16e8b7528','risk_field.py':'cda5178b9c5138df647a0b9fffab2da30a19bde6082aa6fd4cc8b0910c6b92aa'},'needs_root': 'all7source plan+mathreceipt hashes/actualUTC/metadata source pins and independent IO/math static clearance before unique run'}
with (p/'RUNNER_SOURCE_PREPARATION.json').open('x',encoding='utf-8') as s:json.dump(value,s,ensure_ascii=False,indent=2);s.write('\n')
print(hashlib.sha256((p/'RUNNER_SOURCE_PREPARATION.json').read_bytes()).hexdigest())
