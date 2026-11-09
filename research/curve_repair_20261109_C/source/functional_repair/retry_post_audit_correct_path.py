"""Operational retry only: correct authorization CLI path after zero-check I/O FAIL.

No scientific function or tolerance is replaced. Imports only our pinned
independent audit source, never the production implementation.
"""
import importlib.util
from pathlib import Path
import traceback

HERE=Path(__file__).resolve().parent
source=HERE/'independent_audit.py'
spec=importlib.util.spec_from_file_location('functional_independent_v1',source)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
assert audit.sha(source)=='9c2a377e72fbfb1160638faf30c9b8d208ef77252eb56d583a19c2566b7a2669'
out=HERE/'pilot_once';auth=out/'ROOT_POST_AUDIT_AUTHORIZATION.json';reg=HERE/'ROOT_REGISTERED_V3.json'
registration=out/'INDEPENDENT_POST_AUDIT_REGISTRATION_V2.json'
audit.write_new(registration,{'schema':'darus-functional-post-audit-operational-retry-v1',
    'actual_utc':audit.utc(),'authority':'Root explicit team-message POST authorization and corrected file path',
    'source_sha256':audit.sha(source),'retry_wrapper_sha256':audit.sha(__file__),
    'root_authorization_path':str(auth),'root_authorization_sha256':audit.sha(auth),
    'prior_fail_sha256':audit.sha(out/'INDEPENDENT_POST_Y.json'),
    'reason':'First invocation used parent directory instead of pilot_once for root authorization. Failed before check0 and before source Y reads.',
    'scientific_source_changed':False,'tolerances_changed':False,'production_score_repeated':False})
c=audit.Checks();payload={}
try:
    payload=audit.post(reg,out,auth,c);status='PASS';error=None
except Exception as exc:
    status='FAIL';error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
result={'schema':'darus-functional-independent-audit-v1','status':status,'actual_utc':audit.utc(),
    'phase':'post','predictions_sha256':audit.sha(out/'PREDICTIONS.json'),
    'prepared_sha256':audit.sha(out/'PREPARED.json'),'targets_sha256':audit.sha(out/'TARGETS.json'),
    'scores_sha256':audit.sha(out/'SCORES.json'),'audit_source_sha256':audit.sha(source),
    'audit_registration_sha256':audit.sha(registration),'check_count':c.count,'checks':c.groups,
    'operational_retry_only':True,'failure':error,**payload}
report=out/'INDEPENDENT_POST_Y_V2.json';audit.write_new(report,result)
print({'status':status,'checks':c.count,'report':str(report),'sha256':audit.sha(report),'failure':error['message'] if error else None})
if status!='PASS':raise SystemExit(1)
