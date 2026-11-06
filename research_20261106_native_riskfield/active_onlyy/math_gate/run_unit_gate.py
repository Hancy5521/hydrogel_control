"""One bounded synthetic producer-unit check for the native risk readout helper."""
import argparse
from datetime import datetime,timezone
import hashlib,importlib.util,json,math
from pathlib import Path
import traceback
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def require(x,m):
 if not x:raise AssertionError(m)


def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();plan=read(a.plan);approval=read(a.approval);source=sha(__file__)
 require(plan['script_sha256']==source and approval.get('approved')is True and approval['scope']=='NATIVE_READOUT_DECOMPOSITION_BOUNDED_UNIT_GATE'and approval['plan_sha256']==sha(a.plan),'Exactrootunitapproval')
 for pin in plan['inputs'].values():require(sha(pin['file'])==pin['sha256'],'Pinnedinputchanged')
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);started=datetime.now(timezone.utc).isoformat()
 try:
  spec=importlib.util.spec_from_file_location('_frozen_actual_readout_helper',plan['inputs']['core_source']['file']);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
  old=read(plan['inputs']['old_fixed12_result']['file'])['evaluations'];reused=[]
  for r in old:
   f=r['fixture'];family=f['family'];nu=6. if family=='normal'else f['nu'];scale=f['scale']/math.sqrt(nu/(nu-2))if family=='normal'else f['scale']
   if f['loss_coordinate']=='affine':
    value=f['s0']*f['scale']*float(core._student_affine_kernel_integral(nu,np.asarray([f['rho']]))[0]);levels=[]
   else:
    levels=[]
    for n in(32,64,128,256):
     values,bound=core.pair_values(f['mu'],scale,f['s0'],nu,np.asarray([f['rho']]),family,n)
     levels.append({'nodes':n,'value':float(values[0]),'tail':float(bound)})
    value=levels[-1]['value']
   reused.append({'id':f['id'],'value':value,'levels':levels})
  unique=[]
  for nu in(6.,70.):
   for mu in(-1.,1.):
    for rho in(0.,.6):
     levels=[]
     for n in(32,64,128,256):
      val,bound=core.pair_values(mu,1.,2.,nu,np.asarray([rho]),'student_t',n)
      levels.append({'nodes':n,'value':float(val[0]),'tail':float(bound)})
     kernel=core._kernel(np.asarray([0.,.5,2.,8.]),np.asarray([rho]),nu,'student_t',256)[0]
     unique.append({'nu':nu,'mu':mu,'rho':rho,'scale':1.,'s0':2.,'levels':levels,'kernel_probe_v':[0.,.5,2.,8.],'kernel_values':kernel.tolist()})
  negative_u=[-10.,-1.,-1e-5,0.,1.];h=core._negative_correction(np.asarray(negative_u)).tolist()
  # Same pre-affine NIG fixture as fixed12, expressed in a nontrivial standardized response coordinate.
  nig={'mean':[-.5/.7,1/.7],'V':[[.2,0.],[0.,1.]],'alpha':3.,'b':1.5/(.7*.7)}
  api={family:core.risk_values(nig,[[1.],[-1.1]],[[1.],[-1.]],{'center':.5,'sd':.7,'s0':2.},family)for family in('student_t','normal')}
  result={'status':'COMPLETE_BOUNDED_NATIVE_READOUT_UNIT_GATE','reused_fixed12':reused,'new_fixed8_scalars':unique,'negative_correction_probes':{'u':negative_u,'h':h},'nontrivial_response_affine_public_API':api,'source':{'driver':source,'core':sha(plan['inputs']['core_source']['file'])},'scope':'producer-sideunitcallsactualhelper;no source data ormaterialfit;oldindep2Dreference reusednotrerun'}
 except Exception as e:result={'status':'FAIL','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 result.update({'start_utc':started,'end_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(a.plan),'approval_sha256':sha(a.approval),'new_material_data_or_model_fits':0})
 f=out/'READOUT_UNIT_RESULTS.json'
 with f.open('x',encoding='utf-8')as o:json.dump(result,o,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({'status':result['status'],'file':str(f),'sha256':sha(f)}));return 0 if result['status'].startswith('COMPLETE')else 1
if __name__=='__main__':raise SystemExit(main())
