"""Independent fixed arithmetic control; no producer imports or material reads."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime, timezone
import argparse,hashlib,json,math

EXACT_CASES=[
 {'name':'point_at_target','support':['2'],'weights':['1'],'target':'2'},
 {'name':'point_off_target','support':['2'],'weights':['1'],'target':'-3'},
 {'name':'duplicate_support_endpoint','support':['-1','-1','2'],'weights':['1/4','1/4','1/2'],'target':'-1'},
 {'name':'unequal_weights','support':['-2','0','3'],'weights':['1/8','5/8','1/4'],'target':'1/2'},
 {'name':'zero_mass_extreme','support':['-100','0','5'],'weights':['0','1/3','2/3'],'target':'0'},
 {'name':'symmetric_between_atoms','support':['-2','2'],'weights':['1/2','1/2'],'target':'0'},
 {'name':'upper_support_endpoint','support':['-3','1','4'],'weights':['1/6','1/3','1/2'],'target':'4'},
 {'name':'rational_support_negative_target','support':['-1/3','2/3','8/3'],'weights':['1/5','3/10','1/2'],'target':'-2/3'}]
GAUSSIAN_CASES=[{'mu':-2.,'sigma':1.,'target':-1.},{'mu':0.,'sigma':2.,'target':0.},{'mu':1.,'sigma':.25,'target':4.},{'mu':0.,'sigma':1.,'target':-6.}]
QUANTILES=['1/40','1/20','1/10','1/2','9/10','19/20','39/40']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def exact_control(case):
 x=list(map(F,case['support']));w=list(map(F,case['weights']));y=F(case['target'])
 assert len(x)==len(w) and all(a>=0 for a in w) and sum(w)==1
 # Independent energy identity uses all source positions, including duplicate/zero mass.
 energy=sum(a*abs(b-y) for a,b in zip(w,x))-F(1,2)*sum(a*b*abs(c-d) for a,c in zip(w,x) for b,d in zip(w,x))
 # Integral by interval midpoint avoids producer's lower-end step update.
 cuts=sorted(set(x+[y]));integral=F(0)
 for a,b in zip(cuts[:-1],cuts[1:]):
  middle=(a+b)/2;cdf=sum(v for u,v in zip(x,w) if u<=middle)
  integral+=(b-a)*(cdf-int(middle>=y))**2
 assert integral==energy and energy>=0
 left=sum(v for u,v in zip(x,w) if u<y);right=sum(v for u,v in zip(x,w) if u<=y)
 assert 0<=left<=right<=1
 grouped={a:sum(v for u,v in zip(x,w) if u==a) for a in set(x)}
 quantiles={}
 for p0 in QUANTILES:
  p=F(p0);q=min(a for a in sorted(grouped) if sum(v for u,v in grouped.items() if u<=a)>=p)
  qleft=sum(v for u,v in grouped.items() if u<q);qright=sum(v for u,v in grouped.items() if u<=q)
  assert qleft<p<=qright
  quantiles[p0]={'quantile':str(q),'cdf_left':str(qleft),'cdf_right':str(qright)}
 intervals={}
 for level in ['4/5','9/10','19/20']:
  c=F(level);lo=(1-c)/2;hi=(1+c)/2
  lower=F(quantiles[str(lo)]['quantile']);upper=F(quantiles[str(hi)]['quantile'])
  assert lower<=upper
  intervals[level]={'lower':str(lower),'upper':str(upper),'width':str(upper-lower),'covered':bool(lower<=y<=upper)}
 return dict(name=case['name'],CRPS_integral=str(integral),CRPS_energy=str(energy),PIT_left=str(left),PIT_right=str(right),PIT_mid=str((left+right)/2),quantiles=quantiles,intervals=intervals)
def gaussian_control(case):
 from scipy.integrate import quad
 mu,sigma,y=case['mu'],case['sigma'],case['target'];z=(y-mu)/sigma
 phi=math.exp(-z*z/2)/math.sqrt(2*math.pi);cdf=.5*math.erfc(-z/math.sqrt(2))
 closed=sigma*(z*(2*cdf-1)+2*phi-1/math.sqrt(math.pi))
 normalcdf=lambda a:.5*math.erfc(-(a-mu)/(sigma*math.sqrt(2)))
 normalsf=lambda a:.5*math.erfc((a-mu)/(sigma*math.sqrt(2)))
 a,e1=quad(lambda a:normalcdf(a)**2,-math.inf,y,epsabs=1e-11,epsrel=1e-11,limit=300)
 b,e2=quad(lambda a:normalsf(a)**2,y,math.inf,epsabs=1e-11,epsrel=1e-11,limit=300)
 error=abs(closed-a-b)
 assert math.isfinite(error) and error<=1e-9 and e1+e2<=1e-9
 return dict(case=case,closed_CRPS_native=closed,CDF_integral_native=a+b,absolute_error=error,quad_error_estimate=e1+e2)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();plan=read(a.plan);approval=read(a.approval)
 assert plan['status']=='REGISTERED' and plan['source_sha256']==sha(__file__)
 assert plan['exact_cases']==EXACT_CASES and plan['gaussian_cases']==GAUSSIAN_CASES
 assert plan['gaussian_absolute_tolerance']==1e-9 and plan['quad_epsabs']==1e-11 and plan['quad_epsrel']==1e-11
 assert approval['approved'] is True and approval['plan_sha256']==sha(a.plan) and approval['scope']=='INDEPENDENT_FIXED_DISCRETE_GAUSSIAN_SCORE_MATH'
 for item in plan['producer_static_source_pins']:
  assert sha(item['file'])==item['sha256']
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
 write(out/'EXECUTION_START.json',dict(actual_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(a.plan),approval_sha256=sha(a.approval),source_sha256=sha(__file__),material_reads=0,producer_imports=0))
 try:
  exact=[exact_control(c) for c in EXACT_CASES];normal=[gaussian_control(c) for c in GAUSSIAN_CASES]
  result=dict(status='PASS',actual_end_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(a.plan),source_sha256=sha(__file__),exact_case_count=len(exact),gaussian_case_count=len(normal),exact_results=exact,gaussian_results=normal,max_gaussian_absolute_error=max(r['absolute_error'] for r in normal),material_or_native_response_reads=0,producer_imports=0,model_fits=0,scope='Fixed scoring mathematics only; not material performance or universal numerical robustness',ideal_rational_identity_not_all_native_float_arithmetic=True)
  write(out/'INDEPENDENT_DISCRETE_GAUSSIAN_SCORE_MATH.json',result);print(json.dumps(dict(status='PASS',exact_cases=len(exact),gaussian_cases=len(normal),max_gaussian_error=result['max_gaussian_absolute_error'])))
 except Exception as e:
  write(out/'FAILED_STOP.json',dict(actual_utc=datetime.now(timezone.utc).isoformat(),error_type=type(e).__name__,message=str(e),retained_failure=True));raise
if __name__=='__main__':main()
