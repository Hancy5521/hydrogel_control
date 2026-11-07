"""Pure calibration views and fixed beta-transformed-CDF reference.

Classical CDF recalibration/beta transforms/CRPS learning are cited precedents.
No raw-data extraction,forest fitting,source IO or evidence-of-disclosure claim.
"""
from copy import deepcopy
from fractions import Fraction
import math
import numpy as np
from scipy import special, optimize, integrate
from empirical_native import P, canonical, crps
from simplex_transport import fit_global, fit_context, CONTEXT_DEGENERACY_TOLERANCE

DESCRIPTOR_NAMES=('O_present','f_O','f_H','f_C_plus_f_N',
                  'f_F_Cl_Br_I','f_S_Se_Te','element_count_over118','entropy_over_log118')
BETA_LOG_BOUND=math.log(20.)
BETA_STARTS=((0.,0.),(.5,0.),(-.5,0.),(0.,.5),(0.,-.5))


def composition_descriptors(amounts):
    """Known atomic stoichiometry,not polymer mass/volume recipe or new labels."""
    if not isinstance(amounts,dict) or not amounts:
        raise ValueError('nonempty composition_amounts mapping required')
    vals={}
    for symbol,value in amounts.items():
        if not isinstance(symbol,str) or isinstance(value,(bool,np.bool_)):
            raise ValueError('element-string and finite nonnegative amount required')
        v=float(value)
        if not math.isfinite(v) or v<0:
            raise ValueError('finite nonnegative amounts required')
        if v>0:
            vals[symbol]=Fraction(str(v))
    if not vals or len(vals)>118:
        raise ValueError('between1 and118 positive element keys required')
    total=sum(vals.values(),Fraction(0)); f={k:v/total for k,v in vals.items()}
    get=lambda symbol:f.get(symbol,Fraction(0))
    if any(float(v)<=0 for v in f.values()):
        raise ArithmeticError('positive composition fraction not representable')
    entropy=-math.fsum(float(v)*math.log(float(v)) for v in f.values())
    out=[float(get('O')>0),float(get('O')),float(get('H')),
         float(get('C')+get('N')),float(sum((get(k) for k in ('F','Cl','Br','I')),Fraction(0))),
         float(sum((get(k) for k in ('S','Se','Te')),Fraction(0))),len(vals)/118.,entropy/math.log(118.)]
    if not all(math.isfinite(a) and -1e-12<=a<=1+1e-12 for a in out):
        raise ArithmeticError('descriptor outside its known natural bounds')
    # Only roundoff of bounded metadata;not a probability floor on any law.
    return [min(1.,max(0.,a)) for a in out]


def _identity_view(reason):
    return {'status':'IDENTITY_REFERENCE_NO_REPAIR','kind':'identity',
            'theta0':P.tolist(),'theta1':P.tolist(),'descriptor_index':None,
            'diagnostics':{'identity_reason':reason}}


def fit_views(laws, targets, descriptor_rows):
    """All candidate g share the caller's same legally completed OOF laws.

    The regularized calibration training objective is development-only.
    It is not an independent material score or a guarantee of improvement.
    """
    laws=list(laws); targets=list(targets)
    if not laws or len(targets)!=len(laws):
        raise ValueError('nonempty matched completed records required')
    g=np.asarray(descriptor_rows,dtype=float)
    if g.shape!=(len(laws),8) or not np.isfinite(g).all() or (g<0).any() or (g>1).any():
        raise ValueError('matched fixed8 natural-bound descriptors required')
    eligible=[j for j in range(8) if float(np.ptp(g[:,j]))>CONTEXT_DEGENERACY_TOLERANCE]
    anchor_index=eligible[0] if eligible else None  # Metadata-only,before any target fitting.
    identity=_identity_view('unmodified full reference distribution')
    global_fit=fit_global(laws,targets)
    if global_fit['diagnostics']['all_reference_laws_singleton']:
        reason='all references singleton;any T composed with F0 is F0'
        return {'identity':identity,'global':global_fit,'anchor':_identity_view(reason),
                'auto':_identity_view(reason),'anchor_index':anchor_index,'auto_index':None,
                'eligible_descriptor_indices':eligible,'candidate_fits':[],
                'selection_scope':'degenerate no repair information,not a discovery success'}
    fits=[]
    for j in eligible:
        result=fit_context(laws,targets,g[:,j])
        result['descriptor_index']=j; result['descriptor_name']=DESCRIPTOR_NAMES[j]
        fits.append(result)
    successful=[r for r in fits if r['status']=='CONVERGED_KKT']
    auto=min(successful,key=lambda r:(r['diagnostics']['objective_native'],r['descriptor_index'])) if successful else None
    anchor=next((r for r in fits if r['descriptor_index']==anchor_index),None)
    return {'identity':identity,'global':global_fit,
            'anchor':deepcopy(anchor) if anchor is not None else _identity_view('no nondegenerate metadata context'),
            'auto':deepcopy(auto) if auto is not None else _identity_view('no successfully fitted nondegenerate context'),
            'anchor_index':anchor_index,'auto_index':None if auto is None else auto['descriptor_index'],
            'auto_aliases_anchor':auto is not None and auto['descriptor_index']==anchor_index,
            'eligible_descriptor_indices':eligible,'candidate_fits':fits,
            'descriptor_names':list(DESCRIPTOR_NAMES),
            'selection_scope':'same past OOF native regularized CRPS,development only;not independent validation'}


def _beta_score_arrays(laws, targets):
    if not laws or len(laws)!=len(targets):
        raise ValueError('nonempty matched completed calibration pairs required')
    prefix=[]; below=[]; above=[]; tail=[]
    for law,target in zip(laws,targets):
        z=canonical(law); x,mass=z['support'],z['mass']; y=float(target)
        if not math.isfinite(y):
            raise ValueError('finite past target required')
        tail.append(max(x[0]-y,0.)+max(y-x[-1],0.)); v=Fraction(0)
        for j in range(len(x)-1):
            v+=mass[j]
            prefix.append((float(v),float(1-v)))
            below.append(max(0.,min(y,x[j+1])-x[j]))
            above.append(max(0.,x[j+1]-max(y,x[j])))
    return np.asarray(prefix,dtype=float).reshape(-1,2),np.asarray(below),np.asarray(above),math.fsum(tail)


def fit_beta(laws, targets):
    """Fixed5-start bounded L-BFGS-B reference;not an official-code reproduction."""
    laws=list(laws); targets=list(targets); n=len(laws)
    v,lo,hi,tail=_beta_score_arrays(laws,targets)
    identity=float(math.fsum(crps(a,b) for a,b in zip(laws,targets))/n)
    def objective(log_ab):
        a,b=np.exp(log_ab)
        F=special.betainc(a,b,v[:,0])
        # Survival via swapped-parameter complementary argument avoids1-F cancellation.
        S=special.betainc(b,a,v[:,1])
        value=float((np.dot(lo,F*F)+np.dot(hi,S*S)+tail)/n)
        if not math.isfinite(value):
            raise ArithmeticError('beta native objective not finite')
        return value
    starts=[]; successes=[]
    for index,start in enumerate(BETA_STARTS):
        result=optimize.minimize(objective,np.asarray(start),method='L-BFGS-B',
            bounds=[(-BETA_LOG_BOUND,BETA_LOG_BOUND)]*2,
            options={'maxiter':200,'ftol':1e-12,'gtol':1e-8,'maxls':40})
        row={'start_index':index,'start_log_ab':list(start),'success':bool(result.success),
             'status':int(result.status),'message':str(result.message),'iterations':int(result.nit),
             'function_evaluations':int(result.nfev),'objective_native':float(result.fun),
             'log_ab':np.asarray(result.x,dtype=float).tolist()}
        starts.append(row)
        if result.success and math.isfinite(result.fun) and np.isfinite(result.x).all():
            successes.append(row)
    best=min(successes,key=lambda r:(r['objective_native'],r['start_index'])) if successes else None
    if best is None or best['objective_native']>=identity:
        return {'status':'FAILED_REFERENCE_IDENTITY' if best is None else 'CONVERGED_IDENTITY_BEST',
                'alpha':1.,'beta':1.,'log_ab':[0.,0.], 'objective_native':identity,
                'reference_mean_CRPS_native':identity,'starts':starts,
                'scope':'fixed beta reference;identity included;past calibration development only'}
    a,b=np.exp(best['log_ab'])
    return {'status':'CONVERGED_FIXED_BETA_REFERENCE','alpha':float(a),'beta':float(b),
            'log_ab':best['log_ab'],'objective_native':best['objective_native'],
            'reference_mean_CRPS_native':identity,'starts':starts,
            'scope':'fixed beta reference,not official source-code reproduction;no guarantee of new-material gain'}


def _beta_interval_mass(left, right, a, b):
    if left==0:
        value=float(special.betainc(a,b,float(right))); error=0.; method='incomplete_beta_lower'
    elif right==1:
        value=float(special.betainc(b,a,float(1-left))); error=0.; method='incomplete_beta_upper'
    else:
        lower=float(special.betainc(a,b,float(right))-special.betainc(a,b,float(left)))
        upper=float(special.betainc(b,a,float(1-left))-special.betainc(b,a,float(1-right)))
        value=lower if left+right<=1 else upper
        error=0.; method='incomplete_beta_interval'
        if value<=0:
            # Unit-interval quadrature of positive mass,with stable logs and exact rational width.
            width=right-left; lf=float(left); sf=float(1-right); wf=float(width)
            if wf<=0 or lf<=0 or sf<=0:
                raise ArithmeticError('positive beta atom not representable;no floor or deletion allowed')
            logwidth=math.log(wf); norm=float(special.betaln(a,b))
            def integrand(t):
                logu=math.log(lf)+math.log1p(wf*t/lf)
                logs=math.log(sf)+math.log1p(wf*(1-t)/sf)
                return math.exp(logwidth+(a-1)*logu+(b-1)*logs-norm)
            value,error=integrate.quad(integrand,0.,1.,epsabs=0.,epsrel=1e-10,limit=100)
            method='positive_interval_quad_fallback'
    if not math.isfinite(value) or value<=0:
        raise ArithmeticError('positive beta mass lost numerically;do not silently drop source atom')
    return value,float(error),method


def apply_beta(law, beta_fit):
    z=canonical(law); a=float(beta_fit['alpha']); b=float(beta_fit['beta'])
    if not (.05<=a<=20 and .05<=b<=20):
        raise ValueError('beta parameters outside fixed reference bounds')
    if a==1 and b==1:
        return deepcopy(law)
    v=Fraction(0); q=[]; diagnostics=[]
    for w in z['mass']:
        right=v+w; value,error,method=_beta_interval_mass(v,right,a,b)
        q.append(Fraction(str(value)));diagnostics.append({'method':method,'quad_error_estimate':error});v=right
    total=sum(q,Fraction(0))
    if total<=0:
        raise ArithmeticError('positive beta total required')
    normalized=[w/total for w in q]
    return {'coordinate':'native','family':'weighted_empirical','support_native':list(z['support']),
            'weight_fractions':[str(w) for w in normalized],'unit':law.get('unit'),
            'transport':{'kind':'fixed_beta_reference','alpha':a,'beta':b,
            'unnormalized_float_mass_sum_error':float(total-1),
            'mass_arithmetic':'numerical beta interval probabilities,then exact Fraction normalization',
            'atom_numerical_diagnostics':diagnostics,'new_support_created':False,
            'not_a_certified_incomplete_beta_error_bound':True}}
