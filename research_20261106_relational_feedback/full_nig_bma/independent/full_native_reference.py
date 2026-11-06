"""Independent dense NIG algebra and direct modeled-query/native-target CDF integration."""
import math
import numpy as np
from scipy import special
from scipy.integrate import quad_vec


def features(name,x):
    a,b=x
    base=[1.,a,b]
    if name=='affine':return np.array(base)
    if name=='interaction':return np.array(base+[a*b])
    if name=='fullquad':return np.array(base+[a*b,a*a,b*b])
    raise ValueError('fixed known feature map only')


def pair(state,q,t):
    v=np.array(state['V']);m=np.array(state['mean'])
    a=features(state['id'],q);b=features(state['id'],t)
    dq=1+float(a@v@a);dt=1+float(b@v@b)
    return {'q_mu':float(a@m),'t_mu':float(b@m),
            'q_scale':math.sqrt(state['b']/state['alpha']*dq),
            't_scale':math.sqrt(state['b']/state['alpha']*dt),
            'rho':float(a@v@b)/math.sqrt(dq*dt),'nu':2*state['alpha']}


def t_logpdf(value,mu,scale,nu):
    x=(value-mu)/scale
    return float(special.gammaln((nu+1)/2)-special.gammaln(nu/2))-.5*math.log(nu*math.pi)-math.log(scale)-(nu+1)/2*math.log1p(x*x/nu)


def conditional(p,value):
    x=(value-p['q_mu'])/p['q_scale'];nu=p['nu'];rho=p['rho']
    return {'mu':p['t_mu']+rho*p['t_scale']*x,
            'scale':p['t_scale']*math.sqrt((nu+x*x)/(nu+1)*(1-rho*rho)),
            'nu':nu+1}


def dense_update(state,x,value):
    v=np.array(state['V']);m=np.array(state['mean']);phi=features(state['id'],x)
    precision=np.linalg.inv(v);new_precision=precision+np.outer(phi,phi)
    new_v=np.linalg.inv(new_precision);new_m=np.linalg.solve(new_precision,precision@m+phi*value)
    new_b=state['b']+.5*(value*value+float(m@precision@m)-float(new_m@new_precision@new_m))
    return {'id':state['id'],'mean':new_m.tolist(),'V':new_v.tolist(),'alpha':state['alpha']+.5,
            'b':new_b,'labeled_updates':state.get('labeled_updates',0)+1}


def marginal(state,x):
    phi=features(state['id'],x);v=np.array(state['V'])
    return {'mu':float(phi@np.array(state['mean'])),
            'scale':math.sqrt(state['b']/state['alpha']*(1+float(phi@v@phi))),'nu':2*state['alpha']}


def weights_at(weights,pairs,value):
    logs=[math.log(w)+t_logpdf(value,p['q_mu'],p['q_scale'],p['nu']) for w,p in zip(weights,pairs)]
    maximum=max(logs);denominator=math.fsum(math.exp(x-maximum) for x in logs)
    log_density=maximum+math.log(denominator)
    return np.array([math.exp(x-log_density) for x in logs]),log_density


def inverse_yj(u):
    return u*(1+u/4) if u>=0 else -math.expm1((2/3)*math.log1p(-1.5*u))


def forward_yj(x):
    return 2*x/(math.sqrt(1+x)+1) if x>=0 else -math.expm1(1.5*math.log1p(-x))/1.5


def to_native(z,response):
    return response['s0']*inverse_yj(response['center']+response['sd']*z)


def to_modeled(y,response):
    return (forward_yj(y/response['s0'])-response['center'])/response['sd']


def sf(x,nu):return float(special.stdtr(nu,-x))


def tail_integrals(x,nu):
    density=math.exp(t_logpdf(x,0.,1.,nu));survival=sf(x,nu)
    i0=(nu+x*x)/(nu-1)*density-x*survival
    i1=.5*((nu*survival+x*(nu+x*x)*density)/(nu-2)-x*x*survival)
    if i0<0 or i1<0:raise ArithmeticError('negative raw analytic tail integral; no clipping')
    return i0,i1


def ideal_tails(proposal,pairs,lower,upper):
    response=proposal['response_transform'];width=to_native(upper,response)-to_native(lower,response)
    a=response['s0']*response['sd']*(1+.5*abs(response['center']))
    b=.5*response['s0']*response['sd']**2
    target=0.;query=0.
    for w,p in zip(proposal['weights'],pairs):
        total0=total1=0.
        for cutoff in ((upper-p['t_mu'])/p['t_scale'],(p['t_mu']-lower)/p['t_scale']):
            x,y=tail_integrals(cutoff,p['nu']);total0+=x;total1+=y
        target+=w*p['t_scale']*((a+b*abs(p['t_mu']))*total0+b*p['t_scale']*total1)
        query+=w*2*sf(64.,p['nu'])
    return {'target_VOI_ideal_upper_bound':target,'query_omitted_mixture_mass':query,
            'query_VOI_ideal_upper_bound':width*query,'native_target_interval_width':width,
            'finite_closure_ideal_bound_for_equal_initial_targetCDF':2*width*query,
            'not_a_IEEE_or_quad_error_bound':True}


def split_integral(function,lower,upper,points,epsabs=2e-9):
    cuts=sorted(set([lower,upper]+[float(x) for x in points if lower<x<upper]))
    result=None;error=0.
    for a,b in zip(cuts[:-1],cuts[1:]):
        value,estimate,info=quad_vec(function,a,b,epsabs=epsabs/len(cuts),epsrel=2e-8,limit=180,full_output=True)
        if not info.success:raise ArithmeticError('independent quad_vec unsuccessful: '+str(info.message))
        result=value if result is None else result+value;error+=float(estimate)
    return result,error


def native_reference(proposal):
    """One fixed mixture; native target y and direct modeled-query z coordinates."""
    states=proposal['models'];weights=np.array(proposal['weights']);response=proposal['response_transform']
    pairs=[pair(s,proposal['query_context'],proposal['target_context']) for s in states]
    first=pairs[0]
    if any(abs(item[key]-first[key])>1e-14 for item in pairs for key in ('t_mu','t_scale','nu')):
        raise ValueError('this bounded reference requires the declared identical initial target CDFs')
    lower=min(p['t_mu']-32*p['t_scale'] for p in pairs);upper=max(p['t_mu']+32*p['t_scale'] for p in pairs)
    native_lower=to_native(lower,response);native_upper=to_native(upper,response)
    target_cuts=[0.,to_native(0.,response)]
    for p in pairs:
        target_cuts.extend(to_native(p['t_mu']+k*p['t_scale'],response) for k in (-16,-8,-4,-2,-1,0,1,2,4,8,16))
    max_inner_error=0.;outer_errors=[];records=[];allvalue=np.zeros(3)
    for source,(p,source_weight) in enumerate(zip(pairs,weights)):
        def outer(query):
            nonlocal max_inner_error
            post,unused=weights_at(weights,pairs,query)
            conditioned=[conditional(item,query) for item in pairs]
            def inner(native_y):
                modeled=to_modeled(native_y,response)
                before=np.array([special.stdtr(item['nu'],(modeled-item['t_mu'])/item['t_scale']) for item in pairs])
                after=np.array([special.stdtr(item['nu'],(modeled-item['mu'])/item['scale']) for item in conditioned])
                f0=float(weights@before);f1=float(post@after)
                return np.array([(f1-f0)**2,(after[source]-before[source])**2,
                                 float(post@((after-f1)**2))])
            value,error=split_integral(inner,native_lower,native_upper,target_cuts)
            max_inner_error=max(max_inner_error,error)
            return source_weight*math.exp(t_logpdf(query,p['q_mu'],p['q_scale'],p['nu']))*value
        query_lower=p['q_mu']-64*p['q_scale'];query_upper=p['q_mu']+64*p['q_scale']
        query_cuts=[p['q_mu']+k*p['q_scale'] for k in (-16,-8,-4,-2,-1,0,1,2,4,8,16)]
        value,error=split_integral(outer,query_lower,query_upper,query_cuts)
        allvalue+=value;outer_errors.append(error)
        records.append({'generating_H':source,'weighted_total_native':float(value[0]),
                        'weighted_within_H_native':float(value[1]),'weighted_expected_Bafter_native':float(value[2]),
                        'query_domain':[query_lower,query_upper],'quad_error_estimate':error})
    tails=ideal_tails(proposal,pairs,lower,upper)
    return {'full_native_VOI':float(allvalue[0]),'weighted_within_native_VOI':float(allvalue[1]),
            'Bbefore_native':0.,'expected_Bafter_native':float(allvalue[2]),
            'signed_between_native':-float(allvalue[2]),
            'finite_closure_raw':float(allvalue[0]-allvalue[1]+allvalue[2]),
            'by_generating_H':records,'target_modeled_domain':[lower,upper],
            'target_native_domain':[native_lower,native_upper],'ideal_tails':tails,
            'outer_quad_error_sum':math.fsum(outer_errors),'max_inner_quad_error_estimate':max_inner_error,
            'error_proxy_not_directed_certificate':True,'SUT_import_or_fit_or_material_reads':0}
