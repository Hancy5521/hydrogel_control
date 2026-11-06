"""Full NIG model-average native CRPS information value, including within-H learning.

The loss is the integral of conditional-CDF variance in reported native Y units.
A signed between-H closure is diagnostic: B_before-E B_after can be negative and
is never clipped. The pure fixed-Gaussian C:D special case is not used here.
Deterministic finite-domain quadrature is not an IEEE/global error certificate.
"""
from functools import lru_cache
import math
import numpy as np
from scipy import special,stats
from relation_posterior import pairs,normalize_logweights

LEVELS=(32,64,128,256)
LV=32.
LZ=64.


@lru_cache(maxsize=8)
def _legendre(count):
    return np.polynomial.legendre.leggauss(count)


def interval(count,lower,upper):
    nodes,weights=_legendre(count)
    return (lower+upper)/2+(upper-lower)*nodes/2,weights*(upper-lower)/2


def inverse_yj(u):
    u=np.asarray(u,dtype=float);result=np.empty_like(u);positive=u>=0
    result[positive]=u[positive]*(1+u[positive]/4)
    result[~positive]=-np.expm1((2/3)*np.log1p(-1.5*u[~positive]))
    return result


def derivative(u):
    u=np.asarray(u,dtype=float);result=np.empty_like(u);positive=u>=0
    result[positive]=1+.5*u[positive]
    result[~positive]=np.exp(-np.log1p(-1.5*u[~positive])/3)
    return result


def checked_transform(transform):
    c,d,s0=(float(transform[key]) for key in ('center','sd','s0'))
    if not all(math.isfinite(x) for x in (c,d,s0)) or d<=0 or s0<=0:
        raise ValueError('shared finite response center and positive sd/s0 required')
    return c,d,s0


def _query_logpdf(value,pair):
    nu=pair['nu'];z=(value-pair['mu_query'])/pair['scale_query']
    return (special.gammaln((nu+1)/2)-special.gammaln(nu/2)-.5*math.log(nu*math.pi)
        -math.log(pair['scale_query'])-(nu+1)/2*np.log1p(z*z/nu))


def conditional_cdfs(pairlaws,logweights,query_modeled_y,target_modeled_z):
    """All-H query reweighting followed by each complete bivariate-t conditional.

    Shapes: query Q, target V; component CDFs V,Q,H; posterior weights Q,H.
    Both CDF/SF routes are returned so far-tail differences avoid subtracting ones.
    """
    q=np.atleast_1d(np.asarray(query_modeled_y,dtype=float));v=np.atleast_1d(np.asarray(target_modeled_z,dtype=float))
    logits=np.column_stack([_query_logpdf(q,p) for p in pairlaws])+np.asarray(logweights)[None,:]
    postlog=logits-special.logsumexp(logits,axis=1)[:,None];post=np.exp(postlog)
    cdfs=[];sfs=[]
    for p in pairlaws:
        z=(q-p['mu_query'])/p['scale_query'];nu,rho=p['nu'],p['rho']
        mu=p['mu_target']+rho*p['scale_target']*z
        scale=p['scale_target']*np.sqrt((nu+z*z)/(nu+1)*(1-rho*rho))
        arg=(v[:,None]-mu[None,:])/scale[None,:]
        cdfs.append(special.stdtr(nu+1,arg));sfs.append(special.stdtr(nu+1,-arg))
    cdf=np.stack(cdfs,axis=2);sf=np.stack(sfs,axis=2)
    return {'component_CDF':cdf,'component_SF':sf,'posterior_weights':post,'posterior_log_weights':postlog,
        'mixture_CDF':np.sum(cdf*post[None,:,:],axis=2),'mixture_SF':np.sum(sf*post[None,:,:],axis=2)}


def _kernel(v,pairlaws,logweights,count):
    w=np.exp(logweights)
    prior_cdf=np.column_stack([special.stdtr(p['nu'],(v-p['mu_target'])/p['scale_target']) for p in pairlaws])
    prior_sf=np.column_stack([special.stdtr(p['nu'],-(v-p['mu_target'])/p['scale_target']) for p in pairlaws])
    prior_mix_cdf=prior_cdf@w;prior_mix_sf=prior_sf@w
    right=prior_mix_cdf>=.5
    prior_components=np.where(right[:,None],prior_sf,prior_cdf)
    prior_mix=np.where(right,prior_mix_sf,prior_mix_cdf)
    before=np.sum(w[None,:]*(prior_components-prior_mix[:,None])**2,axis=1)
    total=np.zeros(len(v));within=np.zeros((len(v),3));after=np.zeros(len(v));diagnostics=[]
    for j,p in enumerate(pairlaws):
        theta,wt=interval(count,math.atan(-LZ/math.sqrt(p['nu'])),math.atan(LZ/math.sqrt(p['nu'])))
        density=np.exp((p['nu']-1)*np.log(np.cos(theta))-special.betaln(.5,p['nu']/2))
        qweights=wt*density
        query=p['mu_query']+p['scale_query']*math.sqrt(p['nu'])*np.tan(theta)
        cond=conditional_cdfs(pairlaws,logweights,query,v)
        components=np.where(right[:,None,None],cond['component_SF'],cond['component_CDF'])
        mixture=np.where(right[:,None],cond['mixture_SF'],cond['mixture_CDF'])
        total+=w[j]*((mixture-prior_mix[:,None])**2@qweights)
        within[:,j]=w[j]*((components[:,:,j]-prior_components[:,j,None])**2@qweights)
        after+=w[j]*(np.sum(cond['posterior_weights'][None,:,:]*(components-mixture[:,:,None])**2,axis=2)@qweights)
        analytic_mass=1-2*float(special.stdtr(p['nu'],-LZ))
        diagnostics.append({'generating_hypothesis':p['id'],'theta_nodes':count,'weighted_original_density':True,
            'renormalized':False,'mass_quadrature':float(qweights.sum()),'mass_analytic':analytic_mass,
            'mass_absolute_difference':abs(float(qweights.sum())-analytic_mass),
            'posterior_query_logweights_retained':True,'display_weight_underflow_nodes':int((cond['posterior_weights']==0).sum())})
    if (not np.isfinite(total).all() or not np.isfinite(within).all() or not np.isfinite(after).all()
        or (total<0).any() or (within<0).any() or (after<0).any()):
        raise ArithmeticError('squared CDF kernels must be finite and nonnegative without hard clipping')
    return {'full':total,'within':within,'B_before':before,'E_B_after':after},diagnostics


def tail_accounts(pairlaws,logweights,transform,lower,upper):
    center,sd,s0=checked_transform(transform);w=np.exp(logweights);bounds=[]
    for weight,p in zip(w,pairlaws):
        nu=p['nu'];mu=center+sd*p['mu_target'];scale=sd*p['scale_target']
        density=float(stats.t.pdf(LV,nu));sf=float(special.stdtr(nu,-LV))
        i0=(nu+LV*LV)/(nu-1)*density
        i1=.5*(nu*sf+LV*(nu+LV*LV)*density)/(nu-2)
        outer=2*s0*scale*((1+.5*abs(mu))*i0+.5*scale*i1)
        bounds.append({'id':p['id'],'weight':float(weight),'outer_component_bound':outer,
            'weighted_outer_bound':float(weight)*outer,'query_tail_probability':2*float(special.stdtr(nu,-LZ)),
            'I0_conservative_ignored_negative_term':i0,'I1_conservative_ignored_negative_term':i1})
    native_edges=s0*inverse_yj(center+sd*np.asarray([lower,upper]));width=float(native_edges[1]-native_edges[0])
    query_probability=math.fsum(x['weight']*x['query_tail_probability'] for x in bounds)
    outer=math.fsum(x['weighted_outer_bound'] for x in bounds);query=width*query_probability
    return {'component_bounds':bounds,'native_target_window_width':width,'weighted_omitted_query_probability':query_probability,
        'outer_target_union_tail_bound_ideal_arithmetic':outer,'inner_query_tail_bound_ideal_arithmetic':query,
        'total_value_tail_bound_ideal_arithmetic':outer+query,
        'finite_query_decomposition_closure_allowance':2*query,
        'not_IEEE_CDF_or_finite_quadrature_certificate':True,'no_truncation_renormalization':True}


def native_value(ensemble,query_context,target_context,transform,count):
    if count not in LEVELS: raise ValueError('fixed pre-output GL level required')
    center,sd,s0=checked_transform(transform);pairlaws=pairs(ensemble,query_context,target_context)
    logs=np.asarray(normalize_logweights(ensemble['log_weights']))
    lower=min(p['mu_target']-LV*p['scale_target'] for p in pairlaws)
    upper=max(p['mu_target']+LV*p['scale_target'] for p in pairlaws)
    split=sorted(set([lower,upper]+[x for x in (0.,-center/sd) if lower<x<upper]))
    full=[];within=[];before=[];after=[];segments=[]
    for low,high in zip(split[:-1],split[1:]):
        v,wt=interval(count,low,high);kernel,diagnostics=_kernel(v,pairlaws,logs,count)
        native_weights=wt*s0*sd*derivative(center+sd*v)
        total=float(native_weights@kernel['full']);component=(native_weights@kernel['within']).tolist()
        b=float(native_weights@kernel['B_before']);a=float(native_weights@kernel['E_B_after'])
        full.append(total);within.append(component);before.append(b);after.append(a)
        segments.append({'target_modeled_interval':[low,high],'nodes_per_segment':count,
            'full_value':total,'within_components':component,'B_before':b,'E_B_after':a,'query_diagnostics':diagnostics})
    fv=math.fsum(full);wv=[math.fsum(row[h] for row in within) for h in range(3)]
    bv,av=math.fsum(before),math.fsum(after);signed=bv-av
    return {'native_full_VOI':fv,'native_weighted_within_components':wv,'native_weighted_within_total':math.fsum(wv),
        'native_B_before':bv,'native_E_B_after':av,'native_signed_between':signed,
        'decomposition_closure':fv-math.fsum(wv)-signed,'signed_between_clipped':False,
        'nodes_per_segment':count,'target_modeled_union':[lower,upper],'segments':segments,
        'pair_laws':pairlaws,'weights_display_only':np.exp(logs).tolist(),'log_weights':logs.tolist(),
        'tail':tail_accounts(pairlaws,logs,transform,lower,upper),'reported_native_objective':True,
        'model_based_not_actual_outcome_guarantee':True}
