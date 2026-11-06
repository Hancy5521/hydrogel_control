from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
p=Path(r'H:\polymer_completion_public\research\relational_feedback_20261106T061957384774Z\full_nig_bma\material')
f=p/'code'/'acquisition.py';history=p/'pre_execution_history';history.mkdir(exist_ok=True)
old=f.read_bytes();oldsha=hashlib.sha256(old).hexdigest();(history/('acquisition_'+oldsha[:12]+'.py')).write_bytes(old)
s=f.read_text('utf-8-sig');a=s.index('def model_information(');b=s.index('\n\ndef risk_values(',a)
replacement='''def _posterior_KL(postlog,priorlog):
    """Positive f-divergence representation without clipping or tiny-prior ratios.

    KL=sum[p log(p/q)-p+q]. For r=log(p/q), choose p or q form
    to avoid exp(large positive r); use a local series only for cancellation.
    """
    r=postlog-priorlog[None,:];p=np.exp(postlog);q=np.broadcast_to(np.exp(priorlog)[None,:],r.shape)
    terms=np.empty_like(r);small=np.abs(r)<1e-4;positive=(r>=0)&~small;negative=(r<0)&~small
    x=r[small]
    terms[small]=p[small]*x*x*(.5-x/6+x*x/24-x*x*x/120+x**4/720)
    terms[positive]=p[positive]*(r[positive]+np.expm1(-r[positive]))
    terms[negative]=q[negative]*(np.exp(r[negative])*r[negative]-np.expm1(r[negative]))
    if not np.isfinite(terms).all() or (terms<0).any():raise ArithmeticError('positive KL components must be finite without clipping')
    return terms.sum(axis=1)


def model_information(ensemble,context,count):
    """E_query KL(posterior H || current H), one-dimensional full mixture t law.

    The original mixture query measure is integrated as a sum over generating H.
    Shared native Jacobians cancel. Every node uses a nonnegative KL f-divergence
    representation; no signed-component tail cancellation or probability floor.
    """
    laws=_marginals(ensemble,context);logs=np.asarray(posterior.normalize_logweights(ensemble['log_weights']));weights=np.exp(logs)
    pieces=[];mass=[]
    for j,c in enumerate(laws):
        nu=c['nu'];theta,wt=field.interval(count,math.atan(-field.LZ/math.sqrt(nu)),math.atan(field.LZ/math.sqrt(nu)))
        density=np.exp((nu-1)*np.log(np.cos(theta))-special.betaln(.5,nu/2));query=c['mu']+c['scale']*math.sqrt(nu)*np.tan(theta)
        logpdf=[]
        for component in laws:
            v=component['nu'];z=(query-component['mu'])/component['scale']
            logpdf.append(special.gammaln((v+1)/2)-special.gammaln(v/2)-.5*math.log(v*math.pi)
                -math.log(component['scale'])-(v+1)/2*np.log1p(z*z/v))
        logits=np.column_stack(logpdf)+logs[None,:]
        postlog=logits-special.logsumexp(logits,axis=1)[:,None]
        kl=_posterior_KL(postlog,logs)
        pieces.append(float(weights[j])*float((wt*density)@kl))
        mass.append(float(weights[j])*2*float(special.stdtr(nu,-field.LZ)))
    value=math.fsum(pieces);omitted=math.fsum(mass);bound=omitted*float(np.max(-logs))
    if not math.isfinite(value) or value<0:raise ArithmeticError('nonnegative finite model-information integral required')
    return {'value':value,'generating_component_positive_integrals':pieces,'omitted_query_probability':omitted,
        'MI_omitted_tail_bound_ideal_arithmetic':bound,'maximum_pointwise_KL_bound':float(np.max(-logs)),
        'not_IEEE_CDF_or_finite_quadrature_certificate':True,'no_negative_clipping':True,
        'exact_objective':'I(H;Yq|currentdata,knownXq)','deterministic_finite_quadrature':True,
        'positive_node_KL_representation':True,'native_response_Jacobian_cancels':True}
'''
f.write_text(s[:a]+replacement+s[b:],encoding='utf-8')
receipt={'actual_utc':datetime.now(timezone.utc).isoformat(),'root_request_before_material_execution':True,'old_source_sha256':oldsha,'new_source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'change':'same exact MI objective now E_query KL(posteriorH||priorH), positive per-node f-divergence with retained finite logweights and omittedmass*max(-logw) tail bound','old_source_preserved':str(history/('acquisition_'+oldsha[:12]+'.py')),'numerical_import_fit_or_new_material_reads':0,'old_static_receipt_scope':'old SHA preserved; new source requires static review'}
with (history/'MI_REPRESENTATION_CHANGE_BEFORE_EXECUTION.json').open('x',encoding='utf-8') as out:json.dump(receipt,out,ensure_ascii=False,indent=2);out.write('\n')
print(receipt['new_source_sha256'])
