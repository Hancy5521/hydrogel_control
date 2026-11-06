"""Independent batch sufficient statistics for three conditional NIG hypotheses; no SUT."""
import math
import numpy as np
from scipy import special

IDS=('affine','interaction','fullquad')


def transform_row(t,row,forward):
    return [(forward(v/t['native_first_scale_s0'][i])-t['u_center_warm'][i])/t['u_sample_SD_warm'][i]
            for i,v in enumerate(row)]


def expansion_transform(warm_z):
    terms=[[r[0]*r[1],r[0]**2,r[1]**2] for r in warm_z]
    n=len(terms);mean=[math.fsum(r[i] for r in terms)/n for i in range(3)]
    sd=[math.sqrt(math.fsum((r[i]-mean[i])**2 for r in terms)/(n-1)) for i in range(3)]
    return {'center':mean,'sd':sd,'ddof':1,'frozen_after_warm':True}


def phi(name,x,expansion):
    a,b=x;terms=[a*b,a*a,b*b]
    expanded=[(v-expansion['center'][i])/expansion['sd'][i] for i,v in enumerate(terms)]
    return [1.,a,b]+([] if name=='affine' else expanded[:1] if name=='interaction' else expanded)


def batch_nig(name,rows_z,expansion):
    design=[phi(name,row[:2],expansion) for row in rows_z];d={'affine':3,'interaction':4,'fullquad':6}[name];n=len(design)
    precision=np.array([[(1. if i==j else 0.)+math.fsum(x[i]*x[j] for x in design)
                         for j in range(d)] for i in range(d)])
    cross=np.array([math.fsum(x[i]*row[2] for x,row in zip(design,rows_z)) for i in range(d)])
    mean=np.linalg.solve(precision,cross);covariance=np.linalg.inv(precision)
    alpha=3+n/2;b=1.5+.5*(math.fsum(r[2]**2 for r in rows_z)-float(mean@precision@mean))
    sign,logdet=np.linalg.slogdet(precision)
    if sign<=0 or b<=0:raise ArithmeticError('positive dense conjugate statistics required')
    evidence=math.lgamma(alpha)-math.lgamma(3)+3*math.log(1.5)-alpha*math.log(b)-.5*logdet-n/2*math.log(2*math.pi)
    return {'mean':mean.tolist(),'V':covariance.tolist(),'alpha':alpha,'b':b,'labeled_updates':n},evidence


def batch_ensemble(rows_z,expansion):
    states=[];logs=[]
    for name in IDS:
        state,evidence=batch_nig(name,rows_z,expansion);states.append({'id':name,'state':state});logs.append(evidence)
    normalizer=float(special.logsumexp(logs))
    return {'models':states,'log_weights':[float(x-normalizer) for x in logs],
            'expansion_transform':expansion,'true_label_updates':len(rows_z)},logs


def scalar(state,design):
    x=np.array(design);m=np.array(state['mean']);v=np.array(state['V'])
    return {'mu':float(x@m),'scale':math.sqrt(state['b']/state['alpha']*(1+float(x@v@x))),
            'nu':2*state['alpha']}


def law(ensemble,x,response,override=None):
    logs=ensemble['log_weights'] if override is None else override
    components=[]
    for model in ensemble['models']:
        predicted=scalar(model['state'],phi(model['id'],x,ensemble['expansion_transform']))
        components.append({'id':model['id'],'location':response['center']+response['sd']*predicted['mu'],
                           'scale':response['sd']*predicted['scale'],'df':predicted['nu']})
    return {'coordinate':'u=YJ_half(native/s0)','family':'mixture_student_t','components':components,
            'log_weights':logs,'native_first_scale_s0':response['s0'],'lambda':.5,'model_weight_override':override is not None}


def ensemble_check(saved,expected,c,label='ensemble'):
    c.array(label+'_logweights',saved['log_weights'],expected['log_weights'])
    for key in ('center','sd'):c.array(label+'_expansion_'+key,saved['expansion_transform'][key],expected['expansion_transform'][key])
    c.exact(label+'_actual_label_count',saved['true_label_updates'],expected['true_label_updates'])
    for actual,wanted in zip(saved['models'],expected['models']):
        c.exact(label+'_modelid',actual['id'],wanted['id'])
        for key in ('mean','V'):c.array(label+'_NIG_'+key,actual['state'][key],wanted['state'][key])
        for key in ('alpha','b'):c.number(label+'_NIG_'+key,actual['state'][key],wanted['state'][key])
        c.exact(label+'_component_count',actual['state']['labeled_updates'],wanted['state']['labeled_updates'])


def law_check(saved,expected,c):
    for key in ('coordinate','family','model_weight_override'):c.exact('mixture_law_'+key,saved[key],expected[key])
    c.number('mixture_lambda',saved['lambda'],.5,0,0)
    c.number('mixture_s0',saved['native_first_scale_s0'],expected['native_first_scale_s0'])
    c.array('mixture_logweights',saved['log_weights'],expected['log_weights'])
    for actual,wanted in zip(saved['components'],expected['components']):
        c.exact('mixture_componentid',actual['id'],wanted['id'])
        for key in ('location','scale','df'):c.number('mixture_'+key,actual[key],wanted[key])
    if 'weights_display_only' in saved:c.array('mixture_displayweights',saved['weights_display_only'],np.exp(expected['log_weights']).tolist())


def openloop(warm,current,step):
    result={'models':[],'log_weights':list(warm['log_weights']),'expansion_transform':warm['expansion_transform'],
            'true_label_updates':current['true_label_updates']}
    for initial,actual in zip(warm['models'],current['models']):
        first=initial['state'];alpha=first['alpha']+step/2
        result['models'].append({'id':initial['id'],'state':{'mean':first['mean'],'V':actual['state']['V'],
            'alpha':alpha,'b':(alpha-1)*first['b']/(first['alpha']-1),'labeled_updates':first['labeled_updates']+step}})
    return result
