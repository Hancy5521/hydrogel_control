"""Single-slot composition library: pure coordinates and generic conjugate components.

Uses only scalar_law/logpredictive/update_nig from the explicitly pinned prior core.
No hardcoded three-H validator is patched. No IO; calls that fit/update require the
external controller's prior registration and actual revealed labels.
"""
import copy
from fractions import Fraction
import math
import re
import numpy as np
from scipy import special
import relation_posterior as core

CORE_SHA256='1d74cde91519ce282c6ebee54e53807a2a79be0cebf19e06221eb4f16e8b7528'
OLD_IDS=('affine','interaction','fullquad')
DICTIONARY=('O_present','f_O','f_H','f_C_plus_N','f_F_Cl_Br_I','f_S_Se_Te','element_count','stoichiometric_entropy')
CONSTANT_REL_TOL=1e-12
VERIFY_ATOL=1e-9
VERIFY_RTOL=1e-9
ELEMENTS=set('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split())


def normalize_logs(logs):
    a=np.asarray(logs,dtype=float)
    if a.ndim!=1 or not 1<=len(a)<=4 or not np.isfinite(a).all():raise ValueError('one to four finite logweights required')
    return (a-special.logsumexp(a)).tolist()


def composition_descriptors(amounts):
    if not isinstance(amounts,dict):raise ValueError('atomic composition mapping required')
    values={}
    for name,value in amounts.items():
        if not isinstance(name,str):raise ValueError('element symbol must be a string')
        if name.startswith('@'):continue
        if name not in ELEMENTS or type(value) is bool:raise ValueError('valid atomic element/count required')
        try:count=Fraction(str(value))
        except (ValueError,ZeroDivisionError,OverflowError) as exc:raise ValueError('finite rational stoichiometry required') from exc
        if count<0:raise ValueError('negative atomic amount invalid')
        if count>0:values[name]=count
    total=sum(values.values(),Fraction(0))
    if total<=0:raise ValueError('nonempty positive composition required')
    fractions={name:float(value/total) for name,value in values.items()}
    if not all(math.isfinite(x) and x>0 for x in fractions.values()):raise ValueError('supported positive fraction range required')
    f=lambda names:math.fsum(fractions.get(name,0.) for name in names)
    return [float('O' in fractions),f(('O',)),f(('H',)),f(('C','N')),f(('F','Cl','Br','I')),
        f(('S','Se','Te')),float(len(fractions)),-math.fsum(x*math.log(x) for x in fractions.values())]


def descriptor_transform(descriptors40):
    d=np.asarray(descriptors40,dtype=float)
    if d.shape!=(40,8) or not np.isfinite(d).all():raise ValueError('40 by8 known descriptor matrix required')
    center=d.mean(axis=0);sample=d.std(axis=0,ddof=1);magnitude=np.maximum(1.,np.max(np.abs(d),axis=0))
    constant=sample<=CONSTANT_REL_TOL*magnitude;scale=np.where(constant,1.,sample)
    return {'center':center.tolist(),'sd':scale.tolist(),'raw_sample_SD':sample.tolist(),
        'constant_or_numerically_constant':constant.tolist(),'constant_relative_tol':CONSTANT_REL_TOL,'fitted_rows':40,'ddof':1}


def standard_descriptors(transform,descriptors):
    d=np.asarray(descriptors,dtype=float);center=np.asarray(transform['center']);sd=np.asarray(transform['sd'])
    if d.shape[-1:]!=(8,) or not np.isfinite(d).all() or center.shape!=(8,) or sd.shape!=(8,) or (sd<=0).any():raise ValueError('finite eight-descriptor input and fixed transform required')
    return (d-center)/sd


def yj(x):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all():raise ValueError('finite native input required')
    out=np.empty_like(x);positive=x>=0
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        out[positive]=2*x[positive]/(np.sqrt(1+x[positive])+1)
        out[~positive]=-np.expm1(1.5*np.log1p(-x[~positive]))/1.5
    return out


def fit_coordinates(native32):
    native=np.asarray(native32,dtype=float)
    if native.shape!=(32,3) or not np.isfinite(native).all():raise ValueError('exactly32 finite training rows required')
    first=native.std(axis=0,ddof=1)
    if (first<=0).any():raise ValueError('nonconstant training native fields required; no response scale floor')
    u=yj(native/first);center=u.mean(axis=0);sd=u.std(axis=0,ddof=1)
    if not np.isfinite(sd).all() or (sd<=0).any():raise ValueError('nonconstant finite training transformed fields required')
    z=(u-center)/sd;terms=np.column_stack((z[:,0]*z[:,1],z[:,0]**2,z[:,1]**2))
    expansion={'center':terms.mean(axis=0).tolist(),'sd':terms.std(axis=0,ddof=1).tolist()}
    if min(expansion['sd'])<=0:raise ValueError('nonconstant train-only expansion required')
    return {'native_first_scale_s0':first.tolist(),'u_center_warm':center.tolist(),'u_sample_SD_warm':sd.tolist()},expansion


def proxies_z(transform,native_proxy):
    x=np.asarray(native_proxy,dtype=float)
    if x.shape[-1:]!=(2,) or not np.isfinite(x).all():raise ValueError('finite native dual proxies required')
    return (yj(x/np.asarray(transform['native_first_scale_s0'])[:2])-np.asarray(transform['u_center_warm'])[:2])/np.asarray(transform['u_sample_SD_warm'])[:2]


def response_spec(transform):
    return {'center':float(transform['u_center_warm'][2]),'sd':float(transform['u_sample_SD_warm'][2]),'s0':float(transform['native_first_scale_s0'][2])}


def targets_z(transform,y):
    r=response_spec(transform)
    return (yj(np.asarray(y,dtype=float)/r['s0'])-r['center'])/r['sd']


def basis(kind,x,expansion,descriptor_z=None,index=None):
    x=np.asarray(x,dtype=float)
    if x.shape!=(2,) or not np.isfinite(x).all():raise ValueError('finite modeled context required')
    if kind=='descriptor':
        d=np.asarray(descriptor_z,dtype=float)
        if d.shape!=(8,) or index not in range(8):raise ValueError('registered single descriptor index required')
        return np.asarray([1.,x[0],x[1],d[index]])
    terms=np.asarray([x[0]*x[1],x[0]**2,x[1]**2]);terms=(terms-np.asarray(expansion['center']))/np.asarray(expansion['sd'])
    all_features=np.concatenate(([1.],x,terms));dimension={'affine':3,'interaction':4,'fullquad':6}.get(kind)
    if dimension is None or not np.isfinite(all_features).all():raise ValueError('registered basis required')
    return all_features[:dimension]


def prior(dimension):return {'mean':[0.]*dimension,'V':np.eye(dimension).tolist(),'alpha':3.,'b':1.5,'labeled_updates':0}


def fit_component(kind,x,zY,expansion,d=None,index=None):
    x=np.asarray(x,dtype=float);zY=np.asarray(zY,dtype=float)
    if x.ndim!=2 or x.shape[1]!=2 or zY.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(zY).all():raise ValueError('matching finite modeled rows/targets required')
    if kind=='descriptor' and np.asarray(d).shape!=(len(x),8):raise ValueError('matching descriptor rows required')
    dimension={'affine':3,'interaction':4,'fullquad':6,'descriptor':4}[kind];state=prior(dimension);logev=0.;increments=[]
    for i,(context,target) in enumerate(zip(x,zY)):
        phi=basis(kind,context,expansion,None if d is None else d[i],index)
        value=core.logpredictive(state,phi,float(target));increments.append(value);logev+=value
        state=core.update_nig(state,phi,float(target))
    return {'kind':kind,'descriptor_index':index,'state':state},float(logev),increments


def fit_old_three(native32):
    transform,expansion=fit_coordinates(native32);native=np.asarray(native32,dtype=float)
    x=proxies_z(transform,native[:,:2]);zY=targets_z(transform,native[:,2]);models={};ev={}
    for kind in OLD_IDS:models[kind],ev[kind],_=fit_component(kind,x,zY,expansion)
    return {'transform':transform,'expansion_transform':expansion,'components':models,
        'component_ids':list(OLD_IDS),'log_weights':normalize_logs([ev[k] for k in OLD_IDS]),'absolute_log_evidence':ev}


def _same_state(actual,expected):
    for key in ('mean','V','alpha','b'):
        if not np.allclose(actual[key],expected[key],atol=VERIFY_ATOL,rtol=VERIFY_RTOL):raise ValueError('old40 checkpoint mismatch: '+key)
    if actual['labeled_updates']!=expected['labeled_updates']:raise ValueError('old40 update count mismatch')


def initialize_growth(history_native40,descriptor40,old_snapshot,old_final3,proposal):
    native=np.asarray(history_native40,dtype=float)
    if native.shape!=(40,3) or not np.isfinite(native).all():raise ValueError('chronological oldwarm32 plus8 actual queried rows required')
    transform=copy.deepcopy(old_snapshot['transform']);expansion=copy.deepcopy(old_snapshot['warm_ensemble']['expansion_transform'])
    if not np.array_equal(np.asarray(descriptor40),np.asarray(proposal['prepared']['history_descriptors40'])):raise ValueError('old40descriptor order changed after proposal')
    if not np.array_equal(native[:,:2],np.asarray(proposal['prepared']['history_proxy_native40'])):raise ValueError('old40row order changed after anchor')
    dt=copy.deepcopy(proposal['prepared']['descriptor_transform']);d=standard_descriptors(dt,descriptor40)
    x=proxies_z(transform,native[:,:2]);zY=targets_z(transform,native[:,2]);components={};evidence={};checks={}
    oldmap={m['id']:m['state'] for m in old_final3['models']}
    for kind in OLD_IDS:
        fitted,evidence[kind],_=fit_component(kind,x,zY,expansion);_same_state(oldmap[kind],fitted['state'])
        components[kind]={'kind':kind,'descriptor_index':None,'state':copy.deepcopy(oldmap[kind])}
    normalized_old=normalize_logs([evidence[k] for k in OLD_IDS])
    if not np.allclose(normalized_old,normalize_logs(old_final3['log_weights']),atol=VERIFY_ATOL,rtol=VERIFY_RTOL):raise ValueError('old40 evidence odds mismatch')
    selected=proposal['selected_index'];anchor=proposal['prepared']['anchor_index']
    if selected not in range(8) or anchor not in range(8):raise ValueError('no identifiable growth feature: do not fabricate Hnew')
    components['new'],evidence['new'],_=fit_component('descriptor',x,zY,expansion,d,selected)
    alias=selected==anchor
    if alias:anchor_id='new'
    else:
        anchor_id='anchor';components['anchor'],evidence['anchor'],_=fit_component('descriptor',x,zY,expansion,d,anchor)
    oldids=list(OLD_IDS);newids=oldids+['new']
    return {'transform':transform,'expansion_transform':expansion,'descriptor_transform':dt,
        'components':components,'libraries':{'old3_BMA':{'component_ids':oldids,'log_weights':normalize_logs([evidence[k]-math.log(3) for k in oldids])},
            'grown4_BMA':{'component_ids':newids,'log_weights':normalize_logs([evidence[k]-math.log(4) for k in newids])}},
        'new_id':'new','anchor_id':anchor_id,'selected_index':selected,'anchor_index':anchor,'new_anchor_exact_alias':alias,
        'absolute_history40_log_evidence':evidence,'old40_states_preserved_after_verification':True,
        'old3_relative_odds_preserved':True,'conditional_data_selected_initialization_not_selection_valid_Bayesfactor':True,
        'history_rows_seen':40,'fresh_rows_seen':0,'proposal':copy.deepcopy(proposal)}


def update_batch(suite,native_proxies,descriptors,native_y):
    result=copy.deepcopy(suite);x=proxies_z(result['transform'],native_proxies);d=standard_descriptors(result['descriptor_transform'],descriptors)
    z=targets_z(result['transform'],native_y)
    if x.ndim!=2 or d.shape!=(len(x),8) or z.shape!=(len(x),):raise ValueError('same-sized disclosed complete batch required')
    records=[]
    for row,(context,target) in enumerate(zip(x,z)):
        increments={};newcomponents={}
        for name,component in result['components'].items():
            phi=basis(component['kind'],context,result['expansion_transform'],d[row],component['descriptor_index'])
            increments[name]=core.logpredictive(component['state'],phi,float(target))
            newcomponents[name]=dict(component,state=core.update_nig(component['state'],phi,float(target)))
        for library in result['libraries'].values():library['log_weights']=normalize_logs([a+increments[name] for a,name in zip(library['log_weights'],library['component_ids'])])
        result['components']=newcomponents;records.append({'row_in_disclosed_batch':row,'old_predictive_loglikelihood':increments,
            'library_log_weights':{name:list(lib['log_weights']) for name,lib in result['libraries'].items()}})
    result['fresh_rows_seen']+=len(x);result['history_rows_seen']+=len(x)
    return result,{'records':records,'rows_updated_after_whole_batch_reveal':len(x),'unique_components_updated_each_row':len(result['components']),
        'anchor_alias_not_duplicate_update':result['new_anchor_exact_alias'],'no_generated_labels':True}


def law_from_components(components,ids,logs,transform,expansion,x,descriptor_z=None):
    response=response_spec(transform);items=[]
    for name in ids:
        component=components[name];phi=basis(component['kind'],x,expansion,descriptor_z,component['descriptor_index'])
        predicted=core.scalar_law(component['state'],phi)
        items.append({'id':name,'location':response['center']+response['sd']*predicted['mu'],
            'scale':response['sd']*predicted['scale'],'df':predicted['nu']})
    base={'coordinate':'u=YJ_half(native/s0)','native_first_scale_s0':response['s0'],'lambda':.5,
        'unit':'source bulk GPa attribution, aggregation qualified'}
    if len(items)==1:return base|{'family':'student_t','location':items[0]['location'],'scale':items[0]['scale'],'df':items[0]['df']}
    return base|{'family':'mixture_student_t','components':items,'log_weights':normalize_logs(logs)}


def law(suite,method_id,native_proxy,descriptor):
    if method_id in suite['libraries']:
        library=suite['libraries'][method_id];ids=library['component_ids'];logs=library['log_weights']
    else:
        name={'H_new':suite['new_id'],'H_anchor':suite['anchor_id'],'affine':'affine'}.get(method_id)
        if name is None:raise ValueError('registered NIG method required')
        ids=[name];logs=[0.]
    x=proxies_z(suite['transform'],native_proxy);d=standard_descriptors(suite['descriptor_transform'],descriptor)
    return law_from_components(suite['components'],ids,logs,suite['transform'],suite['expansion_transform'],x,d)


def baseline_transform(old_snapshot,descriptor40):
    dt=descriptor_transform(descriptor40);native=old_snapshot['native_reference_feature_transform']
    return {'center':list(native['center'])+dt['center'],'sd':list(native['sd'])+dt['sd'],
        'feature_ids':['native_e_form','native_gap_opt']+list(DICTIONARY),'proxy_scale_origin':'frozen old32native',
        'descriptor_scale_origin':'old40knowncomposition','descriptor_constant_flags':dt['constant_or_numerically_constant']}
