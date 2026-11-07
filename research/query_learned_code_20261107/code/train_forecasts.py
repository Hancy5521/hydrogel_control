"""One fixed historical OOF/reference fit and four same-information controls.

Science functions run only after query_io actual root/phase registration. All
fitted states are exported as auditable JSON; no opaque pickle is required.
"""
import argparse
from fractions import Fraction as F
import hashlib
import importlib.metadata
import math
from pathlib import Path
import time
import warnings
import query_io as io


def jsonable(value):
    if isinstance(value,F): return str(value)
    if isinstance(value,dict): return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
    return value

def core_modules(p):
    geo=io.load_dependency(p,'cdf_geometry');rc=io.load_dependency(p,'risk_code')
    return geo,rc

def native_modules(p):
    native=io.load_dependency(p,'empirical_native')
    io.load_dependency(p,'simplex_transport')
    views=io.load_dependency(p,'calibration_views')
    baselines=io.load_dependency(p,'baselines')
    return native,views,baselines

def runtime_check(plan):
    expected={'numpy':'2.1.3','scipy':'1.15.3','scikit-learn':'1.6.1','ngboost':'0.5.11'}
    actual={k:importlib.metadata.version(k) for k in expected}
    io.require(actual==expected,'exact pinned numerical/NGBoost runtime required')
    environment=io.read(plan['inputs']['ngboost_environment_receipt']['file'])
    io.require(environment['status']=='PASS_ISOLATED_STABLE_NGBOOST_IMPORT_NO_FIT','verified isolated stable package required')
    for file,digest in environment['installed_source_hashes'].items():
        io.require(io.sha(file)==digest,'installed stable NGBoost source changed')
    return actual

def model_call(folder,name,func):
    """One started event is durable before each actual model fit call."""
    logfile=folder/'MODEL_CALLS.jsonl'
    import json
    with logfile.open('a',encoding='utf8') as f:
        f.write(json.dumps({'actual_utc':io.utc(),'name':name,'status':'STARTED'})+'\n');f.flush()
    try: result=func()
    except Exception as exc:
        with logfile.open('a',encoding='utf8') as f:
            f.write(json.dumps({'actual_utc':io.utc(),'name':name,'status':'FAILED','error_type':type(exc).__name__,'message':str(exc)})+'\n')
        raise
    with logfile.open('a',encoding='utf8') as f:
        f.write(json.dumps({'actual_utc':io.utc(),'name':name,'status':'SUCCEEDED'})+'\n')
    return result

def fit_record(result):
    return {'theta':list(map(str,result.theta)),'proposed_theta':None if result.proposed_theta is None else list(map(str,result.proposed_theta)),
            'accepted':result.accepted,'diagnostics':jsonable(result.diagnostics)}

def empirical_candidate(geo,theta):
    pairs=geo.candidate(theta)
    return {'coordinate':'native','family':'weighted_empirical','support_native':[float(x) for x,w in pairs],
            'weight_fractions':[str(w) for x,w in pairs],'unit':io.UNIT}

def initial_forecasts(p,folder):
    io.require(p['_phase']=='initial_forecasts','registered initial phase required')
    runtime=runtime_check(p);geo,rc=core_modules(p);native,views,baseline=native_modules(p)
    history,transform=io.recover808(p);ids=[r['component_id'] for r in history]
    reserved=io.read(p['inputs']['new_roles']['file'])
    io.require(not set(ids) & set(reserved['confirm_component_ids']+reserved['trial_component_ids']),
               'historical training contaminated permanent future roles')
    io.write(folder/'HISTORY808_RECOVERED.json',{'actual_utc':io.utc(),'rows':history,'transform':transform,
          'native_not_inverted_from_float32':True,'source':'original consumed296 + actual consumed512; computed reports'})
    ordered=sorted(range(808),key=lambda i:(hashlib.sha256((io.RECIPE['oof_hash_prefix']+ids[i]).encode()).hexdigest(),ids[i]))
    fold_for={i:rank%5 for rank,i in enumerate(ordered)}
    io.write(folder/'OOF_ASSIGNMENT_BEFORE_FITS.json',{'actual_utc':io.utc(),'prefix':io.RECIPE['oof_hash_prefix'],
        'rows':[{'component_id':ids[i],'history_index':i,'fold':fold_for[i]} for i in range(808)],
        'fold_sizes':[sum(v==k for v in fold_for.values()) for k in range(5)]})
    oof=[None]*808
    for fold in range(5):
        train=[i for i in range(808) if fold_for[i]!=fold];held=[i for i in range(808) if fold_for[i]==fold]
        laws,state=model_call(folder,'OOF_RF_'+str(fold),lambda:baseline.fit_forest(
            [history[i]['X'] for i in train],[history[i]['y'] for i in train],[history[i]['X'] for i in held],transform))
        state.update(training_component_ids=[ids[i] for i in train],heldout_component_ids=[ids[i] for i in held],fold=fold)
        io.write(folder/('OOF_RF_STATE_'+str(fold)+'.json'),state)
        for i,law in zip(held,laws):oof[i]=law
    io.require(all(law is not None for law in oof),'all808 exactlyonce OOF laws required')
    y=[r['y'] for r in history];global_geos=[geo.geometry(law) for law in oof]
    context_geos=[g.lift(r['X'][7]) for g,r in zip(global_geos,history)]
    global_fit=rc.fit_code(rc.aggregate_Q(global_geos),rc.aggregate_c(global_geos,y),rc.identity_theta(3),808)
    context_fit=rc.fit_code(rc.aggregate_Q(context_geos),rc.aggregate_c(context_geos,y),rc.identity_theta(6),808)
    beta_fit=views.fit_beta(oof,y)
    io.write(folder/'PRIOR_HEADS.json',{'actual_utc':io.utc(),'global':fit_record(global_fit),'context':fit_record(context_fit),
        'Beta':beta_fit,'g':'S+Se+Te atomic fraction fixed before fit, no descriptor search',
        'same_oof808_for_all':True,'head_tasks':3,'optimizer_tasks':{'global':global_fit.diagnostics['optimizer_calls'],
        'context':context_fit.diagnostics['optimizer_calls'],'Beta_fixed_starts':len(beta_fit['starts'])},
        'head_data_are_development_not_independent_validation':True})
    io.write(folder/'OOF808_LAWS.json',{'rows':[{'component_id':ids[i],'law':law} for i,law in enumerate(oof)]})
    reader=io.TrustedReader(p,io.pool(p));contexts=reader.contexts()
    X=[r['X'] for r in history];qy=[r['X'] for r in contexts]
    laws,state=model_call(folder,'FINAL_RF808',lambda:baseline.fit_forest(X,y,qy,transform))
    state.update(training_component_ids=ids,prediction_component_ids=[r['component_id'] for r in contexts])
    io.write(folder/'RF808_STATE.json',state);io.write(folder/'CONTEXTS512.json',{'records':contexts})
    predictions=[];geometry_rows=[]
    for r,law in zip(contexts,laws):
        base=geo.geometry(law);g=base.lift(r['X'][7])
        predictions.append({'component_id':r['component_id'],'F808':law,
             'priorglobal':empirical_candidate(base,global_fit.theta),
             'priorcontext':empirical_candidate(g,context_fit.theta),'Beta':views.apply_beta(law,beta_fit)})
        geometry_rows.append({'component_id':r['component_id'],'g_exact':str(geo.rational(r['X'][7])),
            'Q_exact':jsonable(g.Q),'c_bounds_exact':jsonable(g.c_bounds)})
    io.write(folder/'INITIAL512_LAWS.json',{'records':predictions})
    io.write(folder/'GEOMETRY512.json',{'records':geometry_rows,'truth_free':True})
    manifest={f.name:io.sha(f) for f in folder.iterdir() if f.is_file()}
    io.write(folder/'FROZEN.json',{'status':'INITIAL512_F808_PRIORS_GEOMETRY_FROZEN_BEFORE_CURRENT_Y',
        'actual_utc':io.utc(),'plan_sha256':p['_plan_sha256'],'files_sha256':manifest,'runtime':runtime,
        'actual_model_calls':6,'current_targets_read':0,'confirm_extra_property_X_Y_disclosed_to_models':0,'confirm_identity_composition_metadata_already_accessible':True})

def fit_ngboost(train_X,y_native,prediction_X,transform,baseline):
    import numpy as np
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import CRPScore
    from sklearn.tree import DecisionTreeRegressor
    x,y,query=baseline.standardized(train_X,y_native,prediction_X,transform)
    model=NGBRegressor(Dist=Normal,Score=CRPScore,Base=DecisionTreeRegressor(criterion='friedman_mse',max_depth=3),
        natural_gradient=True,n_estimators=500,learning_rate=.01,minibatch_frac=1.,col_sample=1.,
        verbose=False,tol=1e-4,random_state=20261107,validation_fraction=.1,early_stopping_rounds=None)
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(np.asarray(x,dtype=np.float64),y)
        predicted=model.pred_dist(np.asarray(query,dtype=np.float64))
    location=np.asarray(predicted.loc);scale=np.asarray(predicted.scale)
    io.require(np.isfinite(location).all() and np.isfinite(scale).all() and (scale>0).all(),'finitepositive NGBoostNormal laws required; no floor/fallback')
    def tree_state(m):
        t=m.tree_
        return {'children_left':t.children_left.tolist(),'children_right':t.children_right.tolist(),
                'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value.reshape(-1).tolist(),
                'n_node_samples':t.n_node_samples.tolist(),'weighted_n_node_samples':t.weighted_n_node_samples.tolist()}
    state={'kind':'official_NGBoost_Normal_CRPS_0.5.11','actual_utc':io.utc(),'train_rows':len(y),
        'fit_seconds_descriptive_only':time.perf_counter()-start,'fixed_feature_transform':transform,
        'training_X_standardized_float64':np.asarray(x,dtype=float).tolist(),'training_y_native':np.asarray(y,dtype=float).tolist(),
        'prediction_X_standardized_float64':np.asarray(query,dtype=float).tolist(),
        'init_params':model.init_params.tolist(),'learning_rate':model.learning_rate,
        'scalings':list(map(float,model.scalings)),'col_idxs':[a.tolist() for a in model.col_idxs],
        'base_models':[[tree_state(m) for m in stage] for stage in model.base_models],
        'n_stages_actual':len(model.base_models),'n_estimators_requested':500,
        'prediction_rule':'params=init_params-sum_stage learning_rate*scaling*tree_predictions(X[:,col_idxs]);loc=params0,scale=exp(params1)',
        'tree_input_routing_dtype':'float32 inside sklearn DecisionTreeRegressor.predict',
        'warnings':[{'category':a.category.__name__,'message':str(a.message)} for a in caught],
        'score':'CRPScore','natural_gradient':True,'minibatch_frac':1.,'col_sample':1.,
        'tol':1e-4,'random_state':20261107,'early_stopping_rounds':None}
    laws=[{'coordinate':'native','family':'normal','location':float(m),'scale':float(s),'df':None,'unit':io.UNIT} for m,s in zip(location,scale)]
    return laws,state

def updated_baselines(p,folder):
    io.require(p['_phase']=='updated_baselines','registered updated baseline phase required')
    runtime=runtime_check(p);_,_,baseline=native_modules(p)
    io.load_dependency(p,'crps_tree');crps=io.load_dependency(p,'crps_forest')
    initial=Path(p['inputs']['initial_freeze']['file']);freeze=io.read(initial);io.verify_payloads(initial.parent,freeze['files_sha256'])
    querypin=p['inputs']['query_complete'];query=io.read(querypin['file'])
    io.require(query['status']=='ACTUAL128_QUERIES_AND_FOUR_UPDATES_COMPLETE_BEFORE_REMAINING_Y' and query['initial_freeze_sha256']==io.sha(initial),'128 truequeries completion required')
    io.verify_payloads(Path(querypin['file']).parent,query['files_sha256'])
    known=io.read(initial.parent/'HISTORY808_RECOVERED.json');history=known['rows'];transform=known['transform']
    contexts=io.read(initial.parent/'CONTEXTS512.json')['records'];byid={r['component_id']:r for r in contexts}
    queried=io.read(Path(querypin['file']).parent/'ACTUAL128_TARGETS.json')['records']
    io.require(len(queried)==len({r['component_id'] for r in queried})==128,'actual128 distinctquery labels required')
    history=history+[{'component_id':r['component_id'],'X':byid[r['component_id']]['X'],'y':r['native_target']} for r in queried]
    remaining=query['remaining_ids'];io.require(len(remaining)==384 and set(remaining)==set(byid)-{r['component_id'] for r in queried},'exact common384 remaining set required')
    X=[r['X'] for r in history];y=[r['y'] for r in history];qX=[byid[k]['X'] for k in remaining]
    laws,state=model_call(folder,'RF936',lambda:baseline.fit_forest(X,y,qX,transform))
    models={'RF936':laws};state.update(training_component_ids=[r['component_id'] for r in history],prediction_component_ids=remaining);io.write(folder/'RF936_STATE.json',state)
    laws,state=model_call(folder,'GP936',lambda:baseline.fit_gp(X,y,qX,transform))
    models['GP936']=laws;state.update(training_component_ids=[r['component_id'] for r in history],prediction_component_ids=remaining);io.write(folder/'GP936_STATE.json',state)
    def crps_call():
        model=crps.CRPSForest(n_trees=256,min_leaf=3,max_depth=None,subsample_fraction=.6,seed=20261107).fit(X,y)
        return model.laws(qX,unit=io.UNIT),model.to_state()
    laws,state=model_call(folder,'CRPS936',crps_call)
    models['CRPS936']=laws;state.update(training_component_ids=[r['component_id'] for r in history],prediction_component_ids=remaining);io.write(folder/'CRPS936_STATE.json',state)
    laws,state=model_call(folder,'NGBoost936',lambda:fit_ngboost(X,y,qX,transform,baseline))
    models['NGBoost936']=laws;state.update(training_component_ids=[r['component_id'] for r in history],prediction_component_ids=remaining);io.write(folder/'NGBOOST936_STATE.json',state)
    io.write(folder/'UPDATED_BASELINE_LAWS.json',{'records':[{'component_id':key,**{name:law[i] for name,law in models.items()}} for i,key in enumerate(remaining)],
        'same936actual_labels_all4_methods':True,'runtime':runtime})
    io.write(folder/'COMPLETE.json',{'status':'FOUR_SAME_INFORMATION936_BASELINES_FROZEN_BEFORE_REMAINING_Y','actual_utc':io.utc(),
        'plan_sha256':p['_plan_sha256'],'remaining_ids':remaining,'actual_model_calls':4,
        'files_sha256':{f.name:io.sha(f) for f in folder.iterdir() if f.is_file()},'remaining_Y_read':0})

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True);ap.add_argument('--phase',choices=['initial_forecasts','updated_baselines'],required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    p=io.validate(a.plan,a.approval,a.phase);io.require(not a.output_dir.exists(),'one-shot fit directory already exists; no automatic refit')
    a.output_dir.mkdir(parents=True,exist_ok=False);io.write(a.output_dir/'START.json',{'actual_utc':io.utc(),'plan_sha256':p['_plan_sha256'],'phase':a.phase})
    if a.phase=='initial_forecasts':initial_forecasts(p,a.output_dir)
    else:updated_baselines(p,a.output_dir)
if __name__=='__main__':main()
