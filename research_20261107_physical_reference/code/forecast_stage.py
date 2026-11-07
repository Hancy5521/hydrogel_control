"""Produce one complete physical-reference round before its first true target.

Initial448 ionic OOF is done once. Later rounds append only past, frozen
reference/true-report pairs. All full scoring belongs to the root score stage.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
from pathlib import Path
import time
import warnings
import task_io as io
import numerical_adapter as numerical


def fit_reference(modules, kind, X, y, query, transform, residual=False):
    """Frozen RF/GP recipes; the new RF snapshot also records bootstrap draws."""
    import numpy as np
    reference = modules['reference_baselines']
    unit = io.RESIDUAL_UNIT if residual else io.UNIT
    if kind == 'GP':
        laws,state = reference.fit_gp(X,y,query,transform)
    else:
        from sklearn.ensemble import RandomForestRegressor
        x,target,prediction = reference.standardized(X,y,query,transform)
        x,prediction = np.asarray(x,dtype=np.float32),np.asarray(prediction,dtype=np.float32)
        model = RandomForestRegressor(n_estimators=256,min_samples_leaf=3,max_features=1.0,
                 bootstrap=True,criterion='squared_error',random_state=20261106,n_jobs=2,max_depth=None)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always');model.fit(x,target)
        seconds = time.perf_counter()-start
        train_leaf,query_leaf = model.apply(x),model.apply(prediction)
        laws = []
        for i in range(len(prediction)):
            weights = [Fraction() for _ in target]
            for tree in range(256):
                members = np.flatnonzero(train_leaf[:,tree]==query_leaf[i,tree])
                if not len(members): raise ArithmeticError('no original member of predicted RF leaf')
                mass = Fraction(1,256*len(members))
                for j in members: weights[int(j)] += mass
            if sum(weights,Fraction()) != 1: raise AssertionError('exact RF mass simplex required')
            laws.append({'coordinate':'native','family':'weighted_empirical','support_native':target.tolist(),
                         'weight_fractions':list(map(str,weights)),'unit':unit})
        trees = []
        for index,child in enumerate(model.estimators_):
            t = child.tree_
            trees.append({'tree_id':index,'children_left':t.children_left.tolist(),'children_right':t.children_right.tolist(),
                          'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'n_node_samples':t.n_node_samples.tolist(),
                          'weighted_n_node_samples':t.weighted_n_node_samples.tolist(),'training_tree_value_native':t.value.reshape(-1).tolist(),
                          'original_train_leaf_ids':train_leaf[:,index].tolist(),'prediction_leaf_ids':query_leaf[:,index].tolist()})
        state = {'actual_utc':io.utc(),'kind':'fixed_all_original_leaf_RF_state','fit_seconds_descriptive_only':seconds,
                 'train_rows':len(target),'prediction_rows':len(prediction),'feature_count':10,
                 'X_train_standardized_then_float32':x.astype(float).tolist(),
                 'X_prediction_standardized_then_float32':prediction.astype(float).tolist(),'y_train_native':target.tolist(),
                 'leaf_weights_include_OOB_original_rows':True,'bootstrap_multiplicity_in_CDF_weights':False,
                 'trees':trees,'fit_warnings_all_categories':reference._warnings(captured),
                 'tree_random_state_seeds':[int(t.random_state) for t in model.estimators_],
                 'bootstrap_draw_indices':[indices.tolist() for indices in model.estimators_samples_],
                 'bootstrap_snapshot_scope':'sklearn estimators_samples_ deterministic reconstruction from fitted tree seeds',
                 'scope':'QRF-inspired all-original-leaf variant, not official QRF software'}
    for law in laws: law['unit'] = unit
    state.update(target_field='derived_ionic_residual' if residual else 'poly_total',target_unit=unit,
                 feature_ids=list(io.FEATURE_IDS),fixed_feature_transform=transform)
    return laws,state


def fit_crps(modules,X,y,query,transform):
    reference = modules['reference_baselines']
    x,target,prediction = reference.standardized(X,y,query,transform)
    start = time.perf_counter()
    model = modules['crps_forest'].CRPSForest(n_trees=256,min_leaf=3,max_depth=None,
                                             subsample_fraction=.6,seed=20261107).fit(x.tolist(),target.tolist())
    seconds = time.perf_counter()-start
    laws,state = model.laws(prediction.tolist(),unit=io.RESIDUAL_UNIT),model.to_state()
    state.update(actual_utc=io.utc(),fit_seconds_descriptive_only=seconds,train_rows=len(y),prediction_rows=len(query),
                 target_field='derived_ionic_residual',target_unit=io.RESIDUAL_UNIT,feature_ids=list(io.FEATURE_IDS),
                 fixed_feature_transform=transform,X_prediction_standardized_float64=prediction.tolist(),
                 source='independent classical CRPS-CART implementation; not official author software')
    return laws,state


def residual_of(row):
    value = float(row['native_target']-row['known_electronic'])
    if not __import__('math').isfinite(value): raise ArithmeticError('nonrepresentable residual; no clipping')
    return value


def ionic_oof(modules,training,transform,folder):
    n = len(training)
    if n != 448: raise ValueError('initial448 ionic OOF once only')
    identities = [r['identity'] for r in training]
    X,y = [io.feature_row(r) for r in training],[residual_of(r) for r in training]
    order = sorted(range(n),key=lambda i:(hashlib.sha256(('physical_reference_ionic_oof_v1|'+identities[i]['component_id']).encode()).hexdigest(),identities[i]['component_id']))
    folds = {index:rank%5 for rank,index in enumerate(order)}
    io.write(folder/'OOF_ASSIGNMENT448.json',{'actual_utc':io.utc(),'hash_prefix':'physical_reference_ionic_oof_v1|',
             'fold_sizes':[sum(v==j for v in folds.values()) for j in range(5)],
             'records':[{'row':i,'component_id':r['component_id'],'fold':folds[i]} for i,r in enumerate(identities)]})
    calibration,fit_index = [None]*n,[]
    for fold in range(5):
        held = [i for i in range(n) if folds[i]==fold];train = [i for i in range(n) if folds[i]!=fold]
        laws,state = fit_reference(modules,'RF',[X[i] for i in train],[y[i] for i in train],[X[i] for i in held],transform,True)
        state.update(fold=fold,training_component_ids=[identities[i]['component_id'] for i in train],
                     heldout_component_ids=[identities[i]['component_id'] for i in held])
        name = 'RF_IONIC_OOF_STATE_'+str(fold)+'.json';io.write(folder/name,state)
        fit_index.append({'kind':'ionicRF_OOF','fold':fold,'state_file':name,'state_sha256':io.sha(folder/name),
                          'train_rows':len(train),'prediction_rows':len(held),'model_fit_type':'sklearn'})
        for i,law in zip(held,laws):
            exact = Fraction.from_float(float(training[i]['native_target']))-Fraction.from_float(float(training[i]['known_electronic']))
            calibration[i] = {'identity':identities[i],'law':law,'native_target':y[i],
                              'bounded_descriptors8':training[i]['bounded_descriptors8'],'mode':'OOF_EXCLUDED_WHOLE_FOLD',
                              'fold':fold,'reference_state_file':name,'reference_state_sha256':io.sha(folder/name),
                              'derived_residual_roundoff_exact_fraction':str(Fraction.from_float(y[i])-exact)}
    if any(r is None for r in calibration): raise AssertionError('all448 OOF predictions required')
    io.write(folder/'IONIC_OOF_RECORDS448.json',{'records':calibration,'new_measurements':0,
             'label_kind':'derived existing total-minus-known-electronic; OOF for calibration, not independent test'})
    return calibration,fit_index


def continued_history(plan,previous):
    forecast = previous/'forecast';verified = previous/'verification'
    close = io.read(verified/'CLOSED_ROUND.json')
    expected_round = plan['_current_round']-1
    freeze = forecast/'ALL1280_FORECASTS_FROZEN.json'
    if (close['status']!='ROUND_CLOSED_WITH_ALL128_TRUE_TARGETS' or close['round']!=expected_round
        or close['plan_sha256']!=plan['_plan_sha256'] or close['forecast_freeze_sha256']!=io.sha(freeze)):
        raise ValueError('matching completed previous round required before expansion')
    io.verify_payloads(verified,close['files_sha256'])
    closed_freeze = io.read(freeze);io.verify_payloads(forecast,closed_freeze['files_sha256'])
    if closed_freeze['round'] != expected_round or closed_freeze['plan_sha256'] != plan['_plan_sha256']:
        raise ValueError('wrong preceding forecast')
    training = io.read(forecast/'TRAINING_HISTORY_BEFORE.json')['records']
    calibrations = io.read(forecast/'CALIBRATION_HISTORY_BEFORE.json')
    truth = io.read(verified/'ALL128_TARGETS_DISCLOSED.json')
    if (truth['plan_sha256']!=plan['_plan_sha256'] or truth['forecast_freeze_sha256']!=io.sha(freeze)
        or len(truth['records'])!=128): raise ValueError('full previous disclosed target closure required')
    prediction = io.read(forecast/'FORECASTS1280.json')['records']
    references = {(r['method'],r['component_id']):r for r in prediction if r['method'] in ('directRF','ionicRF')}
    if len(references)!=256: raise ValueError('256 previous reference laws required')
    seen = {r['identity']['component_id'] for r in training}
    for row in truth['records']:
        key = row['identity']['component_id']
        if key in seen: raise ValueError('a source component cannot enter training twice')
        direct,ionic = references[('directRF',key)],references[('ionicRF',key)]
        record = {'identity':row['identity'],'native_target':row['native_target'],
                  'raw_known_proxies2':direct['raw_known_proxies2'],'known_electronic':direct['known_electronic'],
                  'bounded_descriptors8':direct['bounded_descriptors8']}
        training.append(record);seen.add(key)
        common = {'identity':row['identity'],'bounded_descriptors8':record['bounded_descriptors8'],
                  'mode':'PRETARGET_FROZEN_REFERENCE','round':expected_round,
                  'reference_forecasts_sha256':io.sha(forecast/'FORECASTS1280.json'),
                  'reference_freeze_sha256':io.sha(freeze)}
        calibrations['direct'].append(dict(common,law=direct['law'],native_target=row['native_target']))
        residual = residual_of(record)
        exact = Fraction.from_float(float(row['native_target']))-Fraction.from_float(float(record['known_electronic']))
        calibrations['ionic'].append(dict(common,law=ionic['law']['residual_law'],native_target=residual,
                    derived_residual_roundoff_exact_fraction=str(Fraction.from_float(residual)-exact)))
    return training,calibrations,io.read(forecast/'FIXED_TRANSFORM.json'),{
           'previous_round_closed_file':str((verified/'CLOSED_ROUND.json').resolve()),
           'previous_round_closed_sha256':io.sha(verified/'CLOSED_ROUND.json')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True)
    ap.add_argument('--roles',required=True);ap.add_argument('--round',type=int,choices=(1,2,3),required=True)
    ap.add_argument('--previous-round-dir',type=Path);ap.add_argument('--output-dir',type=Path,required=True)
    a = ap.parse_args();plan = io.validate(a.plan,a.approval,'produce_round_forecasts');roles = io.load_roles(plan,a.roles)
    if (a.round==1) != (a.previous_round_dir is None): raise ValueError('previous-round-dir required only for rounds2/3')
    if a.output_dir.exists(): raise ValueError('never overwrite a forecast attempt')
    plan['_current_round'] = a.round
    modules = io.load_modules(plan);folder = a.output_dir;folder.mkdir(parents=True,exist_ok=False)
    io.write(folder/'START.json',{'actual_utc':io.utc(),'round':a.round,'plan_sha256':plan['_plan_sha256'],
             'source_sha256':io.sources(),'current_target_disclosures':0})
    try:
        fits = [];predecessor = None
        if a.round==1:
            training,direct,transform = io.initial_history(plan)
            ionic,fits = ionic_oof(modules,training,transform,folder)
            calibration = {'direct':direct,'ionic':ionic,'old_CDF_never_replaced_by_current_in_sample_reference':True}
        else:
            training,calibration,transform,predecessor = continued_history(plan,a.previous_round_dir.resolve())
        expected = 448+128*(a.round-1)
        if len(training)!=expected or any(len(calibration[k])!=expected for k in ('direct','ionic')):
            raise ValueError('exact common prior-history lengths required')
        if len({r['identity']['component_id'] for r in training})!=expected: raise ValueError('whole-group training uniqueness')
        reader = io.TrustedReader(plan,roles,round_index=a.round);contexts = reader.contexts()
        io.write(folder/'TRAINING_HISTORY_BEFORE.json',{'records':training,'all_targets_previously_revealed':True})
        io.write(folder/'CALIBRATION_HISTORY_BEFORE.json',calibration)
        io.write(folder/'FIXED_TRANSFORM.json',transform);io.write(folder/'CONTEXTS128.json',contexts)
        direct_fit = numerical.fit_repair(modules,calibration['direct'],5)
        ionic_views = numerical.fit_fixed_views(modules,calibration['ionic'])
        beta = modules['calibration_views'].fit_beta([r['law'] for r in calibration['ionic']],
                                                    [r['native_target'] for r in calibration['ionic']])
        io.write(folder/'STATE_BEFORE.json',{'actual_utc':io.utc(),'direct_auto':direct_fit,'ionic_views':ionic_views,
                 'ionic_beta':beta,'fixed_auto_g':5,'anchor_g':0,'training_rows':expected,
                 'calibration_history_sha256':io.sha(folder/'CALIBRATION_HISTORY_BEFORE.json')})
        flat = [(role,row) for role in io.ROLES for row in contexts[role]]
        X,y = [io.feature_row(r) for r in training],[r['native_target'] for r in training]
        residual = [residual_of(r) for r in training];query = [io.feature_row(r) for _,r in flat]
        direct,direct_state = fit_reference(modules,'RF',X,y,query,transform)
        ionic,ionic_state = fit_reference(modules,'RF',X,residual,query,transform,True)
        gp,gp_state = fit_reference(modules,'GP',X,y,query,transform)
        crps,crps_state = fit_crps(modules,X,residual,query,transform)
        ids = [r['identity']['component_id'] for r in training];query_ids = [r['identity']['component_id'] for _,r in flat]
        for kind,name,state in (('directRF','RF_DIRECT_STATE.json',direct_state),('ionicRF','RF_IONIC_STATE.json',ionic_state),
                                ('GP10total','GP_TOTAL_STATE.json',gp_state),('CRPSRFionic','CRPS_IONIC_STATE.json',crps_state)):
            state.update(training_component_ids=ids,prediction_component_ids=query_ids)
            io.write(folder/name,state);fits.append({'kind':kind,'state_file':name,'state_sha256':io.sha(folder/name),
                 'train_rows':expected,'prediction_rows':128,'model_fit_type':'independent_CRPS_forest' if kind=='CRPSRFionic' else 'sklearn'})
        forecasts,bounds = [],[];frame = modules['affine_reference']
        for i,(role,row) in enumerate(flat):
            offset = row['known_electronic'];g = row['bounded_descriptors8'];identity = row['identity']
            ionic_pred = {'ionicRF':ionic[i],'ionicBeta':modules['calibration_views'].apply_beta(ionic[i],beta),
                          'ionicGlobal':numerical.corrected(modules,ionic[i],ionic_views['global'],g),
                          'ionicAnchor':numerical.corrected(modules,ionic[i],ionic_views['anchor'],g),
                          'ionicAuto':numerical.corrected(modules,ionic[i],ionic_views['auto'],g),'CRPSRFionic':crps[i],
                          'electronic_delta':{'coordinate':'native','family':'weighted_empirical','support_native':[0.],
                                              'weight_fractions':['1'],'unit':io.RESIDUAL_UNIT}}
            predicted = {'directRF':direct[i],'directAuto':numerical.corrected(modules,direct[i],direct_fit,g),'GP10total':gp[i]}
            for method,law in ionic_pred.items():
                predicted[method] = frame.affine_law(law,offset);predicted[method]['unit'] = io.UNIT
            for method in io.METHODS:
                forecasts.append({'role':role,'method':method,'component_id':identity['component_id'],'identity':identity,
                         'known_electronic':offset,'raw_known_proxies2':row['raw_known_proxies2'],
                         'bounded_descriptors8':g,'law':predicted[method]})
            envelope = frame.gap_envelope(predicted['ionicAuto'],predicted['ionicRF'])
            lower,upper = Fraction(envelope['lower_exact_fraction']),Fraction(envelope['upper_exact_fraction'])
            knot_gaps = dict(zip(map(Fraction,envelope['knots_exact_fraction']),
                                 map(Fraction,envelope['knot_gaps_exact_fraction'])))
            expectation = sum((mass*abs(knot_gaps[value]) for value,mass in frame.atoms(predicted['ionicRF'])),Fraction())
            bounds.append({'role':role,'component_id':identity['component_id'],'envelope':envelope,
                           'width_exact_fraction':str(upper-lower),'amplitude_exact_fraction':str(max(abs(lower),abs(upper))),
                           'reference_expected_abs_gap_exact_fraction':str(expectation)})
        io.write(folder/'FORECASTS1280.json',{'actual_utc':io.utc(),'round':a.round,'records':forecasts,'current_true_Y_disclosures':0})
        io.write(folder/'PRIMARY_GAP_BOUNDS128.json',{'round':a.round,'records':bounds,'domain':'all real outcomes of stored exact affine laws'})
        certificate = [r for r in bounds if r['role']=='certificate']
        order = {name:[r['component_id'] for r in sorted(certificate,key=lambda r:(-Fraction(r[field]),r['component_id']))]
                 for name,field in (('width','width_exact_fraction'),('amplitude','amplitude_exact_fraction'),
                                    ('reference_expected_abs_gap','reference_expected_abs_gap_exact_fraction'))}
        order['metadata_hash'] = sorted([r['component_id'] for r in certificate],key=lambda key:(hashlib.sha256(('physical_reference_label_order_v1|'+key).encode()).hexdigest(),key))
        io.write(folder/'QUERY_ORDERS.json',{'round':a.round,'certificate':order,'all_orders_fixed_before_current_Y':True,
                 'width_minimizes_remaining_interval_width_not_universal_sign_stopping_time':True})
        io.write(folder/'FIT_INDEX.json',{'records':fits,'actual_model_training_calls':len(fits),
                 'sklearn_fit_calls':sum(r['model_fit_type']=='sklearn' for r in fits),'independent_CRPS_forest_fits':1,
                 'CRPS_trees':256,'calibration_optimization_calls_separate':5})
        if io.sources()!=plan['source_sha256']: raise ValueError('source changed before forecast freeze')
        payload = {p.name:io.sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix=='.json'}
        io.write(folder/'ALL1280_FORECASTS_FROZEN.json',{'status':'ALL1280_FORECASTS_FROZEN_NO_FUTURE_TARGET_DISCLOSURE',
                 'actual_utc':io.utc(),'round':a.round,'plan_sha256':plan['_plan_sha256'],'source_sha256':io.sources(),
                 'roles_sha256':io.sha(a.roles),'methods':list(io.METHODS),'files_sha256':payload,'prediction_count':1280,
                 'training_rows':expected,'actual_model_training_calls':len(fits),'previous_round':predecessor,
                 'current_Y_disclosed':0,'new_wet_experiments':0,'new_DFT_calculations':0})
        print({'status':'ALL1280_FROZEN','round':a.round,'training_rows':expected,'model_training_calls':len(fits)})
    except Exception as problem:
        io.write(folder/'FAILED_STOP.json',{'actual_utc':io.utc(),'type':type(problem).__name__,'message':str(problem),
                 'no_same_cohort_posthoc_rescue':True,'round':a.round});raise


if __name__=='__main__':main()
