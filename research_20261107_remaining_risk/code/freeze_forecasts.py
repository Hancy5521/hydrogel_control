"""Replay one saved RF296 and its saved calibration on a new reserved pool.

No estimator fitting, optimizer, new-target access or old192 scoring.
"""
import argparse
from fractions import Fraction
import math
from pathlib import Path
import struct
import pilot_io as io


def float32(value):
    try:result=struct.unpack('<f',struct.pack('<f',float(value)))[0]
    except OverflowError as error:raise ArithmeticError('query float32 not representable') from error
    if not math.isfinite(result):raise ArithmeticError('nonfinite forest query coordinate')
    return result


def replay(state,contexts):
    if state['train_rows']!=296 or len(state['trees'])!=256:raise ValueError('exact frozenRF296 required')
    transform=state['fixed_feature_transform'];center,sd=transform['center'],transform['sd']
    if len(center)!=10 or len(sd)!=10 or any(not math.isfinite(v) or v<=0 for v in sd):raise ValueError('frozen10D feature transform')
    y=list(map(float,state['y_train_native']));bytree=[]
    if len(y)!=296 or not all(math.isfinite(v) for v in y):raise ValueError('saved296 real training labels required')
    for tree in state['trees']:
        groups={}
        for row,leaf in enumerate(tree['original_train_leaf_ids']):groups.setdefault(leaf,[]).append(row)
        bytree.append(groups)
    laws=[];routes=[]
    for row in contexts:
        raw=row['native_proxies']+row['raw_descriptors8'];query=[float32((v-c)/s) for v,c,s in zip(raw,center,sd)]
        weights=[Fraction() for _ in y];leaves=[]
        for tree,groups in zip(state['trees'],bytree):
            node=0
            while tree['children_left'][node]!=-1:
                feature=tree['feature'][node]
                node=tree['children_left'][node] if query[feature]<=tree['threshold'][node] else tree['children_right'][node]
            members=groups.get(node)
            if not members:raise ArithmeticError('no original training member at reached leaf')
            mass=Fraction(1,256*len(members))
            for j in members:weights[j]+=mass
            leaves.append(node)
        if sum(weights,Fraction())!=1:raise AssertionError('unit RF probability required')
        laws.append({'family':'weighted_empirical','coordinate':'native','unit':io.UNIT,
                     'support_native':y,'weight_fractions':list(map(str,weights))})
        routes.append({'component_id':row['component_id'],'query_standardized_float32':query,'leaf_ids':leaves})
    return laws,routes


def curvature(frame,F,G):
    a,b=frame.atoms(F),frame.atoms(G);am,bm=dict(a),dict(b);knots=sorted(set(am)|set(bm))
    difference=Fraction();result=Fraction()
    for j,x in enumerate(knots[:-1]):
        difference+=am.get(x,Fraction())-bm.get(x,Fraction())
        result+=(knots[j+1]-x)*difference*difference
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True)
    ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    plan=io.validate(a.plan,a.approval,'freeze_forecasts');identities=io.pool(plan)
    if a.output_dir.exists():raise ValueError('never overwrite a forecast attempt')
    empirical=io.load_dependency(plan,'empirical_native');frame=io.load_dependency(plan,'affine_reference')
    state=io.read(plan['inputs']['frozen_rf']['file']);auto=io.read(plan['inputs']['frozen_auto']['file'])['views']['auto']
    selection=io.read(plan['inputs']['descriptor_selection']['file'])
    if (auto['kind']!='context' or auto['status']!='CONVERGED_KKT' or auto['descriptor_index']!=5 or selection['auto_index']!=5):
        raise ValueError('exact previously selected/fitted g5 calibration required')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    io.write(a.output_dir/'START.json',{'actual_utc':io.utc(),'plan_sha256':plan['_plan_sha256'],'source_sha256':io.sources(),
             'new_fits':0,'new_optimizer_calls':0,'new_target_access':0})
    try:
        reader=io.TrustedReader(plan,identities);contexts=reader.contexts();F,routes=replay(state,contexts)
        records,geometry=[],[]
        for row,f in zip(contexts,F):
            g=empirical.apply_spline(f,auto['theta0'],auto['theta1'],row['bounded_descriptors8'][5])
            envelope=frame.gap_envelope(g,f);B=curvature(frame,f,g)
            lo,hi=Fraction(envelope['lower_exact_fraction']),Fraction(envelope['upper_exact_fraction'])
            if not lo<=-B<=B<=hi:raise ArithmeticError('proper-score moment identities outside exact bounds')
            records.append({'component_id':row['component_id'],'identity':row['identity'],
                            'native_proxies':row['native_proxies'],'bounded_descriptors8':row['bounded_descriptors8'],
                            'F_law':f,'G_law':g})
            geometry.append({'component_id':row['component_id'],'envelope':envelope,'curvature_exact_fraction':str(B),
                             'width_exact_fraction':str(hi-lo)})
        if len(records)!=512:raise ValueError('complete512 frozen law pairs required')
        io.write(a.output_dir/'CONTEXTS512.json',{'records':contexts,'new_target_values_emitted':0})
        io.write(a.output_dir/'RF_ROUTES512.json',{'records':routes,'frozen_RF_state_sha256':plan['inputs']['frozen_rf']['sha256'],
                 'existing_model_training_rows':296,'estimator_refits':0})
        io.write(a.output_dir/'FORECASTS512.json',{'records':records,'F':'savedRF296','G':'savedg5repair','new_model_fits':0})
        io.write(a.output_dir/'GEOMETRY512.json',{'records':geometry,'exact_for_stored_laws':True})
        Bsum=sum((Fraction(r['curvature_exact_fraction']) for r in geometry),Fraction())
        widthsum=sum((Fraction(r['width_exact_fraction']) for r in geometry),Fraction())
        strata={}
        for name,predicate in (('g_zero',lambda v:v==0),('g_positive',lambda v:v>0)):
            selected=[r for r,c in zip(geometry,contexts) if predicate(c['bounded_descriptors8'][5])]
            strata[name]={'groups':len(selected),'B_sum_exact_fraction':str(sum((Fraction(r['curvature_exact_fraction']) for r in selected),Fraction())),
                          'width_sum_exact_fraction':str(sum((Fraction(r['width_exact_fraction']) for r in selected),Fraction()))}
        io.write(a.output_dir/'PREDICTIVE_POWER_DIAGNOSTICS.json',{'actual_utc':io.utc(),'B_sum_exact_fraction':str(Bsum),
                 'width_sum_exact_fraction':str(widthsum),'nonzero_width_groups':sum(Fraction(r['width_exact_fraction'])>0 for r in geometry),
                 'zero_B_groups':sum(Fraction(r['curvature_exact_fraction'])==0 for r in geometry),
                 'B_sum_float_descriptive':float(Bsum),'width_sum_float_descriptive':float(widthsum),
                 'no_new_targets_no_tuning_from_this_diagnostic':True,'reference_variants':['frozenRF296','previouslyfrozeng5auto'],
                 'predeclared_g0_gpositive_strata_descriptive_only':strata,
                 'delta':'1/20','lambda_exponents':list(range(-6,3)),'decimal_precision':100,'new_optimizer_calls':0})
        pins={p.name:io.sha(p) for p in a.output_dir.iterdir() if p.is_file()}
        io.write(a.output_dir/'FORECAST_FREEZE.json',{'status':'ALL512_F_G_LAWS_FROZEN_BEFORE_ANY_TARGET','actual_utc':io.utc(),
                 'plan_sha256':plan['_plan_sha256'],'source_sha256':io.sources(),'identity_roles_sha256':plan['inputs']['identity_roles']['sha256'],
                 'files_sha256':pins,'law_pairs':512,'new_context_cells':1024,'new_targets':0,'new_fits':0,
                 'new_calibration_optimizations':0,'reference_train_rows':296,'full_raw_JSON_in_trusted_reader':True})
        print({'status':'ALL512_F_G_FROZEN','new_fits':0,'new_targets':0})
    except Exception as e:
        io.write(a.output_dir/'FAILED_STOP.json',{'actual_utc':io.utc(),'error_type':type(e).__name__,'error':str(e),
                 'no_target_based_replacement_or_rescue':True});raise


if __name__=='__main__':main()
