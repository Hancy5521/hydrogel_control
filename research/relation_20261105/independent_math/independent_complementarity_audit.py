"""Independent registered toy-mathematics audit. No producer or material imports."""
from pathlib import Path
from fractions import Fraction as Q
import argparse
import collections
import datetime
import hashlib
import json
import math
import traceback
import itertools

def sha(b):
    return hashlib.sha256(b).hexdigest()

def put(path,obj):
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=True,indent=2);f.write('\n')

def require(condition,message):
    if not condition:
        raise AssertionError(message)

def toy_case(name):
    """Return an exactly normalized finite joint law (h,a,b,y,p) and unit IDs."""
    atoms=[]
    if name=='duplicated_unit':
        for b in (-1,1):
            atoms.append((0,b,b,b,Q(1,2)))
        return atoms,('same_original_unit','same_original_unit')
    if name=='B_informs_theta':
        for h in (0,1):
            theta_sign=2*h-1
            for b in (-1,1):
                pb=Q(3,4) if b==theta_sign else Q(1,4)
                atoms.append((h,h,b,theta_sign,Q(1,2)*pb))
        return atoms,('relation_unit','proxy_unit')
    hs=(0,) if name=='theta_point_mass' else (0,1)
    accuracy=Q(3,4) if name in ('noisy_A','noisy_A_and_Y') else Q(1)
    flip=Q(1,4) if name in ('noisy_Y','noisy_A_and_Y') else Q(0)
    for h in hs:
        wh=Q(1,len(hs))
        for a in hs:
            pa=Q(1) if len(hs)==1 else (accuracy if a==h else 1-accuracy)
            for b in (-1,1):
                if name in ('same_relation','theta_point_mass'):
                    y0=b
                elif name=='A_has_target_main_effect':
                    y0=b+2*h
                else:
                    y0=b if h==0 else -b
                for do_flip,pf in ((False,1-flip),(True,flip)):
                    p=wh*pa*Q(1,2)*pf
                    if p:
                        y=-y0 if do_flip else y0
                        atoms.append((h,a,b,y,p))
    return atoms,('relation_unit','proxy_unit')

def marginal(atoms,positions):
    out=collections.defaultdict(Q)
    for *v,p in atoms:
        out[tuple(v[i] for i in positions)]+=p
    return dict(out)

def exact_discrete(atoms,unit_ids):
    require(sum(x[4] for x in atoms)==1,'Finite joint law not normalized')
    support=sorted({x[3] for x in atoms})
    interval=list(zip(support[:-1],support[1:]))
    wh=marginal(atoms,(0,)); pa=marginal(atoms,(1,)); pb=marginal(atoms,(2,))
    pha=marginal(atoms,(0,1)); phb=marginal(atoms,(0,2)); phby=marginal(atoms,(0,2,3))
    phaby=marginal(atoms,(0,1,2,3))
    hs=sorted(h[0] for h in wh); av=sorted(a[0] for a in pa); bv=sorted(b[0] for b in pb)
    def average_bayes_risk(observed):
        bins=collections.defaultdict(list)
        for atom in atoms:
            bins[tuple(atom[i] for i in observed)].append(atom)
        total=Q(0)
        for block in bins.values():
            pblock=sum(t[4] for t in block)
            for z,z1 in interval:
                cdf=sum(t[4] for t in block if t[3]<=z)/pblock
                total+=pblock*(z1-z)*cdf*(1-cdf)
        return total
    base=average_bayes_risk(())
    ga=base-average_bayes_risk((1,))
    gb=base-average_bayes_risk((2,))
    gab=base-average_bayes_risk((1,2))
    interaction=gab-ga-gb
    F=[[Q(0) for k in hs] for h in hs]
    for a in av:
        delta=[pha.get((h,a),Q(0))/pa[(a,)]-wh[(h,)] for h in hs]
        for i in range(len(hs)):
            for j in range(len(hs)):
                F[i][j]+=pa[(a,)]*delta[i]*delta[j]
    S=[[Q(0) for k in hs] for h in hs]
    for b in bv:
        for z,z1 in interval:
            delta=[]
            for h in hs:
                fh=sum(t[4] for t in atoms if t[0]==h and t[3]<=z)/wh[(h,)]
                hb=phb.get((h,b),Q(0))
                require(hb>0,'This audit requires positive support for every h,b')
                gh=sum(t[4] for t in atoms if t[0]==h and t[2]==b and t[3]<=z)/hb
                delta.append(gh-fh)
            for i in range(len(hs)):
                for j in range(len(hs)):
                    S[i][j]+=pb[(b,)]*(z1-z)*delta[i]*delta[j]
    trace=sum(F[i][j]*S[j][i] for i in range(len(hs)) for j in range(len(hs)))
    b_theta_independent=all(phb.get((h,b),Q(0))==wh[(h,)]*pb[(b,)] for h in hs for b in bv)
    cond_independent=True
    for h in hs:
        for a in av:
            for b in bv:
                for y in support:
                    lhs=phaby.get((h,a,b,y),Q(0))*wh[(h,)]
                    rhs=pha.get((h,a),Q(0))*phby.get((h,b,y),Q(0))
                    cond_independent &= lhs==rhs
    row_sum_zero=all(sum(row)==0 for row in F)
    for M in (F,S):
        require(all(M[i][i]>=0 for i in range(len(hs))),'Negative exact Gram diagonal')
        if len(hs)==2:
            require(M[0][0]*M[1][1]-M[0][1]*M[1][0]>=0,'Negative exact Gram determinant')
    require(row_sum_zero,'Posterior covariance does not annihilate all-ones')
    same_margin=all(
        sum(t[4] for t in atoms if t[0]==h and t[3]<=z)/wh[(h,)]
        ==sum(t[4] for t in atoms if t[0]==hs[0] and t[3]<=z)/wh[(hs[0],)]
        for h in hs for z in support)
    return {
        'atoms':len(atoms),'states':len(hs),'outcome_support':support,
        'baseline_expected_CRPS':str(base),'A_main_gain':str(ga),'B_main_gain':str(gb),
        'AB_total_gain':str(gab),'true_synergy':str(interaction),'trace_prediction':str(trace),
        'exact_difference':str(interaction-trace),
        'B_entire_margin_independent_of_theta':b_theta_independent,
        'A_conditional_independent_of_B_and_Y_given_theta':bool(cond_independent),
        'information_units_are_distinct':unit_ids[0]!=unit_ids[1],
        'Y_marginal_common_across_theta':same_margin,
        'F':[[str(x) for x in row] for row in F],'S':[[str(x) for x in row] for row in S],
    }

def finite_checks():
    expected={
        'perfect_sign_relation':Q(1,2),'noisy_A':Q(1,8),'noisy_Y':Q(1,8),
        'noisy_A_and_Y':Q(1,32),'same_relation':Q(0),'theta_point_mass':Q(0),
        'B_informs_theta':Q(-1,8),'duplicated_unit':Q(-1,2),
    }
    names=list(expected)+['A_has_target_main_effect']
    out={}
    for name in names:
        atoms,units=toy_case(name)
        d=exact_discrete(atoms,units)
        synergy=Q(d['true_synergy']); diff=Q(d['exact_difference'])
        if name in expected:
            require(synergy==expected[name],'Unexpected exact toy synergy: '+name)
        if name in ('B_informs_theta','duplicated_unit'):
            require(diff!=0 and synergy<0,'Violation case failed to falsify misplaced trace identity')
        else:
            require(diff==0 and synergy>=0,'Valid-case exact interaction identity failed: '+name)
            require(d['B_entire_margin_independent_of_theta'] and d['A_conditional_independent_of_B_and_Y_given_theta'] and d['information_units_are_distinct'],'Valid-case assumptions fail')
        if name=='A_has_target_main_effect':
            require(Q(d['A_main_gain'])>0,'A target main-effect counterexample absent')
        else:
            if name not in ('B_informs_theta','duplicated_unit'):
                require(Q(d['A_main_gain'])==0,'Unexpected pure-relation A main effect')
        out[name]=d
    return out

def adaptive_same_F_checks():
    """Recover true Bayes risks from full joint tables, enumerate policies, not E[max] shortcut."""
    designs={
        'posterior_quarters':[(Q(1,2),Q(1,4)),(Q(1,2),Q(3,4))],
        'posterior_endpoints_and_midpoint':[(Q(1,8),Q(0)),(Q(3,4),Q(1,2)),(Q(1,8),Q(1))],
    }
    results={}
    for name,channel in designs.items():
        # Each channel item is (P(A=a), P(theta=0|A=a)); Bayes reconstructs P(A=a|theta).
        atoms=[]
        for h in (0,1):
            for a,(pa,p0) in enumerate(channel):
                likelihood=2*pa*(p0 if h==0 else 1-p0)
                for y in (0,1):
                    for coin in (0,1):
                        b1,b2=(y,coin) if h==0 else (coin,y)
                        p=Q(1,2)*likelihood*Q(1,2)*Q(1,2)
                        if p:
                            atoms.append((h,a,b1,b2,y,p))
        require(sum(x[5] for x in atoms)==1,'Adaptive table not normalized')
        def risk(observation_key):
            bins=collections.defaultdict(list)
            for atom in atoms:
                bins[observation_key(atom)].append(atom)
            result=Q(0)
            for block in bins.values():
                mass=sum(t[5] for t in block)
                p_y0=sum(t[5] for t in block if t[4]==0)/mass
                result+=mass*p_y0*(1-p_y0) # Binary outcomes0,1, integrated CDF gap=1.
            return result
        base=risk(lambda t:())
        risk_A=risk(lambda t:(t[1],))
        # Project the same joint table into each fixed proxy experiment for an independent S/F recovery.
        fixed={}
        F=None
        for j in (0,1):
            projected=[(h,a,b1 if j==0 else b2,y,p) for h,a,b1,b2,y,p in atoms]
            d=exact_discrete(projected,('relation_unit','proxy_'+str(j)))
            require(d['Y_marginal_common_across_theta'] and d['B_entire_margin_independent_of_theta'] and d['A_conditional_independent_of_B_and_Y_given_theta'],'Adaptive fixture violates pure-relation assumptions')
            require(Q(d['AB_total_gain'])==Q(5,64),'Fixed-pair total CRPS gain mismatch')
            if F is None:
                F=d['F']
            else:
                require(F==d['F'],'Posterior F differs by fixed proxy')
            expected_S=[['1/4','0'],['0','0']] if j==0 else [['0','0'],['0','1/4']]
            require(d['S']==expected_S,'Requested diagonal proxy S not recovered from joint law')
            fixed[str(j)]={'risk':str(base-Q(d['AB_total_gain'])),'total_gain':d['AB_total_gain'],'true_synergy':d['true_synergy'],'S':d['S']}
        require(F==[['1/16','-1/16'],['-1/16','1/16']],'Channel posterior covariance differs from requested same-F design')
        policies=[]
        for choices in itertools.product((0,1),repeat=len(channel)):
            # Legal observable key contains A and only the chosen B, never the unrevealed alternative.
            rr=risk(lambda t:(t[1],choices[t[1]],t[2+choices[t[1]]]))
            policies.append({'action_for_each_A':list(choices),'risk':str(rr),'gain':str(base-rr)})
        best=min(policies,key=lambda p:Q(p['risk']))
        expected_gain=Q(9,64) if name=='posterior_quarters' else Q(7,64)
        expected_risk=Q(7,64) if name=='posterior_quarters' else Q(9,64)
        require(base==Q(1,4) and risk_A==base,'Pure relation observation unexpectedly changes target risk')
        require(Q(best['gain'])==expected_gain and Q(best['risk'])==expected_risk,'Full-joint adaptive Bayes risk counterexample not recovered')
        results[name]={'atoms':len(atoms),'posterior_law':[[str(x) for x in pair] for pair in channel],
            'baseline_expected_CRPS':str(base),'A_only_expected_CRPS':str(risk_A),'F':F,'fixed_proxies':fixed,
            'all_legal_A_then_one_B_policies':policies,'best_policy_by_exact_joint_risk':best}
    require(results['posterior_quarters']['F']==results['posterior_endpoints_and_midpoint']['F'],'Channels should have identical posterior covariance')
    require(results['posterior_quarters']['best_policy_by_exact_joint_risk']['risk']!=results['posterior_endpoints_and_midpoint']['best_policy_by_exact_joint_risk']['risk'],'Same-F different adaptive value counterexample absent')
    return {'status':'EXACT_SAME_F_DIFFERENT_ADAPTIVE_BAYES_RISK','designs':results,
        'meaning':'F_A and fixed-pair tr(F_A S_B) do not retain the full posterior law needed by an adaptive choice among B actions. This does not falsify the fixed-pair identity or invent a new planning algorithm.'}

def normal_cdf_matrix(rhos,n_b,n_z,limit):
    import numpy as np
    from scipy.special import roots_hermitenorm,ndtr
    b,wb=roots_hermitenorm(n_b); wb=wb/math.sqrt(2*math.pi)
    z,wz=np.polynomial.legendre.leggauss(n_z);z=z*limit;wz=wz*limit
    rho=np.array(rhos,dtype=float)
    G=ndtr((z[None,None,:]-rho[:,None,None]*b[None,:,None])/np.sqrt(1-rho*rho)[:,None,None])
    D=G-ndtr(z)[None,None,:]
    weight=wb[:,None]*wz[None,:]
    QD=(D*np.sqrt(weight)[None,:,:]).reshape(len(rhos),-1)
    S=QD@QD.T
    return S,D,weight

def absolute_difference_integral(variance,eps_abs,eps_rel):
    from scipy.integrate import quad
    # Independent absolute-value integration of the difference density; no known E|N| formula.
    density_factor=1/math.sqrt(2*math.pi*variance)
    f=lambda x:2*x*density_factor*math.exp(-x*x/(2*variance))
    value,error=quad(f,0,math.inf,epsabs=eps_abs,epsrel=eps_rel,limit=200)
    return value,error

def continuous_checks(plan):
    import numpy as np
    cfg=plan['continuous_gaussian_panel']; tol=plan['tolerances']
    rhos=cfg['rho_values']; limit=cfg['z_truncation_abs']
    require(all(abs(r)<1 for r in rhos),'Invalid Gaussian copula rho')
    matrices=[];lastD=None;lastW=None
    for q in cfg['fixed_quadrature_levels']:
        S,D,W=normal_cdf_matrix(rhos,q['normal_B_nodes'],q['z_Legendre_nodes'],limit)
        matrices.append(S);lastD=D;lastW=W
    S=matrices[-1]
    convergence=float(np.max(np.abs(matrices[-1]-matrices[-2])))
    require(convergence<=tol['cdf_fixed_level_max_difference'],'Fixed CDF quadrature disagreement')
    closed=np.array([[(1-math.sqrt(1-h*k))/math.sqrt(math.pi) for k in rhos] for h in rhos])
    baseline,baseline_err=absolute_difference_integral(2,tol['energy_quad_eps_abs'],tol['energy_quad_eps_rel'])
    energy=np.zeros_like(S);quad_error=baseline_err
    for i,h in enumerate(rhos):
        for j,k in enumerate(rhos):
            # Difference of two latent Y draws sharing B, with conditionally independent residuals.
            variance=(h-k)**2+(math.sqrt(1-h*h))**2+(math.sqrt(1-k*k))**2
            absdiff,err=absolute_difference_integral(variance,tol['energy_quad_eps_abs'],tol['energy_quad_eps_rel'])
            energy[i,j]=(baseline-absdiff)/2
            quad_error=max(quad_error,err)
    cdf_closed=float(np.max(np.abs(S-closed)))
    cdf_energy=float(np.max(np.abs(S-energy)))
    energy_closed=float(np.max(np.abs(energy-closed)))
    require(cdf_closed<=tol['cdf_vs_closed_max_abs'],'Independent CDF integral disagrees with candidate formula')
    require(cdf_energy<=tol['cdf_vs_energy_max_abs'],'CDF integral disagrees with independent energy integral')
    require(energy_closed<=tol['energy_vs_closed_max_abs'],'Independent energy integral disagrees with candidate formula')
    mineig=float(np.min(np.linalg.eigvalsh(S)))
    require(mineig>=-tol['PSD_negative_eigenvalue_slack'],'CDF Gram matrix not PSD')
    require(any(S[i,j]<0 for i in range(len(rhos)) for j in range(len(rhos))),'Negative off-diagonal control absent')
    # A binary relation channel and a common target marginal; derive posterior weights with exact fractions.
    prior=[Q(str(x)) for x in cfg['theta_weights']]
    chan=[Q(str(x)) for x in cfg['P_A0_given_theta']]
    require(sum(prior)==1,'Relation prior not normalized')
    Fq=[[Q(0) for _ in prior] for _ in prior]
    posterior=[]
    for a in (0,1):
        channel=chan if a==0 else [1-x for x in chan]
        pa=sum(w*p for w,p in zip(prior,channel))
        wp=[w*p/pa for w,p in zip(prior,channel)]
        delta=[v-w for v,w in zip(wp,prior)]
        for i in range(len(prior)):
            for j in range(len(prior)):
                Fq[i][j]+=pa*delta[i]*delta[j]
        posterior.append((float(pa),np.array([float(x) for x in wp])))
    require(all(sum(row)==0 for row in Fq),'Exact posterior covariance row sums are nonzero')
    F=np.array([[float(x) for x in row] for row in Fq])
    require(float(np.min(np.linalg.eigvalsh(F)))>=-tol['PSD_negative_eigenvalue_slack'],'Posterior covariance not PSD')
    w=np.array([float(x) for x in prior])
    Db=np.einsum('h,hbz->bz',w,lastD)
    gb=float(np.sum(Db*Db*lastW))
    gab=0.0
    for pa,wp in posterior:
        Dp=np.einsum('h,hbz->bz',wp,lastD)
        gab+=pa*float(np.sum(Dp*Dp*lastW))
    trace=float(np.trace(F@S)); direct_interaction=gab-gb
    trace_error=abs(direct_interaction-trace)
    require(trace_error<=tol['direct_gain_vs_trace_max_abs'],'Continuous direct conditional-CDF gain disagrees with trace')
    require(gb>0 and trace>0,'Continuous B main effect / interaction positive controls absent')
    # Symmetric +/-rho: exact zero Pearson does not imply zero distributional B information.
    a,b=cfg['symmetric_rho_indices']
    require(rhos[a]==-rhos[b] and rhos[b]>0,'Invalid symmetric-rho fixture')
    sw=np.array([0.5,0.5]);SS=S[np.ix_([a,b],[a,b])]
    sym_gb=float(sw@SS@sw)
    sym_F=np.array([[0.25,-0.25],[-0.25,0.25]])
    sym_interaction=float(np.trace(sym_F@SS))
    pearson=0.5*rhos[a]+0.5*rhos[b]
    require(pearson==0 and sym_gb>tol['strict_positive_control_min'] and sym_interaction>tol['strict_positive_control_min'],'Zero-Pearson distributional-information control failed')
    # Identical nonzero rho: a distinguishable theta channel does not create target-relevant relation differences.
    identical=np.full((2,2),S[b,b])
    identical_trace=float(np.trace(sym_F@identical))
    require(abs(identical_trace)<=tol['zero_control_max_abs'],'Identical-rho interaction should vanish')
    # Uncentered CDF matrix entry diverges; do not infer all F-projected expressions necessarily diverge.
    from scipy.special import ndtr
    unc=[]
    nz=cfg['uncentered_z_Legendre_nodes']
    base_nodes,base_weights=np.polynomial.legendre.leggauss(nz)
    for L in cfg['uncentered_positive_tail_limits']:
        z=base_nodes*L;wz=base_weights*L
        unc.append(float(np.sum(wz*ndtr(z)**2)))
    increase=unc[1]-unc[0]
    expected_growth=cfg['uncentered_positive_tail_limits'][1]-cfg['uncentered_positive_tail_limits'][0]
    require(abs(increase-expected_growth)<=tol['uncentered_growth_max_abs'],'Uncentered CDF element did not show tail-length divergence')
    return {
        'rho_values':rhos,'CDF_integral_S':S.tolist(),'independent_energy_integral_S':energy.tolist(),
        'candidate_closed_form_S':closed.tolist(),'F_exact_fractions':[[str(x) for x in row] for row in Fq],
        'fixed_levels':cfg['fixed_quadrature_levels'],'z_limit':limit,
        'analytic_tail_bound_used':'Every centered S entry beyond +/-L is bounded in absolute value by 2*phi(L); L=10.',
        'fixed_level_max_abs_difference':convergence,'CDF_vs_closed_max_abs':cdf_closed,
        'CDF_vs_energy_max_abs':cdf_energy,'energy_vs_closed_max_abs':energy_closed,
        'energy_quad_max_reported_abs_error':quad_error,'min_eigenvalue_CDF_S':mineig,
        'A_main_gain':0.0,'B_main_gain_direct_CDF':gb,'AB_total_gain_direct_CDF':gab,
        'synergy_direct_CDF':direct_interaction,'trace_FA_SB':trace,'direct_gain_vs_trace_max_abs':trace_error,
        'symmetric_zero_Pearson':{'rho_pair':[rhos[a],rhos[b]],'Pearson':pearson,'B_distributional_main_gain':sym_gb,'true_synergy':sym_interaction},
        'identical_nonzero_rho_synergy':identical_trace,
        'uncentered_entry_at_two_limits':unc,'uncentered_entry_growth':increase,
        'all_outcomes_are_artificial_mathematical_fixtures_not_material_measurements':True,
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True);ap.add_argument('--output-dir',required=True)
    args=ap.parse_args();planp=Path(args.plan);approvalp=Path(args.approval);out=Path(args.output_dir)
    planb=planp.read_bytes();plan=json.loads(planb);approval=json.loads(approvalp.read_text('utf-8'))
    require(approval['plan_sha256']==sha(planb),'Approval is not for this exact plan')
    require(approval['source_sha256']==sha(Path(__file__).read_bytes()),'Approval source mismatch')
    require(approval['approved_scope']=='INDEPENDENT_TOY_RELATION_CRPS_MATH_ONLY','Unexpected approval scope')
    require(plan['source_sha256']==sha(Path(__file__).read_bytes()),'Registered source changed')
    require(not out.exists(),'Output already exists; no automatic repeated audit')
    out.mkdir(parents=True)
    start={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':sha(planb),'approval_sha256':sha(approvalp.read_bytes()),'source_sha256':sha(Path(__file__).read_bytes()),'material_sources_or_producer_imported':False}
    put(out/'EXECUTION_START.json',start)
    try:
        finite=finite_checks()
        put(out/'EXACT_DISCRETE_RESULTS.json',finite)
        adaptive=adaptive_same_F_checks()
        put(out/'EXACT_ADAPTIVE_SAME_F_RESULTS.json',adaptive)
        continuous=continuous_checks(plan)
        put(out/'CONTINUOUS_INTEGRAL_RESULTS.json',continuous)
        receipt={'status':'PASS','actual_finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution':start,'exact_cases':len(finite),'same_F_adaptive_joint_table_designs':len(adaptive['designs']),'continuous_independent_methods':['Centered conditional-CDF product quadrature','Absolute-value energy integration of independent Gaussian residual construction'],'tolerances':plan['tolerances'],'files_sha256':{p.name:sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()},'new_material_measurements_fits_predictions_or_target_scores':0,'limits':'Conditional classical information identity and mechanism controls only; not algorithm novelty, cost superiority or material validation.'}
        put(out/'INDEPENDENT_MATH_RESULT.json',receipt)
        print(json.dumps({'status':'PASS','exact_cases':len(finite),'CDF_vs_energy_max_abs':continuous['CDF_vs_energy_max_abs'],'CDF_vs_closed_max_abs':continuous['CDF_vs_closed_max_abs'],'direct_gain_vs_trace_max_abs':continuous['direct_gain_vs_trace_max_abs'],'receipt_sha256':sha((out/'INDEPENDENT_MATH_RESULT.json').read_bytes())},ensure_ascii=True))
    except Exception:
        put(out/'FAILURE.json',{'status':'FAIL_PRESERVED_DO_NOT_CHANGE_TOLERANCE_OR_RERUN','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution':start,'traceback':traceback.format_exc(),'action':'Stop and report original failure. Any fix requires root-reviewed new source/plan and must preserve this execution.'})
        raise

if __name__=='__main__':
    main()
