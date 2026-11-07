"""Audit saved numerical-gate inputs/outputs only; never call SUT or optimizer.

Use after actual root registration. Existing failures stay failures; independent
arithmetic PASS means receipts agree, not that the learner gates all passed.
"""
import argparse
from datetime import datetime,timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np
from reference_math import (P,ZERO,vec,matrix,scalar,inner,times,geometry,
                            law_from_theta,energy_crps,theta_beta,exact_range,
                            objective,policy,ticket_position)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,label):
    if not ok:
        raise AssertionError(label)


def close(a,b,label,tol=2e-10):
    a,b=float(a),float(b)
    require(abs(a-b)<=tol*max(1,abs(a),abs(b)),label)


def msum(values):
    values=list(values)
    n=len(values[0])
    return tuple(tuple(sum((q[j][k] for q in values),ZERO) for k in range(n)) for j in range(n))


def vsum(values):
    values=list(values)
    return tuple(sum((v[j] for v in values),ZERO) for j in range(len(values[0])))


def data(case):
    return [geometry(law,y,g) for law,y,g in zip(case['laws'],case['true_generated_targets'],case['g'])]


def kkt(theta,gradient,priorgradient):
    s=tuple((t-P[j%4]/100)/F(99,100) for j,t in enumerate(vec(theta)))
    gg=np.asarray([float(v) for v in gradient])
    ss=np.asarray([float(v) for v in s])
    scale=max(1.,max(abs(float(v)) for v in gradient),max(abs(float(v)) for v in priorgradient))
    station=dual=comp=0.
    for k in range(0,len(s),4):
        free=ss[k:k+4]>1e-8
        require(free.any(),'free coordinate in each normalized simplex')
        lam=float(np.mean(gg[k:k+4][free]))
        station=max(station,float(np.max(np.abs(gg[k:k+4][free]-lam))))
        if (~free).any():
            dual=max(dual,float(max(0.,np.max(lam-gg[k:k+4][~free]))))
        comp=max(comp,float(np.max(np.abs(ss[k:k+4]*(gg[k:k+4]-lam)))))
    return {'stationarity':station,'active_dual':dual,'complementarity':comp,'gradient_scale':scale,
            'passed':station<=1e-6*scale and dual<=1e-6*scale and comp<=1e-7*scale}


def audit_fit(record,Q,c,prior,count,penalty=None):
    Q,c,prior=matrix(Q),vec(c),vec(prior)
    rank,projection=exact_range(Q)
    cc=times(projection,c)
    d=record['diagnostics']
    require(rank==d['rank_exact'],'exact rank')
    require(projection==matrix(d['projector_exact']),'independent rational projector')
    require(c==vec(d['c_input_exact']),'original c exact')
    require(cc==vec(d['c_projected_exact']),'projected c exact')
    priorvalue,priorgrad=objective(prior,Q,cc,prior,count=count,penalty_Q=penalty)
    close(priorvalue,d['prior_objective'],'prior objective')
    theta=vec(record['theta'])
    # Validate actual output law feasibility whether accepted or fallback.
    theta_beta(theta)
    proposed=record['proposed_theta']
    if proposed is None:
        require(not record['accepted'] and theta==prior,'no proposal returns prior')
        return {'accepted':False,'rank':rank,'proposal':None}
    proposed=vec(proposed)
    value,gradient=objective(proposed,Q,cc,prior,count=count,penalty_Q=penalty)
    close(value,d['postprocessed_objective'],'postprocessed objective')
    close(value-priorvalue,d['objective_change_vs_prior'],'objective difference')
    raw=d['raw_s']
    cleaned=[]
    for k in range(0,len(raw),4):
        require(abs(sum(raw[k:k+4])-1)<=1e-8 and min(raw[k:k+4])>=-1e-10,'raw feasible')
        row=tuple(F(str(max(0.,float(v)))) for v in raw[k:k+4])
        cleaned.extend(v/sum(row,ZERO) for v in row)
    expected=tuple(P[j%4]/100+F(99,100)*s for j,s in enumerate(cleaned))
    require(expected==proposed,'exact float cleanup')
    independently=kkt(proposed,gradient,priorgrad)
    for name in ('stationarity','active_dual','complementarity','gradient_scale'):
        close(independently[name],d['kkt'][name],'KKT '+name,tol=2e-9)
    require(independently['passed']==d['kkt']['passed'],'KKT pass/fail')
    if record['accepted']:
        require(theta==proposed and independently['passed'],'accepted KKT')
        require(float(value)<=float(priorvalue)+1e-10*max(1.,abs(float(priorvalue))),'accepted objective')
    else:
        require(theta==prior,'failed proposal preserves declared prior')
        if d.get('fallback_reason')=='postprocessed_kkt_failed':
            require(not independently['passed'],'KKT failure independently reproduced')
    return {'accepted':record['accepted'],'rank':rank,'kkt':independently,
            'objective':float(value),'exact_objective':str(value),'fallback':d.get('fallback_reason')}


def score_identity(case,geo,theta):
    out=[]
    b=theta_beta(theta)
    Q=geo['Q']
    allgaps=tuple(inner(b,times(Q,b))+2*inner(b,c) for c in geo['knot_c'])
    envelope=(min(allgaps),max(allgaps))
    require(envelope==vec(case['score_checks']['gap_bounds']),'all-real gap extrema')
    candidate=law_from_theta(case['laws'][0],theta,case['g'][0])
    for rec in case['score_checks']['checks']:
        diff=energy_crps(candidate,rec['y'])-energy_crps(case['laws'][0],rec['y'])
        ci=geometry(case['laws'][0],rec['y'],case['g'][0])['c']
        risk=inner(b,times(Q,b))+2*inner(b,ci)
        require(diff==risk==scalar(rec['score_difference'])==scalar(rec['quadratic_difference']),'score identity')
        require(envelope[0]<=diff<=envelope[1],'all-real envelope')
        out.append(str(diff))
    return {'identities':len(out),'envelope':list(map(str,envelope))}


def audit_first(case):
    gs=data(case)
    Q=msum(g['Q'] for g in gs)
    c=vsum(g['c'] for g in gs)
    require(Q==matrix(case['Q_exact']) and c==vec(case['c_exact']),'aggregate Q/c')
    fit=audit_fit(case['fit'],Q,c,case['prior_theta'],case['remaining_count'])
    out={'name':case['name'],'source_gate_passed':case['passed'],'fit':fit,'rows':len(gs)}
    if 'score_checks' in case:
        out['score']=score_identity(case,gs[0],case['fit']['theta'])
    if 'tiny_matrix' in case:
        rank,proj=exact_range(case['tiny_matrix'])
        require(rank==case['tiny_rank']==3 and proj==matrix(case['tiny_projector']),'tiny nonzero rank retained')
    if 'gradient_check' in case:
        rec=case['gradient_check']
        # The declared decimal interior point sums to one; source binary
        # arithmetic is compared at the explicitly saved fixed tolerance.
        ss=tuple(F(str(v)) for v in rec['s'])
        theta=tuple(P[j%4]/100+F(99,100)*s for j,s in enumerate(ss))
        _,projection=exact_range(Q)
        _,gradient=objective(theta,Q,times(projection,c),case['prior_theta'],count=case['remaining_count'])
        for a,b in zip(gradient,rec['analytic']):close(a,b,'analytic gradient')
    if 'boundary_reached' in case:
        # For a target above support and tau=1/4, unconstrained optimum is
        # H=.2F+.8delta_max; no boundary is expected at epsilon=.01.
        exactopt=tuple(p/5+(F(4,5) if j==3 else ZERO) for j,p in enumerate(P))
        risk,_=objective(exactopt,Q,c,case['prior_theta'],count=case['remaining_count'])
        out['outside_support_exact_optimum_theta']=list(map(str,exactopt))
        out['outside_support_exact_optimum_objective']=str(risk)
        out['source_boundary_assertion_is_invalid_with_declared_tau']=True
        for a,b in zip(exactopt,vec(case['fit']['theta'])):close(a,b,'outside-support optimum',tol=2e-10)
    if 'exact_generating_population_gain' in case:
        prior=case['prior_theta'];theta=case['fit']['theta'];hidden=case['generating_theta_not_passed_to_fit']
        old=new=ZERO
        for law,g in zip(case['laws'],case['g']):
            truth=law_from_theta(law,hidden,g)
            oldlaw,newlaw=law_from_theta(law,prior,g),law_from_theta(law,theta,g)
            old+=sum((w*energy_crps(oldlaw,y) for y,w in truth),ZERO)
            new+=sum((w*energy_crps(newlaw,y) for y,w in truth),ZERO)
        old/=len(gs);new/=len(gs)
        require(new-old==scalar(case['exact_generating_population_gain']),'unknown-direction exact population gain')
        out['unknown_direction_exact_gain']=str(new-old)
    return out


def audit_loop(case):
    gs=data(case);n=len(gs);d=len(gs[0]['c'])
    lohi=[g['bounds'] for g in gs]
    mids=[tuple((a+b)/2 for a,b in bounds) for bounds in lohi]
    widths=[max(b-a for a,b in bounds) for bounds in lohi]
    M=vsum(mids)
    past=sumc=sumz=(ZERO,)*d
    remaining=list(range(n))
    theta=P*2
    updates=[]
    selected=[]
    totalc=vsum(g['c'] for g in gs)
    batch_expect=[]
    for t,rec in enumerate(case['query_records']):
        require(vec(rec['theta_before_query'])==theta,'predictable current batch theta')
        b=theta_beta(theta)
        gapwidth=[]
        for i in remaining:
            values=[inner(b,c) for c in gs[i]['knot_c']]
            gapwidth.append(2*(max(values)-min(values)))
        q=policy([widths[i] for i in remaining],gapwidth)
        position=ticket_position(q,rec['ticket'],rec['ticket_total'])
        index=remaining[position]
        require(index==rec['index'] and position==rec['position'],'exact integer ticket selected identity')
        require(q[position]==scalar(rec['q']) and len(remaining)==rec['remaining_before'],'draw q/population')
        if t%32==0:
            expectation=(ZERO,)*d
            for i,qi in zip(remaining,q):
                z=tuple(m+p+(c-mid)/qi for m,p,c,mid in zip(M,past,gs[i]['c'],mids[i]))
                expectation=tuple(x+qi*y for x,y in zip(expectation,z))
            require(expectation==totalc,'predraw fixed-total conditional expectation')
            batch_expect.append(True)
        require(scalar(rec['target_exact'])==scalar(case['true_generated_targets'][index]),'selected actual generated label')
        c=gs[index]['c'];residual=tuple(a-b for a,b in zip(c,mids[index]))
        Z=tuple(m+p+r/q[position] for m,p,r in zip(M,past,residual))
        require(c==vec(rec['c_exact']) and Z==vec(rec['Z_total_exact']),'exact c and residual-total Z')
        sumz=tuple(a+b for a,b in zip(sumz,Z))
        sumc=tuple(a+b for a,b in zip(sumc,c))
        past=tuple(a+b for a,b in zip(past,residual))
        selected.append(index);del remaining[position]
        if (t+1)%32==0:
            update=case['updates'][(t+1)//32-1]
            Q=msum(gs[i]['Q'] for i in remaining)
            chat=tuple(z/(t+1)-c for z,c in zip(sumz,sumc))
            require(Q==matrix(update['Q_remaining']) and chat==vec(update['Chat_remaining']),'remaining Q and Chat subtraction')
            require(remaining==update['remaining_indices'],'actual remaining identities')
            details=audit_fit(update['fit'],Q,chat,update['prior_theta'],len(remaining))
            updates.append({'queries':t+1,**details})
            theta=vec(update['fit']['theta'])
    require(len(selected)==len(set(selected))==128 and len(remaining)==128,'four 32-query batches')
    require(batch_expect==case['conditional_total_expectation_exact_checks'],'all batch expectation records')
    qr=msum(gs[i]['Q'] for i in remaining)
    scale=F(len(remaining),len(selected))
    qq=tuple(tuple(scale*v for v in row) for row in msum(gs[i]['Q'] for i in selected))
    cc=tuple(scale*v for v in vsum(gs[i]['c'] for i in selected))
    erm=audit_fit(case['ordinary_query_erm'],qq,cc,P*2,len(remaining),penalty=qr)
    return {'name':case['name'],'source_gate_passed':case['passed'],'query_records_exact':len(selected),
            'conditional_expectation_exact_checks':len(batch_expect),'updates':updates,'ordinary_erm':erm}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    require(not args.output.exists(),'audit output is append-only')
    d=json.loads(args.input.read_text('utf-8-sig'))
    require(len(d['cases'])==6 and d['optimizer_calls']==10,'saved six-case ten-call gate')
    started=datetime.now(timezone.utc).isoformat()
    results=[audit_first(c) for c in d['cases'][:5]]+[audit_loop(d['cases'][5])]
    out={'status':'PASS_INDEPENDENT_SAVED_ARITHMETIC_NOT_A_LEARNER_GATE_PASS',
         'actual_start_utc':started,'actual_end_utc':datetime.now(timezone.utc).isoformat(),
         'input':str(args.input.resolve()),'input_sha256':sha(args.input),
         'audit_source_sha256':sha(__file__),'reference_math_sha256':sha(Path(__file__).with_name('reference_math.py')),
         'source_gate_passed':d['passed'],'fits_or_optimizers_called':0,'SUT_imports':0,
         'new_material_cells_read':0,'cases':results,
         'scope':'Reconstruct saved inputs, outcomes and postprocessing; preserve original failed assertions/fallbacks.'}
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':out['status'],'source_gate_passed':out['source_gate_passed'],'cases':len(results),'sha256':sha(args.output)}))

if __name__=='__main__':main()
