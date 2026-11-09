"""Copy the closed C-stage resources into an unsealed local package.

Current growth-loop work is intentionally outside this closed snapshot. Every
copied source keeps its literal bytes; no protocol timestamps or hashes change.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent.parent
OUT=Path('H:/PolymerReview/Curve_Repair_20261109_C')


def main():
    state=json.loads((OUT/'BUILD_STATE.json').read_text('utf-8'))
    if state['status']!='BUILDING_NOT_SEALED_NOT_DELIVERED':
        raise ValueError('Only this unsealed checkpoint may receive additions')
    stage=OUT/'stage';sources=[]
    for name in ('quantile_midpoint','generalization','functional_repair','functional_mixture',
                 'scientific_review','literature','examples','example_outputs','figures'):
        sources.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    sources.extend(p for p in (ROOT/'runtime').glob('*') if p.is_file() and
                   (p.name.startswith('curve_reference') or p.name.startswith('CURVE_REFERENCE')))
    sources.extend(p for p in (ROOT/'verification').glob('functional_mixture_independent*') if p.is_file())
    sources.extend(p for p in (ROOT/'sources/generalization_admission_20261109/polymer_flow/candidate2_xanthan').rglob('*') if p.is_file())
    other=ROOT/'sources/generalization_admission_20261109/other_polymer_functions'
    sources.extend(p for p in other.glob('*') if p.is_file() and p.suffix in ('.json','.md','.txt','.py'))
    for name in ('darus_smp_originals','darus_functional_gate'):
        sources.extend(p for p in (other/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    # Unused, not-yet-admitted sources are represented by their gate reports,
    # URLs and hashes; their large/uncleared raw data are not runtime inputs.
    for directory in (ROOT/'sources/generalization_admission_20261109').glob('*'):
        if directory.name in ('polymer_flow','other_polymer_functions'):continue
        sources.extend(p for p in directory.glob('*') if p.is_file() and
                       ('GATE' in p.name or 'REPORT' in p.name or 'FOLLOWUP_PACKAGE' in p.name)
                       and p.suffix in ('.json','.md','.txt'))
    for name in ('prepare_checkpoint_C.py','copy_completed_C_resources.py'):
        sources.append(ROOT/'delivery'/name)
    receipts=[]
    for p in sorted(set(sources)):
        relative=p.relative_to(ROOT);target=stage/relative
        raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=raw:
            # Existing B files are never silently changed by a new stage.
            raise ValueError('Closed-stage addition would change a B payload: '+str(relative))
        if not target.exists():
            with target.open('xb') as f:f.write(raw)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('Source changed while copying: '+str(p))
        receipts.append({'source':str(p),'package_relative':'stage/'+relative.as_posix(),
                         'bytes':len(raw),'sha256':digest})
    record={'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'state':'UNSEALED_C_SCIENTIFIC_RESOURCES_COPIED','files':receipts,
        'source_files':len(receipts),'pending_growth_loop_included':False,
        'unused_unadmitted_raw_sources_included':False,
        'used_XG_and_DaRUS_raw_and_all_frozen_dependencies_included':True}
    with (OUT/'C_RESOURCE_ORIGINS.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print(json.dumps({'copied_or_verified':len(receipts),'out':str(OUT),'state':'UNSEALED'}))


if __name__=='__main__':main()
