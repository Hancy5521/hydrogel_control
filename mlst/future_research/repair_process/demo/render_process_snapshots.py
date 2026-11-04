"""Render recorded local-repair states with readable panel spacing.

Plot-only helper. It reads the saved synthetic fixture and six actual states;
it does not run relaxation, sampling, model fitting or target scoring.
"""
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=Path(__file__).with_name('output'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or args.input_dir/'REPAIR_PROCESS_R2.png'
    if output.exists():raise FileExistsError('Do not overwrite a previous figure')
    fixture_path=args.input_dir/'SYNTHETIC_FIXTURE.json'
    trajectory_path=args.input_dir/'TRAJECTORY.json'
    fixture=json.loads(fixture_path.read_text('utf-8'))
    trajectory=json.loads(trajectory_path.read_text('utf-8'))
    assert fixture['synthetic_only'] and trajectory['synthetic_only']
    states=trajectory['snapshots'];size=int(np.sqrt(len(fixture['reference_mean'])))
    shown=np.concatenate([np.asarray(s['values'])[s['active']]for s in states])
    norm=colors.Normalize(float(shown.min()),float(shown.max()))
    cmap=plt.colormaps['viridis'].copy();cmap.set_bad('#e0e3e7')
    fig,axes=plt.subplots(2,3,figsize=(12,8.4),constrained_layout=True)
    for index,(ax,state)in enumerate(zip(axes.ravel(),states)):
        field=np.asarray(state['values'],float).copy()
        field[~np.asarray(state['active'],bool)]=np.nan
        im=ax.imshow(field.reshape(size,size),cmap=cmap,norm=norm,origin='lower')
        ax.set_title(f"Sweep {state['sweep']} | repaired region {state['activated_missing_nodes']}/49",fontsize=11,pad=8)
        ax.add_patch(plt.Rectangle((2.5,2.5),7,7,fill=False,ec='#222222',lw=1.1,ls='--'))
        ax.set_xticks([0,6,12]);ax.set_yticks([0,6,12])
        if index>=3:ax.set_xlabel('Synthetic grid column')
        ax.set_ylabel('Synthetic grid row')
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.85,pad=.02)
    cb.set_label('Synthetic field value (arbitrary units)')
    fig.suptitle('Local repair process\nGray = not yet activated; fixed observations lie outside the dashed patch',fontsize=12)
    fig.savefig(output,dpi=170);plt.close(fig)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt={'actual_utc':datetime.now(timezone.utc).isoformat(),'classification':'Plot-only label-spacing correction from unmodified saved synthetic states',
        'source_sha256':digest(Path(__file__)),'fixture_sha256':digest(fixture_path),
        'trajectory_sha256':digest(trajectory_path),'selected_output':output.name,'output_sha256':digest(output),
        'snapshots':[s['sweep']for s in states],'new_numerical_execution':False,
        'initial_figure_retained':'REPAIR_PROCESS.png: upper x labels crowded lower panel titles; not selected for delivery',
        'visual_QA_pending':True}
    receipt_path=output.with_suffix('.json');assert not receipt_path.exists()
    receipt_path.write_text(json.dumps(receipt,indent=2),'utf-8')
    print(json.dumps({'status':'RENDERED_SAVED_STATES_ONLY','output':output.name}))


if __name__=='__main__':main()
