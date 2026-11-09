"""Plot already-frozen scores only. No fitting or target-source reader."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent


def main():
    original=json.loads((ROOT/'functional_repair/pilot_once/SCORES.json').read_text('utf-8'))
    extra=json.loads((ROOT/'functional_mixture/development_once/SCORES.json').read_text('utf-8'))
    rows=original['macro_by_mask']+extra['macro_by_mask']
    table={(r['mask'],r['method']):r for r in rows}
    primary='interior_holes'
    methods=['full_anchor_gaussian','full_anchor_student4','half_maxwell_mean_gaussian',
             'whole_curve_student4_density_mixture','moment_matched_student4',
             'channel_relative_Maxwell_NNLS_point']
    labels=['GP Gaussian','GP Student4','Mean midpoint Gaussian','Whole-curve Student mixture',
            'Same-moment Student4','Maxwell point only']
    colors=['#64748b','#64748b','#d97706','#0284c7','#16a34a','#9d174d']
    fig,axes=plt.subplots(1,3,figsize=(14,5.2),layout='constrained')
    for ax,metric,title in zip(axes,['log_crps','log_rmse','joint_nll_per_dim'],
                            ['Marginal CRPS in log units','Point log RMSE','Joint NLL per target dimension']):
        for i,(method,label,color) in enumerate(zip(methods,labels,colors)):
            value=table[primary,method].get(metric)
            if value is not None:
                ax.scatter(value,i,s=75,c=color,zorder=3)
                ax.annotate(f'{value:.5f}',(value,i),xytext=(7,6),textcoords='offset points',fontsize=9)
        ax.set_yticks(range(len(labels)),labels if ax is axes[0] else ['']*len(labels))
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(title+'\n(lower is better)',fontsize=10)
        ax.margins(x=.27,y=.18)
    fig.suptitle('Internal gaps: one polyurethane, six humidity/history states near 30 C\n'
                 '12 observed scalars -> 14 target scalars; midpoint follow-up is reused development',fontsize=12)
    fig.savefig(ROOT/'figures/C_primary_scores.png',dpi=170);plt.close(fig)

    masks=['low_frequency_prefix_extrapolation','interior_holes','high_frequency_prefix_extrapolation']
    mask_labels=['Low-frequency anchors -> high f','Internal gaps','High-frequency anchors -> low f']
    methods2=['full_anchor_student4','whole_curve_student4_density_mixture','moment_matched_student4']
    names=['GP Student4','Whole-curve mixture','Same-moment Student4']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric,title in zip(axes,['log_crps','log_rmse','coverage90'],
                              ['Log CRPS (lower better)','Log RMSE (lower better)','90% interval coverage']):
        array=np.array([[table[mask,m][metric] for m in methods2] for mask in masks])
        image=ax.imshow(array,cmap='Blues' if metric=='coverage90' else 'YlOrRd',
                        vmin=0,vmax=1 if metric=='coverage90' else float(array.max()),aspect='auto')
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{array[i,j]:.3f}',ha='center',va='center',
                        color='white' if array[i,j]>.65*image.norm.vmax else '#111827',fontsize=10)
        ax.set_xticks(range(3),names,rotation=22,ha='right')
        ax.set_yticks(range(3),mask_labels if ax is axes[0] else ['']*3)
        ax.set_title(title,fontsize=11);fig.colorbar(image,ax=ax,shrink=.73)
    fig.suptitle('All declared geometries: extrapolation failures remain visible\n'
                 'No independent batch or reliable-calibration claim',fontsize=12)
    fig.savefig(ROOT/'figures/C_all_geometries.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
