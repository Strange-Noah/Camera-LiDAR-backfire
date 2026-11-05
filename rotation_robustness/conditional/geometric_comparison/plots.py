"""Regenerate missing statistical graphics from published aggregates only."""
import argparse,ast,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screening'))
from screen import heatmaps


def plots(src,out):
    src=Path(src);out=Path(out);out.mkdir(exist_ok=False)
    r=json.loads((src/'summary.json').read_text());variables=r['variables']
    names={'width':'Clean projected width','r':'Jacobian r_rot at 5°','q':'Physical q_rot at 5°'}
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=180)
        plt.close(fig)
    for name,v in variables.items():
        fig,axs=plt.subplots(2,5,figsize=(20,8),sharex=True)
        for i,distance in enumerate(v['distance_labels']):
            for j,label in enumerate(v['indicator_labels']):
                g=v['groups'][i*4+j]
                if not g['N']:continue
                for ax,key in zip(axs[:,i],['D','G']):
                    mean=g[key+'_mean'];sd=g[key+'_sd'];interval=np.array(g['uncertainty']['scene'][key+'_CI95'])
                    ax.plot(range(1,6),mean,marker='o',ms=3,color='C%d'%j,ls='-' if g['supported'] else ':',label=label+(' *' if not g['supported'] else ''))
                    if g['supported']:
                        ax.fill_between(range(1,6),interval[:,0],interval[:,1],color='C%d'%j,alpha=.12)
                        ax.errorbar(range(1,6),mean,yerr=sd,fmt='none',color='C%d'%j,lw=.6,capsize=2)
            for ax,key in zip(axs[:,i],['D','G']):
                ax.axhline(0,color='black',lw=.6);ax.set(title=distance,xlabel='Rotation σ (degrees per axis)',ylabel=key+' (pp)');ax.set_xticks(range(1,6));ax.tick_params(labelbottom=True);ax.legend(fontsize=7)
        fig.suptitle(names[name]+' × distance; shade: scene CI95; whiskers: seed SD; dotted/*: limited support; panel y scales vary')
        save(fig,name+'_conditional_DG')
    q=variables['q']
    heatmaps(dict(pair=['distance','q_rot5'],left_labels=q['distance_labels'],right_labels=q['indicator_labels'],grid=[q['groups'][i*4:i*4+4] for i in range(5)]),out/'q_heatmaps')
    contrasts=list(csv.DictReader(open(src/'contrasts.csv')))
    fig,axs=plt.subplots(2,3,figsize=(15,9),sharey=True)
    for j,name in enumerate(['width','q','r']):
        for row,metric in enumerate(['D_curve_average','G_sigma5']):
            ax=axs[row,j]
            for i,dist in enumerate(variables[name]['distance_labels']):
                for mode,color,offset in [('scene','C0',-.08),('scene_seed','C1',.08)]:
                    records=[x for x in contrasts if x['variable']==name and x['axis']=='indicator_within_distance' and x['stratum']==dist and x['mode']==mode and x['kind']=='endpoint' and x['metric']==metric]
                    if not records:continue
                    x=records[0];est=float(x['estimate_pp']);lo,hi=ast.literal_eval(x['ci95_pp'])
                    ax.plot([lo,hi],[i+offset]*2,color=color);ax.plot(est,i+offset,'o',color=color,ms=4,label=mode if i==1 else None)
            ax.axvline(0,color='black',lw=.7);ax.set_yticks(range(5),variables[name]['distance_labels']);ax.invert_yaxis() if j==0 and row==0 else None
            ax.set(title=names[name],xlabel=('Mean D over σ1…5' if row==0 else 'G at σ=5°')+': high-risk minus low-risk (pp)');ax.legend(fontsize=8)
    fig.suptitle('Supported endpoint contrasts; inverse-width risk order, positive q/r order; bars: CI95; blank = not testable')
    save(fig,'conditional_endpoint_contrasts')
    fig,ax=plt.subplots(figsize=(9,4.5))
    for i,c in enumerate(r['heldout']['comparisons']):
        lo,hi=c['improvement_CI95_pp2'];est=c['MSE_improvement_pp2'];ax.plot([lo,hi],[i]*2,color='C0');ax.plot(est,i,'o',color='C0')
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(4),['Distance + width vs distance','Distance + r vs distance','Distance + q vs distance','Distance + q + r vs distance + q']);ax.invert_yaxis();ax.set(xlabel='Held-out MSE improvement (pp²); positive favors added indicator',title='Same 52,664 objects; five scene folds; paired scene CI95 of fixed OOF losses')
    save(fig,'heldout_comparison')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args();plots(a.results,a.output)
