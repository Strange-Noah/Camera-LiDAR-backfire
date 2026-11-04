"""Regenerate three figures from subgroup summaries; never reads predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIST=['[0,10)','[10,20)','[20,30)','[30,40)','[40,∞)']
POINTS=['0–2','3–10','>10']

def render(data,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    colors=plt.get_cmap('tab10').colors
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(output/(name+'.'+ext),dpi=180)
        plt.close(fig)
    x=np.arange(1,6)
    fig,ax=plt.subplots(figsize=(8,5))
    for i,row in enumerate(data['distance']):
        if row['N']==0:continue
        mean=np.array(row['G_mean']);sd=np.array(row['G_sd'])
        ax.errorbar(x,mean,yerr=sd,color=colors[i],marker='o',label=DIST[i]+' m')
        for bracket in row['upward_brackets']:
            ax.axvspan(*bracket,color=colors[i],alpha=.04)
    ax.axhline(0,color='black',lw=1);ax.set(xlabel='Rotation σ (degrees per axis)',ylabel='G (percentage points)',title='Car: conditional backfire (mean ± seed SD)');ax.legend()
    save(fig,'figure1_backfire')
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    values=[]
    for field in ['B','D_mean','G_mean']:
        a=np.full((5,3),np.nan)
        for i in range(5):
            for j in range(3):
                r=data['distance_points'][i][j]
                if r['N']:a[i,j]=r[field] if field=='B' else r[field][-1]
        values.append(a)
    finite=np.concatenate([v[np.isfinite(v)] for v in values]);lim=max(float(np.abs(finite).max()),.1) if len(finite) else 1
    for ax,a,label in zip(axes,values,['Initial benefit B','Degradation D, σ=5°','Backfire G, σ=5°']):
        im=ax.imshow(a,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
        for i in range(5):
            for j in range(3):
                r=data['distance_points'][i][j];v=a[i,j]
                txt='empty' if not r['N'] else f'{v:.1f}\nN={r["N"]}, S={r["scenes"]}'
                if r['N'] and (r['N']<200 or r['scenes']<10):txt+='*'
                ax.text(j,i,txt,ha='center',va='center',fontsize=6)
        ax.set_xticks(range(3),POINTS);ax.set_yticks(range(5),DIST);ax.set_title(label);ax.set_xlabel('GT LiDAR points')
        fig.colorbar(im,ax=ax,label='percentage points')
    fig.suptitle('Car distance × LiDAR points; * limited support (N<200 or scenes<10)')
    save(fig,'figure2_benefit_degradation')
    fig,ax=plt.subplots(figsize=(8,5))
    for i,r in enumerate(data['distance']):
        if not r['N']:continue
        ax.plot(x,r['D_mean'],marker='o',color=colors[i],label=DIST[i]+' m')
        ax.axhline(r['B'],color=colors[i],ls='--',lw=1)
    ax.set(xlabel='Rotation σ (degrees per axis)',ylabel='Percentage points',title='Car: degradation D (solid) vs initial benefit B (dashed)');ax.legend()
    save(fig,'figure3_D_vs_B')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    render(json.loads(a.summary.read_text()),a.output)
