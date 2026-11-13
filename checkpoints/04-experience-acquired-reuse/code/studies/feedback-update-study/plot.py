from pathlib import Path
import json,os,numpy as np
LAB=Path(__file__).resolve().parent.parent;OUT=LAB.parent.parent/'outputs/neural-stimulus/update-policy';os.environ['MPLCONFIGDIR']=str(OUT/'plot-cache');os.environ['XDG_CACHE_HOME']=str(OUT/'cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 s=json.loads((OUT/'feedback-analysis.json').read_text());rows=s['cases'];plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,1,figsize=(10,6.6),gridspec_kw={'height_ratios':[1.1,1]});x=np.arange(len(rows))
 for b,marker,color in [(0,'o','#197d9c'),(1,'s','#da9846')]:axes[0].scatter(x+(b-.5)*.16,[r['feedback']['relative_delta'][b]*100 for r in rows],marker=marker,s=38,color=color,label=f'Feedback-acquired, fresh block {b+1}')
 axes[0].scatter(x,[min(r['oracle']['relative_delta'])*100 for r in rows],marker='_',s=160,color='#8b8b8b',label='Best joint-success opportunity (retrospective)');axes[0].axhline(0,color='#555',lw=.8);axes[0].set_ylabel('Processing-quality change (%)');axes[0].legend(ncol=3,fontsize=8,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False);axes[0].grid(axis='y',alpha=.2)
 eligible=np.array([r['feedback']['eligible_old'] for r in rows]);kept=np.array([r['feedback']['retained_old'] for r in rows]);raw=np.array([r['feedback']['direct_retained_old'] for r in rows]);den=np.maximum(eligible,1)
 axes[1].bar(x-.16,np.where(eligible>0,raw/den*100,np.nan),.3,color='#9aa6af',label='Direct old readouts');bars=axes[1].bar(x+.16,np.where(eligible>0,kept/den*100,np.nan),.3,color='#197d9c',label='Selected shared interface')
 for bar,n,d in zip(bars,kept,eligible):axes[1].text(bar.get_x()+bar.get_width()/2,(bar.get_height()+1.5 if d else 2), (f'{n}/{d}' if d else 'N/A'),ha='center',fontsize=7)
 axes[1].set_ylim(0,116);axes[1].set_ylabel('Existing programs retained (%)');axes[1].legend(ncol=2,fontsize=8,frameon=False,loc='lower left');labels=[f"{r['model']} / {r['seed']}\n{r['feedback']['fraction']:+.0%}" for r in rows]
 for ax in axes:ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=7)
 axes[0].set_xticklabels([]);axes[1].set_xlabel('Fresh runtime chart / selected source-action fraction (0% = baseline)');fig.tight_layout();fig.savefig(OUT/'feedback-transfer.png',dpi=210);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,3.8));groups=[r for r in s['aggregate'] if r['model']!='pooled'];x=np.arange(2)
 for a,arm,col in [(0,'feedback','#197d9c'),(1,'fixed','#a8adb2'),(2,'random','#d8a657')]:
  vals=[np.mean(r['arms'][arm]['mean_delta']) for r in groups];ax.bar(x+(a-1)*.24,vals,.23,color=col,label={'feedback':'Feedback-acquired choice','fixed':'Fixed acquired +2%','random':'Same-action random +2%'}[arm])
 ax.axhline(0,color='#555',lw=.8);ax.set_xticks(x,[r['model'] for r in groups]);ax.set_ylabel('Mean score change across both fresh blocks');ax.legend(fontsize=8,frameon=False);ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(OUT/'feedback-comparison.png',dpi=210);plt.close(fig)
if __name__=='__main__':main()
