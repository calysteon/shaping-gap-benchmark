"""Reconstruct both studies and publication tables/figures from recorded outcomes."""
from pathlib import Path
import argparse,json,importlib.util,os,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/report');p.add_argument('--paper-assets',type=Path);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 for i,study in enumerate(['update-policy-study','feedback-update-study']):
  spec=importlib.util.spec_from_file_location('analysis_'+str(i),ROOT/'code/studies'/study/'analyze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.LAB=ROOT/'data';m.OUT=out;m.main()
 shutil.copy2(ROOT/'results/cost-ledger.json',out/'cost-ledger.json')
 first=json.loads((out/'analysis.json').read_text());second=json.loads((out/'feedback-analysis.json').read_text());ledger=json.loads((out/'cost-ledger.json').read_text());assert ledger['all_new_native_sequences_including_verification']==1852268
 os.environ['MPLCONFIGDIR']=str(out/'plot-cache');os.environ['XDG_CACHE_HOME']=str(out/'cache')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 # Prominent mechanism + explicitly scoped success examples + complete-cohort retention.
 from matplotlib.patches import FancyBboxPatch
 fig=plt.figure(figsize=(6.5,3.4));grid=fig.add_gridspec(2,2,height_ratios=[.55,2]);top=fig.add_subplot(grid[0,:]);top.set_axis_off()
 steps=['Native\nexperience','Acquire update\n+ interface','Lock the\nconstruction','Reuse on\nfresh inputs']
 for i,label in enumerate(steps):
  x=i*.255;top.add_patch(FancyBboxPatch((x,.3),.225,.48,boxstyle='round,pad=0.01',facecolor='#e5f0f4',edgecolor='#237e9a',transform=top.transAxes));top.text(x+.1125,.54,label,ha='center',va='center',fontsize=9,transform=top.transAxes)
  if i<3:top.annotate('',xy=(x+.252,.54),xytext=(x+.23,.54),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#444'})
 ax=fig.add_subplot(grid[1,0]);examples=[r for r in second['cases'] if r['feedback']['joint_success']];x=np.arange(len(examples))
 for b,col in [(0,'#1b7f9c'),(1,'#dfaa52')]:ax.bar(x+(b-.5)*.28,[100*r['feedback']['relative_delta'][b] for r in examples],.27,label=f'Fresh stream {b+1}',color=col)
 ax.set_xticks(x,[r['model']+' / '+str(r['seed']) for r in examples]);ax.set_ylim(0,6);ax.set_ylabel('Processing-quality gain (%)');ax.set_title('Two strict joint-success contexts');ax.legend(frameon=False,fontsize=8,loc='upper left')
 ax=fig.add_subplot(grid[1,1]);pooled=second['aggregate'][-1]['arms']['feedback'];values=[pooled['direct_retained_old'],pooled['retained_old']];den=pooled['eligible_old'];bars=ax.bar([0,1],np.array(values)/den*100,color=['#98a4ae','#1b7f9c'],width=.55)
 for bar,n in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f'{n}/{den}',ha='center')
 ax.set_xticks([0,1],['Direct old readouts','Shared interface']);ax.set_ylim(0,115);ax.set_ylabel('Old program instances retained (%)');ax.set_title('All twelve contexts, retained choices');fig.tight_layout();fig.savefig(out/'checkpoint-overview.png',dpi=220);plt.close(fig)
 rows=second['cases'];x=np.arange(len(rows));fig,axes=plt.subplots(2,1,figsize=(6.5,4.2),gridspec_kw={'height_ratios':[1,1]})
 for b,marker in [(0,'o'),(1,'s')]:axes[0].scatter(x+(b-.5)*.16,[r['feedback']['relative_delta'][b]*100 for r in rows],marker=marker,s=35,label=f'Fresh stream {b+1}')
 axes[0].axhline(0,color='#555',lw=.8);axes[0].set_ylabel('Processing-quality change (%)');axes[0].legend(frameon=False,ncol=2);axes[0].grid(axis='y',alpha=.2)
 den=np.array([r['feedback']['eligible_old'] for r in rows]);kept=np.array([r['feedback']['retained_old'] for r in rows]);ax=axes[1];bars=ax.bar(x,np.where(den>0,kept/np.maximum(den,1)*100,np.nan),color='#1b7f9c',width=.6)
 for bar,n,d in zip(bars,kept,den):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5 if d else 2,f'{n}/{d}' if d else 'N/A',ha='center',fontsize=8)
 ax.set_ylim(0,116);ax.set_ylabel('Old programs retained (%)');labels=[f"{r['model']} / {r['seed']}\n{r['feedback']['fraction']:+.0%}" for r in rows]
 for ax in axes:ax.set_xticks(x,labels,rotation=45,ha='right',fontsize=7)
 axes[0].set_xticklabels([]);fig.tight_layout();fig.savefig(out/'all-contexts.png',dpi=220);plt.close(fig)
 if a.paper_assets:
  paper=a.paper_assets.resolve();paper.mkdir(parents=True,exist_ok=True)
  for name in ['checkpoint-overview.png','all-contexts.png']:shutil.copy2(out/name,paper/name)
  def write(name,cols,header,rows):
   text=['\\begin{tabular}{'+cols+'}\\toprule',header+'\\\\\\midrule',*rows,'\\bottomrule\\end{tabular}'];(paper/name).write_text('\n'.join(text)+'\n')
  labels={'feedback':'Feedback acquired','cross_chart':'Frozen predictor','fixed':'Fixed acquired +2\\%','random':'Random +2\\%'};table=[]
  for m in second['aggregate']:
   for arm in ['feedback','cross_chart','fixed','random']:
    r=m['arms'][arm];table.append(m['model']+' & '+labels[arm]+' & '+' & '.join(f'{v:+.3f}' for v in r['mean_delta'])+f" & {r['joint_successes']} & {r['retained_old']}/{r['eligible_old']}\\\\")
  write('aggregate-table.tex','llrrrr','Size & Rule & $\\Delta_1$ & $\\Delta_2$ & Joint & Old retained',table)
  table=[]
  for r in rows:
   q=r['feedback'];table.append(f"{r['model']} & {r['seed']} & {q['fraction']*100:+.0f}\\% & "+' & '.join(f'{v:+.3f}' for v in q['delta'])+f" & {q['retained_old']}/{q['eligible_old']} & {'yes' if q['joint_success'] else '--'}\\\\")
  write('cases-table.tex','llrrrrl','Size & Seed & Action & $\\Delta_1$ & $\\Delta_2$ & Retained & Joint',table)
  table=[]
  for m in first['aggregate']:
   r=m['arms']['policy'];table.append(f"{m['model']} & {r['updates']} & "+' & '.join(f'{v:+.3f}' for v in r['mean_delta'])+f" & {r['joint_successes']} & {r['retained_old']}/{r['eligible_old']}\\\\")
  write('transfer-table.tex','lrrrrr','Size & Updates & $\\Delta_1$ & $\\Delta_2$ & Joint & Old retained',table)
  table=[]
  for study,key in [('P61','models'),('P62','p62_models')]:
   for m in ledger[key]:table.append(f"{study} & {m['model']} & {m['native_work']['sequences']:,} & {m['wall_seconds']:.1f} & {m['process_peak_rss_bytes']/2**20:.1f}\\\\")
  write('cost-table.tex','llrrr','Study & Size & Native sequences & Wall s & Peak RSS MiB',table)
 print(json.dumps({'strict_examples':len(examples),'direct_old':pooled['direct_retained_old'],'shared_old':pooled['retained_old'],'reference_old':pooled['eligible_old']}))
if __name__=='__main__':main()
