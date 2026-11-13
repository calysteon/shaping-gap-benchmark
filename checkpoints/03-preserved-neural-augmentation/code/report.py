"""Reconstruct the checkpoint tables and figures from complete recorded scores."""
from pathlib import Path
import argparse,json,os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/report');p.add_argument('--paper-assets',type=Path);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 os.environ['MPLCONFIGDIR']=str(out/'plot-cache');os.environ['XDG_CACHE_HOME']=str(out/'cache')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 data=ROOT/'data/results';s59=json.loads((data/'p59-preserved-return-augmentation-001/summary.json').read_text());s60=json.loads((data/'p60-shared-observer-transport-001/summary.json').read_text());ab=json.loads((ROOT/'results/transport-direction-ablation.json').read_text())['records'];records=[];means=[]
 for model in ['0.6B','1.7B']:
  for case in range(6):
   base=next(r for r in s59['results'] if r['model']==model and r['case']==case and r['kind']=='baseline');b60=next(r for r in s60['results'] if r['model']==model and r['case']==case and r['kind']=='baseline');prior=np.min([b['r2'] for b in base['blocks']],0)>=.5
   for kind in ['acquired','random_same_norm','random_same_action']:
    r=next(r for r in s59['results'] if r['model']==model and r['case']==case and r['kind']==kind and r['fraction']==.02);q=next(r for r in s60['results'] if r['model']==model and r['case']==case and r['kind']==kind)
    capacities=[z['nonlinear_temporal_capacity'] for z in r['blocks']]+[q['capability_nonlinear_capacity']];bases=[z['nonlinear_temporal_capacity'] for z in base['blocks']]+[b60['capability_nonlinear_capacity']];delta=np.array(capacities)-bases
    records.append(dict(model=model,case=case,seed=r['seed'],kind=kind,capacity=capacities,baseline_capacity=bases,delta=delta.tolist(),stable_old=q['fresh_stable_old_functions'],raw_retained=q['raw_old_retained'],transport_retained=q['transported_old_retained'],all_fresh_retained=bool(q['fresh_stable_old_functions'] and not q['transported_old_losses']),
     prior_old=int(prior.sum()),raw_retained_unfiltered=int(sum(prior&(np.array(q['score_vectors']['raw_old'])>=.5))),transport_retained_unfiltered=int(sum(prior&(np.array(q['score_vectors']['transported_old'])>=.5))),baseline_retained_unfiltered=int(sum(prior&(np.array(q['score_vectors']['baseline'])>=.5))),
     reliable_new=q['reliably_new_nonlinear_targets'],positive_three_blocks=bool(min(delta)>0)))
  for kind in ['acquired','random_same_norm','random_same_action']:
   rows=[r for r in records if r['model']==model and r['kind']==kind];means.append(dict(model=model,kind=kind,mean_delta=np.mean([r['delta'] for r in rows],0).tolist(),positive_three=sum(r['positive_three_blocks'] for r in rows)))
 totals={}
 for model in ['0.6B','1.7B']:
  rows=[r for r in records if r['model']==model and r['kind']=='acquired'];b=[r for r in ab if r['model']==model]
  totals[model]={k:sum(r[k] for r in rows) for k in ['stable_old','raw_retained','transport_retained','prior_old','raw_retained_unfiltered','transport_retained_unfiltered','baseline_retained_unfiltered']}
  totals[model].update(complete_fresh_cases=sum(r['all_fresh_retained'] for r in rows),positive_three_cases=sum(r['positive_three_blocks'] for r in rows),normalization_only=sum(r['normalization_only'] for r in b),learned_diagonal=sum(r['learned_diagonal'] for r in b),retained_all_three=sum(r['transported_retained_all_three_blocks'] for r in b),complete_three_cases=sum(not r['transported_losses_across_three'] for r in b))
 report=dict(cases=records,mean_changes=means,retention=totals,scope='Prespecified+2percent updates; twelve charts, two test blocks and fresh validation. Shared interfaces use only training feature pairs. Diagonal/normalization and three-block retention ablations are post-hoc.')
 (out/'tables.json').write_text(json.dumps(report,indent=2)+'\n')
 text=['Checkpoint 03: preserved neural augmentation and reusable observation interfaces','',json.dumps(totals,indent=2),'','All-model primary mean processing-capacity changes: ',json.dumps(means,indent=2)]
 (out/'RESULTS.txt').write_text('\n'.join(text)+'\n')
 plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(6.5,2.7));colors=['#8998a3','#23799b','#d2a45c','#946ca4'];labels=['Baseline','Acquired','Same-norm random','Same-action random'];x=np.arange(3)
 sample=next(r for r in records if r['model']=='1.7B' and r['case']==1 and r['kind']=='acquired')
 for j,(kind,label,color) in enumerate(zip(['baseline','acquired','random_same_norm','random_same_action'],labels,colors)):
  y=sample['baseline_capacity'] if kind=='baseline' else next(r['capacity'] for r in records if r['model']=='1.7B' and r['case']==1 and r['kind']==kind)
  ax.bar(x+(j-1.5)*.19,y,.18,color=color,label=label)
 ax.set_xticks(x,['Test 1','Test 2','Fresh validation']);ax.set_ylabel('Nonlinear temporal score');ax.legend(ncol=4,fontsize=7,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01));ax.set_ylim(0,8);fig.tight_layout();fig.savefig(out/'controlled-example.png',dpi=240);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6.5,2.8));rows=[r for r in records if r['kind']=='acquired'];x=np.arange(len(rows));markers=['o','s','^']
 for j,label in enumerate(['Test 1','Test 2','Fresh validation']):ax.scatter(x+(j-1)*.15,[r['delta'][j] for r in rows],s=20,marker=markers[j],label=label)
 ax.axhline(0,color='#666',lw=.8);ax.set_xticks(x,[f"{r['model']} / {r['seed']}" for r in rows],rotation=42,ha='right',fontsize=7);ax.set_ylabel('Change in nonlinear temporal score');ax.legend(ncol=3,fontsize=7,frameon=False,loc='lower left');ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'all-primary-updates.png',dpi=240);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6.5,2.7));x=np.arange(2);den=np.array([totals[m]['stable_old'] for m in ['0.6B','1.7B']]);series=[('raw_retained','Direct old readout','#8998a3'),('transport_retained','Shared interface','#23799b')]
 for j,(key,label,color) in enumerate(series):
  vals=[totals[m][key] for m in ['0.6B','1.7B']];bars=ax.bar(x+(j-.5)*.25,np.array(vals)/den*100,.24,color=color,label=label)
  for bar,n,d in zip(bars,vals,den):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{n}/{d}',ha='center',fontsize=8)
 ax.set_xticks(x,['0.6B','1.7B']);ax.set_ylim(0,114);ax.set_ylabel('Reference-reliable programs retained (%)');ax.legend(ncol=2,fontsize=7,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01));fig.tight_layout();fig.savefig(out/'program-retention.png',dpi=240);plt.close(fig)
 if a.paper_assets:
  paper=a.paper_assets.resolve();paper.mkdir(parents=True,exist_ok=True)
  import shutil
  for n in ['controlled-example.png','all-primary-updates.png','program-retention.png']:shutil.copy2(out/n,paper/n)
  table=['\\begin{tabular}{lrrrrr}\\toprule','Model & Old & Direct & Shared & Complete cases & Three-block\\\\\\midrule']
  for m,t in totals.items():table.append(f"{m} & {t['stable_old']} & {t['raw_retained']} & {t['transport_retained']} & {t['complete_fresh_cases']}/6 & {t['retained_all_three']}\\\\")
  table+=['\\bottomrule\\end{tabular}'];(paper/'retention-table.tex').write_text('\n'.join(table)+'\n')
  table=['\\begin{tabular}{llrrr}\\toprule','Model & Addition & Test 1 & Test 2 & Fresh\\\\\\midrule']
  names={'acquired':'Acquired','random_same_norm':'Same-norm random','random_same_action':'Same-action random'}
  for r in means:table.append(f"{r['model']} & {names[r['kind']]} & "+' & '.join(f'{v:+.3f}' for v in r['mean_delta'])+'\\\\')
  table+=['\\bottomrule\\end{tabular}'];(paper/'means-table.tex').write_text('\n'.join(table)+'\n')
  table=['\\begin{tabular}{llrrrrr}\\toprule','Model & Seed & $\\Delta_1$ & $\\Delta_2$ & $\\Delta_3$ & Direct & Shared\\\\\\midrule']
  for r in rows:table.append(f"{r['model']} & {r['seed']} & "+' & '.join(f'{v:+.3f}' for v in r['delta'])+f" & {r['raw_retained']}/{r['stable_old']} & {r['transport_retained']}/{r['stable_old']}\\\\")
  table+=['\\bottomrule\\end{tabular}'];(paper/'cases-table.tex').write_text('\n'.join(table)+'\n')
  table=['\\begin{tabular}{llrrrr}\\toprule','Study & Model & Native sequences & Source/fit s & Whole run s & Peak RSS MiB\\\\\\midrule']
  for label,study in [('P59',s59),('P60',s60)]:
   for m in study['models']:table.append(f"{label} & {m['model']} & {m['native_work']['sequences']:,} & {m.get('source_seconds',m.get('source_fit_seconds')):.2f} & {m['wall_seconds']:.1f} & {m['process_peak_rss_bytes']/2**20:.1f}\\\\")
  table+=['\\bottomrule\\end{tabular}'];(paper/'cost-table.tex').write_text('\n'.join(table)+'\n')
 print(json.dumps(totals,indent=2))

if __name__=='__main__':main()
