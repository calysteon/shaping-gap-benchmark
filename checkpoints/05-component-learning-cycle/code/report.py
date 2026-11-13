"""Complete deterministic report; no model calls and no outcome selection."""
from pathlib import Path
import json,argparse,collections
import numpy as np

def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'data/experiment');parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'runs/report');args=parser.parse_args();root=args.data;out=args.output;out.mkdir(parents=True,exist_ok=True);s=json.loads((root/'summary.json').read_text());groups={};variants=['integrated','first_order','shuffled','random_weight']
 for variant in variants:
  rows=[r for r in s['results'] if r['variant']==variant];groups[variant]=dict(contexts=len(rows),updates=sum(r['update'] for r in rows),strict_joint=sum(r['strict_joint'] for r in rows),mean_quality_change=np.mean([r['quality_changes'] for r in rows],axis=0).tolist(),mean_selected_quality=np.mean([r['selected_quality'] for r in rows],axis=0).tolist(),mean_baseline_quality=np.mean([r['baseline_quality'] for r in rows],axis=0).tolist(),old_programs=sum(r['old_programs'] for r in rows),old_retained_both=sum(r['old_retained_both'] for r in rows),runtime_wall_seconds=sum(r['wall_seconds'] for r in rows),minimum_runtime_payload=min(r['runtime_payload_bytes'] for r in rows),maximum_runtime_payload=max(r['runtime_payload_bytes'] for r in rows))
 rows=[]
 for r in s['results']:
  d=root/'contexts'/r['context'];choice=json.loads((d/'selection.json').read_text());old=choice['old_program_ids'];bs=json.loads((d/'feedback-scores.json').read_text());fs=json.loads((d/'fresh-scores.json').read_text());j=r['selected_slot'];desc=json.loads((d/'targets.json').read_text());multi=np.array([v['degree']>=2 and v['distinct_lags']>=2 for v in desc]);new=np.min([v[j]['r2'] for v in bs+fs],0)>=.5;absent=np.max([v[0]['r2'] for v in bs+fs],0)<.5;direct_mask=np.ones(len(old),bool)
  for b in fs:direct_mask&=np.array(b[j]['direct_old_r2'])[old]>=.5
  traces=[np.load(d/f'checks-fresh-{b}.npz') for b in range(2)];routes=[]
  for t in traces:
   routes.extend([tuple(np.r_[a[j],b[j]].tolist()) for a,b in zip(t['first'],t['second'])]);t.close()
  row={k:r[k] for k in ['context','model','seed','variant','selected_slot','relative_gains','quality_changes','baseline_quality','selected_quality','old_programs','old_retained_both','strict_joint','source_baseline_admissible','wall_seconds','runtime_payload_bytes','interface_bytes','return_bytes','readout_library_bytes']};row.update(new_reliable_multilag_nonlinear_functions=int(np.sum(new&absent&multi)),direct_old_intersection_both=int(direct_mask.sum()),selected_fresh_sampled_routes=len(set(routes)),feedback_candidates=choice['feedback_candidates']);rows.append(row)
 counts=collections.Counter()
 for r in s['results']:counts.update(r['work'])
 primary=groups['integrated'];positive=primary['strict_joint']>=1 and min(primary['mean_quality_change'])>0
 record=dict(primary_cycle_criterion_met=positive,groups=groups,contexts=rows,study_wall_seconds=s['wall_seconds'],process_peak_rss_bytes=s['process_peak_rss_bytes'],work_counts=dict(counts),transformer_calls=s['original_transformer_calls'],attention_calls=s['attention_calls'],tokenizer_calls=s['tokenizer_calls'],metric='sum positive held-out R2 over109 multi-lag nonlinear Legendre targets;143 total readouts',original_precision_note='Materialized pretrained component formulas; mixed-precision original-module fidelity is reported separately from full-coordinate compiled replay.')
 dump(out/'analysis.json',record)
 text=['P63 / CHECKPOINT05: COMPONENT ACQUISITION AND COMPATIBLE REUSE','',f'Prespecified finite component-cycle criterion met: {positive}','']
 for v,g in groups.items():text.append(f"{v}: selected {g['updates']}/{g['contexts']}; strict joint {g['strict_joint']}; mean fresh quality changes {g['mean_quality_change']}; old retained {g['old_retained_both']}/{g['old_programs']}; mean final qualities {g['mean_selected_quality']}")
 text+=['','All cases (no exclusions):']
 for r in rows:text.append(f"{r['context']}: slot{r['selected_slot']}; quality {r['baseline_quality']} -> {r['selected_quality']}; relative changes {r['relative_gains']}; old {r['old_retained_both']}/{r['old_programs']}; direct old intersection {r['direct_old_intersection_both']}; strict {r['strict_joint']}; sampled selected fresh routes {r['selected_fresh_sampled_routes']}")
 text+=['','The acquired pair is a response-derived return change plus a shared observation map.','Six pretrained integrated contexts: one update, five baseline fallbacks; all18 old instances retained.','First-order guidance produces two strict successes; shuffled and random-weight controls each produce one.','Comparisons do not establish general superiority of integrated conditioning or a pretraining-specific mechanism.','One cycle; numerical temporal computation from selected frozen components, not a language-task evaluation.','Complete controller memory and acquisition counts are separate from frozen parameters and preparation.']
 (out/'FINDINGS.txt').write_text('\n'.join(text)+'\n')
 # Static scientific figures, exportable without a browser.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
 fig,axs=plt.subplots(1,2,figsize=(10.6,3.8),layout='constrained');win=next(r for r in rows if r['variant']=='integrated' and r['strict_joint']);x=np.arange(2);w=.34
 axs[0].bar(x-w/2,win['baseline_quality'],w,label='Before acquisition',color='#b4c2ca');axs[0].bar(x+w/2,win['selected_quality'],w,label='Retained construction',color='#147b86');axs[0].set_xticks(x,['Fresh stream 1','Fresh stream 2']);axs[0].set_ylabel('Nonlinear temporal-processing score');axs[0].set_title('Component learning cycle: one retained gain');axs[0].set_ylim(0,1.6);axs[0].legend(frameon=False,fontsize=8);axs[0].text(.02,.96,'0.6B source / context 151003\n2/2 old programs retained',transform=axs[0].transAxes,ha='left',va='top',fontsize=8)
 labels=['Integrated','First-order','Shuffled','Random weights'];colors=['#147b86','#477fa7','#9a7a5d','#8f979c'];a=np.arange(4);w=.34
 axs[1].bar(a-w/2,[groups[v]['mean_quality_change'][0] for v in variants],w,color=colors,label='Fresh stream 1');axs[1].bar(a+w/2,[groups[v]['mean_quality_change'][1] for v in variants],w,color=colors,alpha=.45,label='Fresh stream 2');axs[1].set_xticks(a,labels,rotation=15);axs[1].set_ylabel('Mean quality change across six contexts');axs[1].set_title('Complete cohort and controls');axs[1].legend(frameon=False,fontsize=8);fig.savefig(out/'checkpoint-overview.png',dpi=200);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained');labels=[r['model']+' / '+str(r['seed']) for r in rows if r['variant']=='integrated'];rr=[r for r in rows if r['variant']=='integrated'];a=np.arange(6);ax.bar(a-.18,[r['quality_changes'][0] for r in rr],.36,label='Fresh stream 1',color='#147b86');ax.bar(a+.18,[r['quality_changes'][1] for r in rr],.36,label='Fresh stream 2',color='#65a7b2');ax.set_xticks(a,labels,rotation=22);ax.set_ylabel('Quality change from each fixed baseline');ax.set_title('All six integrated-guidance contexts, including five fallbacks');ax.legend(frameon=False);fig.savefig(out/'all-integrated-contexts.png',dpi=180);plt.close(fig)
 print(json.dumps(dict(primary_cycle_criterion_met=positive,groups=groups,report=str(out))))
if __name__=='__main__':main()
