"""Post-hoc source-only diagonal controls and three-block interface backtest."""
from pathlib import Path
import json
import numpy as np
from capacity_engine import capacity_targets

LAB=Path(__file__).resolve().parents[1]
def main():
 p59=LAB/'results/p59-preserved-return-augmentation-001';p60=LAB/'results/p60-shared-observer-transport-001';s=json.loads((p60/'summary.json').read_text());ps=json.loads((p59/'summary.json').read_text());c=s['config'];pc=ps['config'];rows=[]
 for model in ['0.6B','1.7B']:
  d=p60/'models'/model;pd=p59/'models'/model;data=np.load(d/'validation.npz');past=np.load(pd/'assessment.npz');Y,_=capacity_targets(data['drive'],c['washout'],c['test_steps']);meta=json.loads((d/'systems.json').read_text())
  for i,ms in enumerate(meta):
   j=next(j for j,m in enumerate(ms) if m['kind']=='acquired');slot=ms[j]['slot'];h=np.load(pd/f'readout-{i}-{slot}.npz');bh=np.load(pd/f'readout-{i}-0.npz');face=np.load(d/f'interface-{i}-{j}.npz');T=face['transport'];X=data['features'][i,j,c['washout']:].astype(float);ZX=(X-h['feature_mean'])/h['feature_scale']
   start=pc['evaluation_washout'];N=pc['evaluation_training'];A=(past['features'][i,slot,start:start+N].astype(float)-h['feature_mean'])/h['feature_scale'];B=(past['features'][i,0,start:start+N].astype(float)-bh['feature_mean'])/bh['feature_scale'];ridge=c['transport_ridge']*N;diag=(np.sum(A*B,0)+ridge)/(np.sum(A*A,0)+ridge)
   record=next(r for r in s['results'] if r['model']==model and r['case']==i and r['kind']=='acquired');base=next(r for r in ps['results'] if r['model']==model and r['case']==i and r['kind']=='baseline');old=(np.min([b['r2'] for b in base['blocks']],0)>=.5)&(np.array(record['score_vectors']['baseline'])>=.5)
   counts={}
   for name,D in [('normalization_only',np.ones(40)),('learned_diagonal',diag)]:
    prediction=(ZX*D)@bh['readout']+bh['target_mean'];rr=1-np.sum((prediction-Y)**2,0)/np.sum((Y-Y.mean(0))**2,0);counts[name]=int(sum(old&(rr>=.5)))
   transported=[]
   for block in range(2):
    begin=pc['evaluation_washout']+pc['evaluation_training']+block*pc['evaluation_test'];truth,_=capacity_targets(past['drive'],begin,pc['evaluation_test']);F=past['features'][i,slot,begin:begin+pc['evaluation_test']].astype(float);P=((F-h['feature_mean'])/h['feature_scale'])@T@bh['readout']+bh['target_mean'];transported.append(1-np.sum((P-truth)**2,0)/np.sum((truth-truth.mean(0))**2,0))
   transported.append(np.array(record['score_vectors']['transported_old']));keep=np.min(transported,0)>=.5
   rows.append(dict(model=model,case=i,old=int(sum(old)),direct=record['raw_old_retained'],**counts,full_transport=record['transported_old_retained'],transported_retained_all_three_blocks=int(sum(old&keep)),transported_losses_across_three=np.flatnonzero(old&~keep).tolist()))
 out=LAB.parent.parent/'outputs/neural-stimulus/preserved-augmentation';(out/'transport-direction-ablation.json').write_text(json.dumps(dict(classification='Post-hoc source-only diagonal/normalization controls and retained-program backtest; source fits use fixed training features.',records=rows),indent=2)+'\n')
 print(json.dumps({k:sum(r[k] for r in rows) for k in ['old','direct','normalization_only','learned_diagonal','full_transport','transported_retained_all_three_blocks']},indent=2))

if __name__=='__main__':main()
