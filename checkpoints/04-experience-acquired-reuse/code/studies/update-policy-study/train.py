"""Train only on historical recorded P59 charts; seal before fresh chart acquisition."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,numpy as np
from policy import source_features,fit_policy,predict,choose,FEATURES
from capacity_engine import capacity_targets
HERE=Path(__file__).resolve().parent;LAB=HERE.parent

def load(p):
 with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 started=time.perf_counter();plan=json.loads((HERE/'protocol.json').read_text());c=plan['planned_config']['config'];out=LAB/'results/p61-learned-update-compatibility-001';out.mkdir(exist_ok=False);(out/'helpers').mkdir();(out/'models').mkdir()
 for name in ['native.py','capacity_engine.py','stimulus.py','policy.py','augmentation.py']:__import__('shutil').copy2(HERE/name,out/'helpers'/name)
 __import__('shutil').copy2(__file__,out/'training_snapshot.py');__import__('shutil').copy2(HERE/'protocol.json',out/'protocol.json')
 parent=LAB/'results'/c['training_parent'];summary=json.loads((parent/'summary.json').read_text());cfg=summary['config'];features=[];targets=[];rows=[];allfaces=[]
 for label in c['models']:
  d=parent/'models'/label;data=load(d/'assessment.npz');meta=json.loads((d/'systems.json').read_text());start=cfg['evaluation_washout'];N=cfg['evaluation_training'];desc=json.loads((d/'targets.json').read_text());multi=np.array([q['degree']>=2 and q['distinct_lags']>=2 for q in desc]);rowsum=[r for r in summary['results'] if r['model']==label]
  for i in range(6):
   base=next(r for r in rowsum if r['case']==i and r['kind']=='baseline');old=np.min([b['r2'] for b in base['blocks']],0)>=.5;bh=load(d/f'readout-{i}-0.npz');B=data['features'][i,0,start:start+N]
   for fraction in c['signed_action_fractions']:
    m=next(m for m in meta[i] if m['kind']=='acquired' and m['fraction']==fraction);j=m['slot'];r=next(r for r in rowsum if r['case']==i and r['slot']==j);X=data['features'][i,j,start:start+N];f,face=source_features(B,X,data['drive'],start,fraction,c['feature_seed']);ret=[];delta=[]
    for b in range(2):
     ix=start+N+b*cfg['evaluation_test'];Y,_=capacity_targets(data['drive'],ix,cfg['evaluation_test']);Xtest=data['features'][i,j,ix:ix+cfg['evaluation_test']].astype(float);pred=((Xtest-face['new_feature_mean'])/face['new_feature_scale'])@face['transport']@bh['readout']+bh['target_mean'];r2=1-np.sum((pred-Y)**2,0)/np.sum((Y-Y.mean(0))**2,0);ret.append(bool(np.all(r2[old]>=.5) and r['bounded']));delta.append((r['blocks'][b]['nonlinear_temporal_capacity']-base['blocks'][b]['nonlinear_temporal_capacity'])/max(base['blocks'][b]['nonlinear_temporal_capacity'],.25))
    y=[float(np.clip(min(delta),-2,2)),float(all(ret))];features.append(f);targets.append(y);rows.append(dict(model=label,case=i,seed=m['seed'],fraction=fraction,quality_min_relative=y[0],retained_both=bool(y[1]),source_features=f.tolist()))
    print(json.dumps(dict(stage='historical_features',model=label,case=i,fraction=fraction)),flush=True)
 X=np.array(features);Y=np.array(targets);groups=np.array([r['seed'] for r in rows]);grid=[];best=None
 for bandwidth in c['kernel_bandwidth_factors']:
  for ridge in c['kernel_ridge_grid']:
   pred=np.empty_like(Y)
   for group in sorted(set(groups)):
    train=groups!=group;test=~train;p=fit_policy(X[train],Y[train],bandwidth,ridge);pred[test]=predict(X[test],p)
   norm=np.maximum(Y.std(0),.1);loss=float(np.mean(((pred-Y)/norm)**2));grid.append(dict(bandwidth=bandwidth,ridge=ridge,grouped_cv_loss=loss))
   if best is None or loss<best[0]:best=(loss,bandwidth,ridge,pred.copy())
 p=fit_policy(X,Y,best[1],best[2]);np.savez_compressed(out/'policy.npz',**p);np.savez_compressed(out/'training-data.npz',features=X,targets=Y,groups=groups,grouped_cv_predictions=best[3]);dump(out/'training-rows.json',rows)
 cv=[]
 for label in c['models']:
  for i in range(6):
   ids=[j for j,r in enumerate(rows) if r['model']==label and r['case']==i];j=choose(best[3][ids],c);cv.append(dict(model=label,case=i,selected_slot=j,selected_fraction=c['signed_action_fractions'][j-1] if j else 0.,observed_target=Y[ids[j-1]].tolist() if j else [0.,1.]))
 coeff=sum(v.size for v in p.values());assert coeff<=c['policy_max_retained_coefficients']
 report=dict(training_charts=12,training_candidates=len(rows),feature_names=FEATURES,hyperparameter_grid=grid,selected_hyperparameters=dict(bandwidth=best[1],ridge=best[2],grouped_cv_loss=best[0]),grouped_cv_selections=cv,retained_policy_coefficients=coeff,retained_policy_array_bytes=sum(v.nbytes for v in p.values()),training_cpu_seconds=time.perf_counter()-started,information='Historical target feedback only. New charts, new source and new assessment not yet acquired.')
 dump(out/'training-summary.json',report);dump(out/'policy-sealed.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),fresh_chart_acquisition_started=False,files={n:sha(out/n) for n in ['policy.npz','training-data.npz','training-rows.json','training-summary.json']},source_sha256={p.name:sha(p) for p in HERE.glob('*.py')}));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
