from pathlib import Path
import json,time,hashlib,argparse,gc
import numpy as np
from numpy.polynomial.legendre import legval
LAB=Path(__file__).resolve().parent.parent;RUN=LAB/'results/p62-feedback-acquired-update-reuse-001';PARENT=LAB/'results/p61-learned-update-compatibility-001';OUT=LAB.parent.parent/'outputs/neural-stimulus/update-policy'
def load(p):
 with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def truth(u,desc,start,n):
 ix=np.arange(start,start+n);ys=[]
 for d in desc:
  y=np.ones(n)
  for lag,degree in d['parts']:
   c=np.zeros(degree+1);c[-1]=1;y*=np.sqrt(2*degree+1)*legval(u[ix-lag],c)
  ys.append(y)
 return np.stack(ys,1)
def rr(pred,y):return 1-np.sum((pred-y)**2,0)/np.maximum(np.sum((y-y.mean(0))**2,0),1e-30)
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');a=p.parse_args();start=time.perf_counter();s=json.loads((RUN/'summary.json').read_text());ps=json.loads((PARENT/'summary.json').read_text());c=s['config'];dec=json.loads((RUN/'selections.json').read_text());seals=0;vectors=0;maxerr=0.;records=[]
 seal=json.loads((RUN/'selections-sealed.json').read_text());assert sha(PARENT/'summary.json')==seal['parent_summary_sha256']
 for n,h in seal['files'].items():assert sha(RUN/n)==h;seals+=1
 for model in c['models']:
  d=RUN/'models'/model;pd=PARENT/'models'/model
  for n,h in json.loads((d/'inherited-sealed.json').read_text())['files'].items():assert sha(pd/n)==h;seals+=1
  desc=json.loads((pd/'targets.json').read_text())
  for case in range(6):
   rows=[r for r in ps['results'] if r['model']==model and r['case']==case];b=[next(r for r in rows if r['block']==k and r['slot']==0) for k in range(2)];old=np.min([r['r2'] for r in b],0)>=.5;eligible=[]
   for slot in range(1,5):
    rr0=[next(r for r in rows if r['block']==k and r['slot']==slot) for k in range(2)];delta=[(rr0[k]['nonlinear_temporal_capacity']-b[k]['nonlinear_temporal_capacity'])/max(b[k]['nonlinear_temporal_capacity'],.25) for k in range(2)];kept=old&(np.min([r['transported_old_r2'] for r in rr0],0)>=.5)
    if old.sum()>0 and all(r['bounded'] for r in b+rr0) and min(delta)>=c['quality_acceptance'] and kept.sum()==old.sum():eligible.append((min(delta),-abs(rr0[0]['fraction']),-slot,slot))
   choice=max(eligible)[3] if eligible else 0;expected=next(r for r in dec if r['model']==model and r['case']==case);assert choice==expected['selected_slot'];assert np.array_equal(np.flatnonzero(old),expected['prior_old_ids'])
  for block in range(2):
   data=load(d/f'assessment-{block}.npz');Y=truth(data['drive'],desc,c['assessment_washout'],c['assessment_steps'])
   for case in range(6):
    assert np.array_equal(data['features'][case,0],data['features'][case,6]);bh=load(pd/f'readout-{case}-0.npz')
    for slot in range(7):
     h=load(pd/f'readout-{case}-{slot}.npz');f=load(pd/f'interface-{case}-{slot}.npz');X=data['features'][case,slot,c['assessment_washout']:].astype(float);preds={'r2':((X-h['feature_mean'])/h['feature_scale'])@h['readout']+h['target_mean'],'direct_old_r2':((X-bh['feature_mean'])/bh['feature_scale'])@bh['readout']+bh['target_mean'],'transported_old_r2':((X-f['new_feature_mean'])/f['new_feature_scale'])@f['transport']@bh['readout']+bh['target_mean']};row=next(r for r in s['results'] if r['model']==model and r['case']==case and r['slot']==slot and r['block']==block)
     for name,pred in preds.items():
      r=rr(pred,Y)
      if not row['bounded']:r[:]=-1
      err=float(np.max(abs(r-row[name])));maxerr=max(err,maxerr);assert err<1e-7,(model,case,slot,name,err);vectors+=1
  if a.native:
   from native import NativeField
   OUT.mkdir(parents=True,exist_ok=True)
   with (OUT/f'p62-verification-{model}-native.jsonl').open('w') as log:
    def emit(v):log.write(json.dumps(v)+'\n')
    receipt=json.loads((d/'checkpoint.json').read_text());native=NativeField(receipt['local_path'],emit,2,32);assert native.before[0]==next(m['parameter_sha256'] for m in s['models'] if m['model']==model);count=0
    for block in range(2):
     checks=load(d/f'assessment-{block}-native-checks.npz')
     for X,Y in zip(checks['inputs'],checks['outputs']):
      flat=X.reshape(-1,native.n);pad=(-len(flat))%32;pred=native.run(np.concatenate([flat,np.repeat(flat[-1:],pad,axis=0)]))[:len(flat)].reshape(Y.shape);assert np.array_equal(pred,Y);count+=len(flat)
    assert native.before==native.digest();records.append(dict(model=model,saved_outputs_bitexact=count,native_work=native.counts,weights_unchanged=True,parameter_sha256=native.before[0]));del native;gc.collect()
    import mlx.core as mx;mx.clear_cache()
 report=dict(score_vectors_reproduced=vectors,max_r2_error=maxerr,sealed_files_verified=seals,feedback_choices_reproduced=12,native=records,verification_seconds=time.perf_counter()-start);OUT.mkdir(exist_ok=True,parents=True);(OUT/('feedback-native-verification.json' if a.native else 'feedback-offline-verification.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
