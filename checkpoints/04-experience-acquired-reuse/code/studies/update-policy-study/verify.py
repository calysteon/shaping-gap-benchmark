"""Independent target/readout/interface checks plus recorded native re-execution."""
from pathlib import Path
import json,hashlib,argparse,time,gc
import numpy as np
from numpy.polynomial.legendre import legval
HERE=Path(__file__).resolve().parent;LAB=HERE.parent;RUN=LAB/'results/p61-learned-update-compatibility-001';OUT=LAB.parent.parent/'outputs/neural-stimulus/update-policy'
def load(p):
 with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def target(u,desc,start,n):
 ix=np.arange(start,start+n);ys=[]
 for d in desc:
  y=np.ones(n)
  for lag,degree in d['parts']:
   coeff=np.zeros(degree+1);coeff[-1]=1;y*=np.sqrt(2*degree+1)*legval(u[ix-lag],coeff)
  ys.append(y)
 return np.stack(ys,1)
def score(pred,Y):return 1-np.sum((pred-Y)**2,0)/np.maximum(np.sum((Y-Y.mean(0))**2,0),1e-30)
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');a=p.parse_args();started=time.perf_counter();summary=json.loads((RUN/'summary.json').read_text());c=summary['config'];maxerr=0.;tmax=0.;seals=0;vectors=0;native_records=[]
 from policy import fit_policy
 tr=load(RUN/'training-data.npz');saved=load(RUN/'policy.npz');refit=fit_policy(tr['features'],tr['targets'],float(saved['bandwidth']),float(saved['ridge']));policy_refit_error=max(float(np.max(abs(refit[k]-saved[k]))) for k in saved);assert policy_refit_error<1e-10
 for name,h in json.loads((RUN/'policy-sealed.json').read_text())['files'].items():assert sha(RUN/name)==h;seals+=1
 for model in c['models']:
  d=RUN/'models'/model
  for seal in ['couplings-sealed.json','decisions-sealed.json','programs-sealed.json']:
   for name,h in json.loads((d/seal).read_text())['files'].items():assert sha(d/name)==h;seals+=1
  source=load(d/'source.npz');start=c['source_washout'];faces={};heads={}
  for case in range(6):
   base=source['features'][case,0,start:].astype(float);bm=base.mean(0);bs=np.maximum(base.std(0),1e-8);Y=(base-bm)/bs
   for slot in range(7):
    face=load(d/f'interface-{case}-{slot}.npz');X=source['features'][case,slot,start:].astype(float);mu=X.mean(0);sd=np.maximum(X.std(0),1e-8);Z=(X-mu)/sd;alpha=c['transport_ridge']*len(X);T=np.linalg.solve(Z.T@Z+alpha*np.eye(40),Z.T@Y+alpha*np.eye(40));tmax=max(tmax,float(np.max(abs(T-face['transport']))));assert np.max(abs(T-face['transport']))<1e-8;faces[case,slot]=face;heads[case,slot]=load(d/f'readout-{case}-{slot}.npz')
  desc=json.loads((d/'targets.json').read_text())
  for block in range(2):
   data=load(d/f'assessment-{block}.npz');Y=target(data['drive'],desc,c['assessment_washout'],c['assessment_steps'])
   for case in range(6):
    assert np.array_equal(data['features'][case,0],data['features'][case,6])
    for slot in range(7):
     X=data['features'][case,slot,c['assessment_washout']:].astype(float);h=heads[case,slot];bh=heads[case,0];f=faces[case,slot];predictions={'r2':((X-h['feature_mean'])/h['feature_scale'])@h['readout']+h['target_mean'],'direct_old_r2':((X-bh['feature_mean'])/bh['feature_scale'])@bh['readout']+bh['target_mean'],'transported_old_r2':((X-f['new_feature_mean'])/f['new_feature_scale'])@f['transport']@bh['readout']+bh['target_mean']};row=next(r for r in summary['results'] if r['model']==model and r['case']==case and r['slot']==slot and r['block']==block)
     for key,pred in predictions.items():
      r=score(pred,Y)
      if not row['bounded']:r[:]=-1
      err=float(np.max(abs(r-row[key])));maxerr=max(maxerr,err);assert err<1e-7,(model,case,slot,key,err);vectors+=1
  # Recompute the sealed selector from source-only features and saved kernel arrays.
  from policy import source_features,predict,choose
  policy=load(RUN/'policy.npz');decisions=json.loads((d/'decisions.json').read_text());meta=json.loads((d/'systems.json').read_text());charts=json.loads((d/'source-charts.json').read_text())
  for case in range(6):
   fs=[source_features(source['features'][case,0,start:],source['features'][case,slot,start:],source['drive'],start,meta[case][slot]['fraction'],c['feature_seed'])[0] for slot in range(1,5)];pred=predict(np.array(fs),policy);choice=choose(pred,c)
   if not (charts[case]['source_admissible'] and source['valid'][case,0] and source['valid'][case,choice]):choice=0
   assert choice==decisions[case]['selected_slot'];assert np.max(abs(pred-decisions[case]['predictions']))<1e-8
  if a.native:
   from native import NativeField
   OUT.mkdir(parents=True,exist_ok=True)
   with (OUT/f'p61-verification-{model}-native.jsonl').open('w') as log:
    def emit(v):log.write(json.dumps(v)+'\n')
    receipt=json.loads((d/'checkpoint.json').read_text());native=NativeField(receipt['local_path'],emit,2,c['native_batch_size']);assert native.before[0]==next(m['parameter_sha256'] for m in summary['models'] if m['model']==model);count=0
    for phase in ['source','assessment-0','assessment-1']:
     check=load(d/f'{phase}-native-checks.npz');native.stage=f'verification-{phase}'
     for X,Y in zip(check['inputs'],check['outputs']):
      n=native.n;flat=X.reshape(-1,n);pad=(-len(flat))%32;pred=native.run(np.concatenate([flat,np.repeat(flat[-1:],pad,axis=0)]))[:len(flat)].reshape(Y.shape);assert np.array_equal(pred,Y),(model,phase,float(np.max(abs(pred-Y))));count+=len(flat)
    assert native.before==native.digest();native_records.append(dict(model=model,saved_outputs_bitexact=count,native_work=native.counts,parameter_sha256=native.before[0],weights_unchanged=True));del native;gc.collect()
    import mlx.core as mx;mx.clear_cache()
 report=dict(score_vectors_reproduced=vectors,max_r2_error=maxerr,transport_refit_max_error=tmax,sealed_files_verified=seals,source_only_decisions_reproduced=12,policy_refit_max_error=policy_refit_error,native=native_records,verification_seconds=time.perf_counter()-started)
 OUT.mkdir(parents=True,exist_ok=True);(OUT/('native-verification.json' if a.native else 'offline-verification.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
