"""P60: shared unlabeled observation transport, then fresh frozen-program validation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,time,gc,resource
import numpy as np
from native import NativeField
from capacity_engine import capacity_targets

HERE=Path(__file__).resolve().parent;LAB=HERE.parent
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def r2(pred,Y):return 1-np.sum((pred-Y)**2,0)/np.maximum(np.sum((Y-Y.mean(0))**2,0),1e-30)
def normalized(X,h):return (X-h['feature_mean'])/h['feature_scale']

def main():
 p=json.loads((HERE/'protocol.json').read_text());c=p['planned_config']['config'];parent=LAB/'results'/c['parent'];ps=json.loads((parent/'summary.json').read_text());pc=ps['config']
 out=LAB/'results/p60-shared-observer-transport-001';out.mkdir(exist_ok=False);(out/'helpers').mkdir();(out/'models').mkdir()
 shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(HERE/'protocol.json',out/'protocol.json')
 for n in ['native.py','capacity_engine.py','stimulus.py']:shutil.copy2(HERE/n,out/'helpers'/n)
 dump(out/'manifest.json',dict(plan_id=p['id'],config=c,created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),helper_sha256={n:sha(HERE/n) for n in ['native.py','capacity_engine.py','stimulus.py']},parent_summary_sha256=sha(parent/'summary.json')))
 results=[];models=[];began=time.perf_counter()
 with (out/'runs.jsonl').open('w') as log:
  def emit(v):log.write(json.dumps(v,allow_nan=False)+'\n');log.flush()
  try:
   for label in ['0.6B','1.7B']:
    ts=time.perf_counter();d=parent/'models'/label;dest=out/'models'/label;dest.mkdir();meta=json.loads((d/'systems.json').read_text());cases=[load(d/f'case-{i}.npz') for i in range(6)];returns=load(d/'returns.npz')['returns']
    start=pc['evaluation_washout'];N=pc['evaluation_training'];past=load(d/'assessment.npz');selected=[];heads=[];base_heads=[];transports=[];source=[];selected_returns=[];a=time.perf_counter()
    for i,rows in enumerate(meta):
     chosen=[]
     for kind in c['families']:
      chosen.append(next(m for m in rows if m['kind']==kind and (kind in ['baseline','baseline_clone'] or m['fraction']==c['primary_action_fraction'])))
     selected.append(chosen);selected_returns.append(returns[i,[m['slot'] for m in chosen]])
     bh=load(d/f'readout-{i}-0.npz');base_heads.append(bh);Y=normalized(past['features'][i,0,start:start+N].astype(float),bh)
     hs=[];trs=[]
     for j,m in enumerate(chosen):
      h=load(d/f"readout-{i}-{m['slot']}.npz");hs.append(h);X=normalized(past['features'][i,m['slot'],start:start+N].astype(float),h)
      ridge=c['transport_ridge']*N;T=np.linalg.solve(X.T@X+ridge*np.eye(40),X.T@Y+ridge*np.eye(40));trs.append(T)
      source.append(dict(case=i,kind=m['kind'],parent_slot=m['slot'],source_relative_mse=float(np.sum((X@T-Y)**2)/max(np.sum(Y**2),1e-30)),transport_identity_distance=float(np.linalg.norm(T-np.eye(40)))))
      np.savez_compressed(dest/f'interface-{i}-{j}.npz',transport=T,new_feature_mean=h['feature_mean'],new_feature_scale=h['feature_scale'],old_feature_mean=bh['feature_mean'],old_feature_scale=bh['feature_scale'])
     heads.append(hs);transports.append(trs)
    Cs=np.stack(selected_returns);transports=np.array(transports);source_seconds=time.perf_counter()-a
    dump(dest/'systems.json',selected);dump(dest/'source-fit.json',source);dump(dest/'sealed.json',dict(files={f.name:sha(f) for f in dest.glob('interface-*.npz')},validation_exposed=False))
    receipt=json.loads((d/'checkpoint.json').read_text());native=NativeField(receipt['local_path'],emit,2,c['native_batch_size']);assert native.before[0]==next(m['parameter_sha256'] for m in ps['models'] if m['model']==label)
    K,S,m,n=Cs.shape;frames=np.stack([s['frame'] for s in cases]);U=np.stack([s['U'] for s in cases]);base=np.stack([s['base'] for s in cases]);R=np.stack([s['observable'] for s in cases]);eps=np.array([rows[0]['epsilon'] for rows in meta]);Z=np.zeros((K,S,8));valid=np.ones((K,S),bool)
    drive=np.random.default_rng(c['assessment_seed']).uniform(-1,1,c['washout']+c['test_steps']);features=np.empty((K,S,len(drive),40),np.float32);check_times=set(np.random.default_rng(c['verification_seed']).choice(len(drive),c['verification_steps'],replace=False));checks=[]
    native.stage='fresh-observer-validation';a=time.perf_counter()
    for t,u in enumerate(drive):
     prior=Z.copy();X=frames[:,None]+eps[:,None,None]*(np.einsum('csm,cmn->csn',Z,U[:,:8])+pc['input_gain']*u*U[:,8,None,:]);flat=X.reshape(-1,n)
     Y=native.run(np.concatenate([flat,np.repeat(flat[-1:],2,axis=0)]))[:K*S].reshape(K,S,n).astype(float);dy=Y-base[:,None];Z=np.einsum('csmn,csn->csm',Cs,dy);valid&=np.isfinite(Z).all(2)&(np.max(abs(Z),2)<=pc['state_limit']);Z[~valid]=0
     features[:,:,t,:8]=Z;features[:,:,t,8:]=np.einsum('crn,csn->csr',R,dy);features[~valid,t]=0
     if t in check_times:checks.append(dict(time=t,prior=prior,input=X.copy(),output=Y.astype(np.float32),state=Z.copy()))
     if (t+1)%512==0:print(json.dumps(dict(model=label,validation_steps=t+1,total=len(drive))),flush=True)
    validation_seconds=time.perf_counter()-a
    np.savez_compressed(dest/'validation.npz',drive=drive,features=features,valid=valid)
    np.savez_compressed(dest/'native-checks.npz',times=np.array([x['time'] for x in checks]),prior=np.array([x['prior'] for x in checks]),inputs=np.array([x['input'] for x in checks]),outputs=np.array([x['output'] for x in checks]),states=np.array([x['state'] for x in checks]))
    truth,desc=capacity_targets(drive,c['washout'],c['test_steps']);multi=np.array([q['degree']>=2 and q['distinct_lags']>=2 for q in desc])
    for i,rows in enumerate(selected):
     assert np.array_equal(features[i,0],features[i,4]),('baseline_clone',label,i)
     bh=base_heads[i];basepred=normalized(features[i,0,c['washout']:].astype(float),bh)@bh['readout']+bh['target_mean'];base_r2=r2(basepred,truth)
     priorbase=next(r for r in ps['results'] if r['model']==label and r['case']==i and r['kind']=='baseline');prior_r2=np.array([b['r2'] for b in priorbase['blocks']]);old=np.min(prior_r2,0)>=.5;stableold=old&(base_r2>=.5)
     for j,mdata in enumerate(rows):
      h=heads[i][j];X=features[i,j,c['washout']:].astype(float);ZX=normalized(X,h);rawold=normalized(X,bh)@bh['readout']+bh['target_mean'];translated=ZX@transports[i,j]@bh['readout']+bh['target_mean'];newpred=ZX@h['readout']+h['target_mean']
      raw_r2=r2(rawold,truth);tr_r2=r2(translated,truth);new_r2=r2(newpred,truth)
      if not valid[i,j]:raw_r2[:]=-1;tr_r2[:]=-1;new_r2[:]=-1
      pr=next(r for r in ps['results'] if r['model']==label and r['case']==i and r['slot']==mdata['slot']);pnew=np.array([b['r2'] for b in pr['blocks']]);strictnew=multi&(np.min(pnew,0)>=.5)&(new_r2>=.5)&(np.max(prior_r2,0)<.5)&(base_r2<.5)
      rec=dict(model=label,case=i,seed=mdata['seed'],kind=mdata['kind'],parent_slot=mdata['slot'],bounded=bool(valid[i,j]),
          prior_reliable_functions=int(old.sum()),fresh_stable_old_functions=int(stableold.sum()),baseline_stream_losses=np.flatnonzero(old&(base_r2<.5)).tolist(),
          raw_old_losses=np.flatnonzero(stableold&(raw_r2<.5)).tolist(),transported_old_losses=np.flatnonzero(stableold&(tr_r2<.5)).tolist(),
          raw_old_retained=int(sum(stableold&(raw_r2>=.5))),transported_old_retained=int(sum(stableold&(tr_r2>=.5))),
          reliably_new_nonlinear_targets=np.flatnonzero(strictnew).tolist(),
          capability_nonlinear_capacity=float(np.maximum(new_r2[multi],0).sum()),baseline_nonlinear_capacity=float(np.maximum(base_r2[multi],0).sum()),
          score_vectors=dict(baseline=base_r2.tolist(),raw_old=raw_r2.tolist(),transported_old=tr_r2.tolist(),capability=new_r2.tolist()))
      results.append(rec);emit(dict(event='program_validation',**rec))
     print(json.dumps({k:v for k,v in results[-4].items() if k!='score_vectors'}),flush=True)
    assert native.digest()==native.before
    rec=dict(model=label,weights_unchanged=True,parameter_sha256=native.before[0],source_fit_seconds=source_seconds,native_work=native.counts,native_seconds=native.seconds,validation_seconds=validation_seconds,wall_seconds=time.perf_counter()-ts,transport_shape=[40,40],transport_matrix_bytes=12800,source_normalization_values_per_interface=160,source_normalization_bytes_per_interface=1280,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    models.append(rec);dump(dest/'model-summary.json',rec);dump(out/'progress.json',dict(completed_models=len(models)));del native,past,features,checks,cases,Cs;gc.collect()
    import mlx.core as mx;mx.clear_cache()
   dump(out/'summary.json',dict(config=c,models=models,results=results,wall_seconds=time.perf_counter()-began));print(json.dumps(dict(complete=True,conditions=len(results))),flush=True)
  except Exception as e:dump(out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise

if __name__=='__main__':main()
