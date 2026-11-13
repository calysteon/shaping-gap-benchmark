"""P61: sealed outcome-trained selection on new acquired recurrent runtimes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,time,gc,resource
import numpy as np
from native import NativeField
from capacity_engine import build_candidates,source_selection,capacity_targets
from augmentation import acquire,construct
from policy import source_features,fit_interface,predict,choose,apply_interface
HERE=Path(__file__).resolve().parent;LAB=HERE.parent

def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def counts(native,before):return {k:native.counts[k]-before[k] for k in before}
def rr(pred,y):return 1-np.sum((pred-y)**2,0)/np.maximum(np.sum((y-y.mean(0))**2,0),1e-30)
def head(X,Y,ridge):
 mu=X.mean(0);sd=np.maximum(X.std(0),1e-8);Z=(X-mu)/sd;ym=Y.mean(0);W=np.linalg.solve(Z.T@Z+ridge*len(X)*np.eye(40),Z.T@(Y-ym));return dict(feature_mean=mu,feature_scale=sd,readout=W,target_mean=ym)
def hp(X,h):return ((X-h['feature_mean'])/h['feature_scale'])@h['readout']+h['target_mean']
def execute(native,cases,Cs,drive,c,emit,stage):
 K,S,m,n=Cs.shape;frames=np.stack([s['frame'] for s in cases]);U=np.stack([s['U'] for s in cases]);base=np.stack([s['base'] for s in cases]);R=np.stack([s['observable'] for s in cases]);eps=np.array([s['epsilon'] for s in cases]);Z=np.zeros((K,S,m));valid=np.ones((K,S),bool);failures=np.full((K,S),-1,int);features=np.zeros((K,S,len(drive),40),np.float32);check_times=set(np.random.default_rng(c['verification_seed']).choice(len(drive),c['verification_steps'],replace=False));checks=[]
 native.stage=stage;pad=(-K*S)%c['native_batch_size']
 for t,u in enumerate(drive):
  prior=Z.copy();X=frames[:,None]+eps[:,None,None]*(np.einsum('csm,cmn->csn',Z,U[:,:8])+c['input_gain']*u*U[:,8,None,:]);flat=X.reshape(-1,n);Y=native.run(np.concatenate([flat,np.repeat(flat[-1:],pad,axis=0)]))[:K*S].reshape(K,S,n).astype(float);dy=Y-base[:,None];Z=np.einsum('csmn,csn->csm',Cs,dy);good=np.isfinite(Z).all(2)&(np.max(abs(Z),2)<=c['state_limit']);failures[valid&~good]=t;valid&=good;Z[~valid]=0;features[:,:,t,:8]=Z;features[:,:,t,8:]=np.einsum('crn,csn->csr',R,dy);features[~valid,t]=0
  if t in check_times:checks.append(dict(time=t,inputs=X.copy(),outputs=Y.astype(np.float32),prior=prior,state=Z.copy()))
  if (t+1)%512==0:
   row=dict(event='trajectory_progress',stage=stage,completed=t+1,total=len(drive),bounded=int(valid.sum()));emit(row);print(json.dumps(row),flush=True)
 assert all(np.array_equal(features[i,0],features[i,6]) for i in range(K))
 return features,valid,failures,checks

def save_phase(d,name,drive,features,valid,failures,checks):
 np.savez_compressed(d/f'{name}.npz',drive=drive,features=features,valid=valid,first_failure=failures)
 np.savez_compressed(d/f'{name}-native-checks.npz',times=np.array([r['time'] for r in checks]),inputs=np.array([r['inputs'] for r in checks]),outputs=np.array([r['outputs'] for r in checks]),prior=np.array([r['prior'] for r in checks]),states=np.array([r['state'] for r in checks]))

def main():
 plan=json.loads((HERE/'protocol.json').read_text());c=plan['planned_config']['config'];out=LAB/'results/p61-learned-update-compatibility-001';assert (out/'policy-sealed.json').exists();assert not (out/'manifest.json').exists();ps=json.loads((out/'policy-sealed.json').read_text())
 for name,h in ps['files'].items():assert sha(out/name)==h
 assert sha(HERE/'policy.py')==ps['source_sha256']['policy.py'];policy=load(out/'policy.npz');shutil.copy2(__file__,out/'runner_snapshot.py')
 dump(out/'manifest.json',dict(plan_id=plan['id'],config=c,created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),helper_sha256={p.name:sha(p) for p in (out/'helpers').glob('*.py')},sealed_policy_sha256=sha(out/'policy.npz'),policy_seal_sha256=sha(out/'policy-sealed.json')))
 old=LAB/'results'/c['baseline_source_config'];oldcfg=json.loads((old/'summary.json').read_text())['config'];allresults=[];models=[];started=time.perf_counter()
 with (out/'runs.jsonl').open('w') as log:
  def emit(v):log.write(json.dumps(v,allow_nan=False)+'\n');log.flush()
  try:
   for label in c['models']:
    ts=time.perf_counter();d=out/'models'/label;d.mkdir();receipt=json.loads((old/'models'/label/'checkpoint.json').read_text());dump(d/'checkpoint.json',receipt);native=NativeField(receipt['local_path'],emit,2,c['native_batch_size']);expected=json.loads((old/'models'/label/'model-summary.json').read_text())['parameter_sha256'];assert native.before[0]==expected
    before=dict(native.counts);a=time.perf_counter();cases=[];baseline_rows=[]
    for seed in c['fresh_seeds']:
     native.stage=f'fresh-chart-{seed}';setup=build_candidates(native,seed,oldcfg);records,selection,trace,bound=source_selection(native,setup,oldcfg,seed);idx=selection['unaligned'];admissible=idx is not None
     if idx is None:idx=max((r for r in records if r['kind']=='unaligned'),key=lambda r:r['intrinsic_effective_rank'])['index']
     candidate=setup['candidates'][idx];g=setup['geometries'][candidate['geometry']];s=dict(C=candidate['C'].copy(),frame=setup['frame'].copy(),U=setup['U'].copy(),base=g['base'].copy(),observable=g['observable'].copy(),J=g['J'].copy(),q=g['q'].copy(),v=g['v'].copy(),epsilon=candidate['epsilon'],seed=seed,parent='P61-fresh-source',source_admissible=admissible);cases.append(s)
     np.savez_compressed(d/f'chart-{seed}-source.npz',trace=trace,token_ids=setup['token_ids']);dump(d/f'chart-{seed}-source.json',dict(seed=seed,selected_index=idx,source_admissible=admissible,records=records,explicit_array_reservation=bound));baseline_rows.append(dict(seed=seed,selected_index=idx,source_admissible=admissible,selected_record=records[idx]));emit(dict(event='fresh_chart_acquired',model=label,seed=seed,source_admissible=admissible));print(json.dumps(dict(model=label,stage='chart_acquired',seed=seed,epsilon=s['epsilon'],source_admissible=admissible)),flush=True);del setup,trace;gc.collect()
    chart_work=counts(native,before);chart_seconds=time.perf_counter()-a
    ac=dict(source_seed=c['calibration_seed'],source_steps=c['calibration_steps'],source_burn=c['calibration_burn'],random_seed=c['random_seed'],input_gain=c['input_gain'],state_limit=c['state_limit'],sign_probe_steps=c['sign_probe_steps'],signed_action_fractions=c['signed_action_fractions'],conditions_per_case=14,padded_conditions_per_case=16)
    before=dict(native.counts);a=time.perf_counter();cal,caldata=acquire(native,cases,ac);full,meta=construct(cases,cal,ac);slots=[0,1,2,3,4,11,13];Cs=full[:,slots].copy();del full;meta=[[dict(row,slot=j,parent_slot=row['slot']) for j,row in enumerate([rows[s] for s in slots])] for rows in meta];cal_work=counts(native,before);cal_seconds=time.perf_counter()-a
    for i,s in enumerate(cases):np.savez_compressed(d/f'case-{i}.npz',**{k:v for k,v in s.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(d/'returns.npz',returns=Cs);np.savez_compressed(d/'calibration.npz',**caldata);dump(d/'calibration.json',cal);dump(d/'systems.json',meta);dump(d/'source-charts.json',baseline_rows);dump(d/'couplings-sealed.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files={p.name:sha(p) for p in d.glob('*.npz')},source_trajectory_started=False,assessment_labels_exposed=False))
    drive=np.random.default_rng(c['source_seed']).uniform(-1,1,c['source_steps']);before=dict(native.counts);a=time.perf_counter();features,valid,failures,checks=execute(native,cases,Cs,drive,c,emit,'unlabeled-update-source');source_work=counts(native,before);source_seconds=time.perf_counter()-a;save_phase(d,'source',drive,features,valid,failures,checks);faces=[];decisions=[];feature_rows=[];a=time.perf_counter();start=c['source_washout']
    for i,s in enumerate(cases):
     B=features[i,0,start:].astype(float);fs=[];interfaces=[]
     for j,m in enumerate(meta[i]):
      X=features[i,j,start:].astype(float)
      if j in range(1,5):f,face=source_features(B,X,drive,start,m['fraction'],c['feature_seed']);fs.append(f);feature_rows.append(dict(case=i,slot=j,features=f.tolist()))
      else:face=fit_interface(B,X,c['transport_ridge'])
      interfaces.append(face);np.savez_compressed(d/f'interface-{i}-{j}.npz',**face)
     pred=predict(np.array(fs),policy);choice=choose(pred,c)
     if not (s['source_admissible'] and valid[i,0] and valid[i,choice]):choice=0
     decision=dict(case=i,seed=s['seed'],selected_slot=int(choice),fraction=meta[i][choice]['fraction'],predictions=pred.tolist(),source_bounded=valid[i].tolist());decisions.append(decision);faces.append(interfaces)
    fit_seconds=time.perf_counter()-a;dump(d/'decisions.json',decisions);dump(d/'source-policy-features.json',feature_rows);dump(d/'decisions-sealed.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),policy_sha256=sha(out/'policy.npz'),files={p.name:sha(p) for p in list(d.glob('interface-*.npz'))+[d/'decisions.json',d/'source-policy-features.json']},new_chart_function_labels_exposed=False,fresh_assessment_started=False));print(json.dumps(dict(model=label,stage='decisions_sealed',choices=[r['fraction'] for r in decisions],fit_seconds=fit_seconds)),flush=True)
    # Only now generate functional targets and fit the assessment programs.
    traintruth,desc=capacity_targets(drive,start,len(drive)-start);heads=[];a=time.perf_counter()
    for i in range(len(cases)):
     hh=[]
     for j in range(len(meta[i])):
      h=head(features[i,j,start:].astype(float),traintruth,c['readout_ridge']);hh.append(h);np.savez_compressed(d/f'readout-{i}-{j}.npz',**h)
     heads.append(hh)
    head_seconds=time.perf_counter()-a;dump(d/'targets.json',desc);dump(d/'programs-sealed.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files={p.name:sha(p) for p in d.glob('readout-*.npz')},fresh_assessment_started=False));multi=np.array([q['degree']>=2 and q['distinct_lags']>=2 for q in desc]);blocks=[];assessment_work=[];assessment_seconds=[]
    del features,checks;gc.collect()
    for block,seed in enumerate(c['assessment_seeds']):
     drive=np.random.default_rng(seed).uniform(-1,1,c['assessment_washout']+c['assessment_steps']);before=dict(native.counts);a=time.perf_counter();features,valid,failures,checks=execute(native,cases,Cs,drive,c,emit,f'fresh-assessment-{block}');assessment_seconds.append(time.perf_counter()-a);assessment_work.append(counts(native,before));save_phase(d,f'assessment-{block}',drive,features,valid,failures,checks);truth,_=capacity_targets(drive,c['assessment_washout'],c['assessment_steps']);br=[]
     for i in range(len(cases)):
      scores=[]
      for j,m in enumerate(meta[i]):
       X=features[i,j,c['assessment_washout']:].astype(float);h=heads[i][j];bh=heads[i][0];new=rr(hp(X,h),truth);direct=rr(hp(X,bh),truth);transport=rr(apply_interface(X,faces[i][j])@bh['readout']+bh['target_mean'],truth)
       if not valid[i,j]:new[:]=-1;direct[:]=-1;transport[:]=-1
       scores.append(dict(model=label,case=i,seed=cases[i]['seed'],block=block,slot=j,kind=m['kind'],fraction=m['fraction'],bounded=bool(valid[i,j]),nonlinear_temporal_capacity=float(np.maximum(new[multi],0).sum()),r2=new.tolist(),direct_old_r2=direct.tolist(),transported_old_r2=transport.tolist()));emit(dict(event='fresh_score',**scores[-1]))
      br.append(scores)
     blocks.append(br);print(json.dumps(dict(model=label,stage='fresh_block_scored',block=block)),flush=True);del features,checks;gc.collect()
    assert native.digest()==native.before
    rows=[r for block in blocks for chart in block for r in chart];allresults.extend(rows);dump(d/'scores.json',rows);rec=dict(model=label,parameter_sha256=native.before[0],stored_parameter_bytes=native.before[1],weights_unchanged=True,chart_acquisition_work=chart_work,chart_acquisition_seconds=chart_seconds,calibration_work=cal_work,calibration_seconds=cal_seconds,source_work=source_work,source_seconds=source_seconds,source_policy_interface_fit_seconds=fit_seconds,readout_fit_seconds=head_seconds,assessment_work=assessment_work,assessment_seconds=assessment_seconds,native_work=native.counts,native_seconds=native.seconds,wall_seconds=time.perf_counter()-ts,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mlx_peak_bytes=native.mx.get_peak_memory(),return_shape=list(Cs.shape[2:]),retained_selected_interface_bytes=14080,decisions=decisions);models.append(rec);dump(d/'model-summary.json',rec);dump(out/'progress.json',dict(completed_models=len(models),models=2));del native,Cs,cases,faces,heads,blocks;gc.collect()
    import mlx.core as mx;mx.clear_cache()
   dump(out/'summary.json',dict(config=c,models=models,results=allresults,wall_seconds=time.perf_counter()-started));print(json.dumps(dict(complete=True,score_vectors=len(allresults))),flush=True)
  except Exception as e:dump(out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise
if __name__=='__main__':main()
