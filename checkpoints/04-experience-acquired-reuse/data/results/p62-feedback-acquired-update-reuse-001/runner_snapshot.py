"""P62: acquire context-specific execution/interface choices, then fresh reuse."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,hashlib,shutil,gc,resource
import numpy as np
from native import NativeField
from engine import execute,save_phase,counts,load,hp,rr
from policy import apply_interface
from capacity_engine import capacity_targets
HERE=Path(__file__).resolve().parent;LAB=HERE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 plan=json.loads((HERE/'protocol.json').read_text());c=plan['planned_config']['config'];parent=LAB/'results'/c['parent'];ps=json.loads((parent/'summary.json').read_text());out=LAB/'results/p62-feedback-acquired-update-reuse-001';out.mkdir(exist_ok=False);(out/'helpers').mkdir();(out/'models').mkdir();shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(HERE/'protocol.json',out/'protocol.json')
 for name in ['native.py','engine.py','augmentation.py','policy.py','capacity_engine.py','stimulus.py']:shutil.copy2(HERE/name,out/'helpers'/name)
 dump(out/'manifest.json',dict(plan_id=plan['id'],config=c,created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),helper_sha256={p.name:sha(p) for p in (out/'helpers').glob('*.py')},feedback_parent_summary_sha256=sha(parent/'summary.json')))
 a=time.perf_counter();selections=[]
 for label in c['models']:
  decisions=[]
  for case in range(6):
   rows=[r for r in ps['results'] if r['model']==label and r['case']==case];base=[next(r for r in rows if r['block']==b and r['slot']==0) for b in range(2)];old=np.min([r['r2'] for r in base],0)>=.5;bc=np.array([r['nonlinear_temporal_capacity'] for r in base]);candidates=[]
   for slot in range(1,5):
    rr0=[next(r for r in rows if r['block']==b and r['slot']==slot) for b in range(2)];cap=np.array([r['nonlinear_temporal_capacity'] for r in rr0]);relative=(cap-bc)/np.maximum(bc,.25);kept=old&(np.min([r['transported_old_r2'] for r in rr0],0)>=.5);eligible=bool(old.sum()>0 and all(r['bounded'] for r in base+rr0) and min(relative)>=c['quality_acceptance'] and kept.sum()==old.sum());candidates.append(dict(slot=slot,fraction=rr0[0]['fraction'],relative_delta=relative.tolist(),prior_old=int(old.sum()),retained=int(kept.sum()),feedback_admissible=eligible))
   valid=[r for r in candidates if r['feedback_admissible']];best=max(valid,key=lambda r:(min(r['relative_delta']),-abs(r['fraction']),-r['slot'])) if valid else dict(slot=0,fraction=0.);decisions.append(dict(model=label,case=case,seed=base[0]['seed'],selected_slot=best['slot'],fraction=best['fraction'],prior_old_ids=np.flatnonzero(old).tolist(),candidate_feedback=candidates))
  selections+=decisions
 choice_seconds=time.perf_counter()-a;dump(out/'selections.json',selections);dump(out/'selections-sealed.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files={'selections.json':sha(out/'selections.json')},parent_summary_sha256=sha(parent/'summary.json'),fresh_validation_started=False,selection_seconds=choice_seconds));print(json.dumps(dict(stage='feedback_choices_sealed',choices=[(r['model'],r['seed'],r['fraction']) for r in selections])),flush=True)
 models=[];results=[];started=time.perf_counter()
 with (out/'runs.jsonl').open('w') as log:
  def emit(v):log.write(json.dumps(v,allow_nan=False)+'\n');log.flush()
  try:
   for label in c['models']:
    ts=time.perf_counter();d=out/'models'/label;d.mkdir();pd=parent/'models'/label;receipt=json.loads((pd/'checkpoint.json').read_text());dump(d/'checkpoint.json',receipt);native=NativeField(receipt['local_path'],emit,2,c['native_batch_size']);assert native.before[0]==next(m['parameter_sha256'] for m in ps['models'] if m['model']==label);Cs=load(pd/'returns.npz')['returns'];cases=[load(pd/f'case-{i}.npz') for i in range(6)];meta=json.loads((pd/'systems.json').read_text())
    for i,s in enumerate(cases):s['epsilon']=meta[i][0]['epsilon']
    faces={(i,j):load(pd/f'interface-{i}-{j}.npz') for i in range(6) for j in range(7)};heads={(i,j):load(pd/f'readout-{i}-{j}.npz') for i in range(6) for j in range(7)};inherited=[pd/'returns.npz']+list(pd.glob('case-*.npz'))+list(pd.glob('interface-*.npz'))+list(pd.glob('readout-*.npz'));dump(d/'inherited-sealed.json',dict(files={p.name:sha(p) for p in inherited},source='P61 fixed source-fitted interfaces and programs; no refitting',fresh_validation_started=False));desc=json.loads((pd/'targets.json').read_text());multi=np.array([q['degree']>=2 and q['distinct_lags']>=2 for q in desc]);work=[];seconds=[];mr=[]
    for block,seed in enumerate(c['assessment_seeds']):
     drive=np.random.default_rng(seed).uniform(-1,1,c['assessment_washout']+c['assessment_steps']);before=dict(native.counts);a=time.perf_counter();features,valid,failures,checks=execute(native,cases,Cs,drive,c,emit,f'feedback-reuse-{label}-{block}');work.append(counts(native,before));seconds.append(time.perf_counter()-a);save_phase(d,f'assessment-{block}',drive,features,valid,failures,checks);truth,_=capacity_targets(drive,c['assessment_washout'],c['assessment_steps'])
     for i in range(6):
      for j,m in enumerate(meta[i]):
       X=features[i,j,c['assessment_washout']:].astype(float);h=heads[i,j];bh=heads[i,0];new=rr(hp(X,h),truth);direct=rr(hp(X,bh),truth);transport=rr(apply_interface(X,faces[i,j])@bh['readout']+bh['target_mean'],truth)
       if not valid[i,j]:new[:]=-1;direct[:]=-1;transport[:]=-1
       r=dict(model=label,case=i,seed=m['seed'],slot=j,block=block,kind=m['kind'],fraction=m['fraction'],bounded=bool(valid[i,j]),nonlinear_temporal_capacity=float(np.maximum(new[multi],0).sum()),r2=new.tolist(),direct_old_r2=direct.tolist(),transported_old_r2=transport.tolist());results.append(r);mr.append(r);emit(dict(event='fresh_reuse_score',**r))
     print(json.dumps(dict(model=label,stage='reuse_block_complete',block=block)),flush=True);del features,checks;gc.collect()
    assert native.digest()==native.before;rec=dict(model=label,parameter_sha256=native.before[0],weights_unchanged=True,native_work=native.counts,native_seconds=native.seconds,assessment_work=work,assessment_seconds=seconds,wall_seconds=time.perf_counter()-ts,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mlx_peak_bytes=native.mx.get_peak_memory(),choices=[r for r in selections if r['model']==label]);models.append(rec);dump(d/'scores.json',mr);dump(d/'model-summary.json',rec);del native,Cs,cases,faces,heads;gc.collect()
    import mlx.core as mx;mx.clear_cache()
   dump(out/'summary.json',dict(config=c,models=models,results=results,selection_seconds=choice_seconds,wall_seconds=time.perf_counter()-started));print(json.dumps(dict(complete=True,score_vectors=len(results))),flush=True)
  except Exception as e:dump(out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise
if __name__=='__main__':main()
