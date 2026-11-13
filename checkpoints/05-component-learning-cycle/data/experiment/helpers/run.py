"""P63: response-guided component execution with acquired compatibility and reuse."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,resource,sys,shutil
import numpy as np
from component import ComponentRuntime,digest
from engine import baseline,proposals,fit,evaluate,select,outcome
HERE=Path(__file__).resolve().parent;LAB=HERE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p) as x:return {k:x[k].copy() for k in x.files}
def save_checks(p,checks):
 if checks:np.savez_compressed(p,**{k:np.stack([r[k] for r in checks]) for k in checks[0]})
def run():
 plan=json.loads((HERE/'protocol.json').read_text());c=plan['config'];out=LAB/'results'/plan['run_id'];out.mkdir(exist_ok=False);(out/'helpers').mkdir();(out/'contexts').mkdir()
 for p in HERE.glob('*.py'):shutil.copy2(p,out/'helpers'/p.name)
 shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(HERE/'protocol.json',out/'protocol.json')
 dump(out/'manifest.json',dict(plan_id=plan['id'],created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),helper_sha256={p.name:sha(p) for p in (out/'helpers').glob('*.py')},bank_manifest_sha256=sha(HERE/'banks/manifest.json'),config=c))
 manifests=json.loads((HERE/'banks/manifest.json').read_text());shutil.copy2(HERE/'banks/manifest.json',out/'bank-manifest.json')
 results=[];started=time.perf_counter();phase='start'
 with (out/'runs.jsonl').open('x') as log:
  def emit(v):log.write(json.dumps(v,allow_nan=False)+'\n');log.flush()
  try:
   for rec in manifests['records']:
    if rec['model'] not in c['models'] or rec['seed'] not in c['seeds']:continue
    bank=load(HERE/'banks'/rec['file']);assert sha(HERE/'banks'/rec['file'])==rec['sha256']
    for variant in c['variants']:
     key=rec['model']+'-'+str(rec['seed'])+'-'+variant;folder=out/'contexts'/key;folder.mkdir();begin=time.perf_counter();timing={}
     cc={**c,'anchor_rms':c['anchor_rms_by_seed'][str(rec['seed'])]};rt=ComponentRuntime(bank,rec['seed'],cc,variant=='random_weight');rt.mode='integrated' if variant=='random_weight' else variant
     phase=key+':profile';a=time.perf_counter();source,diagnostic,errors=rt.acquire();timing['profile']=time.perf_counter()-a;np.savez_compressed(folder/'profile-source.npz',**source);np.savez_compressed(folder/'profile-diagnostic.npz',**diagnostic);rt.save(folder/'runtime.npz');dump(folder/'profile-diagnostic.json',errors)
     phase=key+':geometry';a=time.perf_counter();g=rt.geometry();timing['geometry']=time.perf_counter()-a;np.savez_compressed(folder/'geometry.npz',**g)
     a=time.perf_counter();C,basemeta,basetrace=baseline(rt,g,cc);timing['baseline_acquisition']=time.perf_counter()-a;dump(folder/'baseline-acquisition.json',basemeta);np.savez_compressed(folder/'baseline-acquisition.npz',**basetrace)
     a=time.perf_counter();Cs,pm,pt=proposals(rt,g,C,cc);timing['return_acquisition']=time.perf_counter()-a;dump(folder/'proposals.json',pm);np.savez_compressed(folder/'return-calibration.npz',**pt);np.savez_compressed(folder/'returns.npz',returns=Cs)
     phase=key+':source';a=time.perf_counter();heads,maps,desc,source,checks=fit(rt,g,Cs,cc);timing['source_programs_and_compatibility']=time.perf_counter()-a;np.savez_compressed(folder/'source.npz',**source);save_checks(folder/'checks-source.npz',checks);dump(folder/'targets.json',desc)
     for i,(h,m) in enumerate(zip(heads,maps)):np.savez_compressed(folder/f'head-{i}.npz',**h);np.savez_compressed(folder/f'interface-{i}.npz',**m)
     files=[p for p in folder.glob('*.npz') if p.name.startswith(('head-','interface-','runtime','returns','geometry'))];sealed={p.name:sha(p) for p in files};dump(folder/'construction-seal.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files=sealed,feedback_exposed=False,validation_exposed=False,original_bank_sha256=rec['sha256'],variant=variant))
     feedback=[]
     for b,seed in enumerate(cc['feedback_seeds']):
      phase=key+f':feedback{b}';a=time.perf_counter();scores,tr,checks=evaluate(rt,g,Cs,heads,maps,cc,seed,f'feedback-{b}',desc);timing[f'feedback-{b}']=time.perf_counter()-a;feedback.append(scores);np.savez_compressed(folder/f'feedback-{b}.npz',**tr);save_checks(folder/f'checks-feedback-{b}.npz',checks)
     dump(folder/'feedback-scores.json',feedback);choice=select(feedback,cc);dump(folder/'selection.json',choice);dump(folder/'selection-seal.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=sha(folder/'selection.json'),construction_seal_sha256=sha(folder/'construction-seal.json'),fresh_validation_started=False));emit(dict(event='construction_selected',context=key,**choice));print(json.dumps(dict(context=key,stage='selection_sealed',selected=choice['selected_slot'],old=len(choice['old_program_ids']),profile_error=errors['value_relative_rmse'])),flush=True)
     fresh=[]
     for b,seed in enumerate(cc['fresh_seeds']):
      phase=key+f':fresh{b}';a=time.perf_counter();scores,tr,checks=evaluate(rt,g,Cs,heads,maps,cc,seed,f'fresh-{b}',desc);timing[f'fresh-{b}']=time.perf_counter()-a;fresh.append(scores);np.savez_compressed(folder/f'fresh-{b}.npz',**tr);save_checks(folder/f'checks-fresh-{b}.npz',checks)
     dump(folder/'fresh-scores.json',fresh)
     for p,h in sealed.items():assert sha(folder/p)==h
     assert digest(rt.bank)==rt.before
     result=dict(context=key,model=rec['model'],seed=rec['seed'],variant=variant,source_baseline_admissible=basemeta['source_admissible'],profile_diagnostic=errors,**outcome(choice,fresh,cc),timing=timing,wall_seconds=time.perf_counter()-begin,work=rt.counts,route_counts=rt.route_counts.tolist(),route_switches=rt.route_switches,pretrained_parameters_unchanged=True,runtime_payload_bytes=sum(a.nbytes for a in load(folder/'runtime.npz').values()),return_bytes=Cs[choice['selected_slot']].nbytes,interface_bytes=sum(a.nbytes for a in maps[choice['selected_slot']].values()),readout_library_bytes=sum(a.nbytes for a in heads[0].values()),bank_source_sha256=rec['sha256'])
     dump(folder/'result.json',result);results.append(result);emit(dict(event='fresh_result',**result));dump(out/'progress.json',dict(completed=len(results),expected=len(c['models'])*len(c['seeds'])*len(c['variants']),last=key));print(json.dumps({k:result[k] for k in ['context','selected_slot','relative_gains','old_programs','old_retained_both','strict_joint','wall_seconds']}),flush=True)
   dump(out/'summary.json',dict(plan_id=plan['id'],results=results,wall_seconds=time.perf_counter()-started,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,original_transformer_calls=0,attention_calls=0,tokenizer_calls=0))
  except BaseException as e:dump(out/'failure.json',dict(phase=phase,type=type(e).__name__,message=str(e),completed=len(results)));raise
if __name__=='__main__':run()
