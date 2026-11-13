"""P64: acquired bounded recruitment, protected programs and cumulative reuse."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,resource,shutil
import numpy as np
from recruit import restore_runtime,Archive,save_state,payload,load
from cycle import execute,source_gradient,build_candidates,fit_candidates,evaluate,select,outcome,canonical,targets,mask,library_prediction,quality
from legacy_engine import head,rr,fit_head
HERE=Path(__file__).resolve().parent;LAB=HERE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def save_checks(p,rows):
 if rows:np.savez_compressed(p,**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
def save_library(folder,lib):
 folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'head.npz',**lib['head']);np.savez_compressed(folder/'input-map.npz',A=lib['A'],b=lib['b']);dump(folder/'program-ids.json',lib['ids'])
def state_hash(state):
 h=hashlib.sha256()
 for a in [state['A'],state['b']]+[np.asarray(v) for e in state['edges'] for k,v in sorted(e.items())]:h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()
def setup_libraries(parent,slot):
 h0=load(parent/'head-0.npz');h1=load(parent/f'head-{slot}.npz');m=load(parent/f'interface-{slot}.npz');A=m['transport']*m['old_feature_scale'][None,:]/m['new_feature_scale'][:,None];b=m['old_feature_mean']-m['new_feature_mean']@A
 choice=json.loads((parent/'selection.json').read_text());blocks=json.loads((parent/'feedback-scores.json').read_text())+json.loads((parent/'fresh-scores.json').read_text());ids=np.flatnonzero(np.min([x[slot]['r2'] for x in blocks],0)>=.5).tolist()
 return [dict(head=h0,A=A,b=b,ids=choice['old_program_ids'],name='pre_checkpoint05_programs'),dict(head=h1,A=np.eye(24),b=np.zeros(24),ids=ids,name='checkpoint05_accepted_programs')],h1

def main():
 plan=json.loads((HERE/'protocol.json').read_text());c=plan['config'];out=LAB/'results'/plan['run_id'];out.mkdir(exist_ok=False);(out/'helpers').mkdir();(out/'contexts').mkdir();started=time.perf_counter();all_results=[]
 for p in HERE.glob('*.py'):shutil.copy2(p,out/'helpers'/p.name)
 shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(HERE/'protocol.json',out/'protocol.json');dump(out/'manifest.json',dict(plan_id=plan['id'],created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),helper_sha256={p.name:sha(p) for p in (out/'helpers').glob('*.py')},config=c,pool_manifest_sha256=sha(HERE/'pools/manifest.json')))
 parentroot=LAB/'results/p63-component-acquisition-reuse-001/contexts';parentconfig=json.loads((LAB/'component-cycle-study/protocol.json').read_text())['config'];phase='setup'
 with (out/'runs.jsonl').open('x') as log:
  def emit(v):log.write(json.dumps(v,allow_nan=False)+'\n');log.flush()
  try:
   for model in c['models']:
    for seed in c['seeds']:
     parent=parentroot/f'{model}-{seed}-integrated';pc={**parentconfig,'anchor_rms':parentconfig['anchor_rms_by_seed'][str(seed)]};rootrt,g,C,root_slot=restore_runtime(parent,pc)
     for arm in c['arms']:
      key=f'{model}-{seed}-{arm}';folder=out/'contexts'/key;folder.mkdir();rt,g,C,_=restore_runtime(parent,pc);arc=Archive(HERE/'pools'/f'{model}-{seed}',rt,C,c);state=dict(edges=[],A=np.eye(24),b=np.zeros(24));anchor=dict(edges=[],A=np.eye(24),b=np.zeros(24));libs,current_head=setup_libraries(parent,root_slot);history=[(state,current_head)];rounds=[];round_count=c['deep_rounds'] if model==c['deep_model'] and seed==c['deep_seed'] else 1;total_begin=time.perf_counter();peak_payload=0
      for i,lib in enumerate(libs):save_library(folder/f'library-{i}',lib)
      dump(folder/'parent.json',dict(context=parent.name,selected_slot=root_slot,source_runtime_sha256=sha(parent/'runtime.npz'),source_return_sha256=sha(parent/'returns.npz'),initial_compatibility='Canonical space is the accepted Checkpoint05 runtime. Original readouts unchanged.'))
      for step in range(round_count):
       phase=f'{key}:round{step}';d=folder/f'round-{step}';d.mkdir();begin=time.perf_counter();before=arc.counters();runtime_before=dict(rt.counts);counts={};timing={};multi=mask(json.loads((parent/'targets.json').read_text()));seedbase=c['seed_base']+step*c['seed_stride'];drive=np.random.default_rng(seedbase+1).uniform(-1,1,c['burn']+c['training_steps'])
       a=time.perf_counter();screen,checks=execute(rt,arc,g,C,[state],drive,c,'screening_parent',4);timing['screening_parent']=time.perf_counter()-a;np.savez_compressed(d/'screening-parent.npz',**screen);save_checks(d/'checks-screening.npz',checks);s,gradient=source_gradient(screen,state,current_head,c,multi)
       pool='resident' if arm=='resident' else int(arm.split('-')[-1]);a=time.perf_counter();ids,scores=arc.select(s,gradient,state['edges'],pool,arm.startswith('random'),seedbase+seed+9);timing['screening_ports']=time.perf_counter()-a;dump(d/'screening-scores.json',scores)
       a=time.perf_counter();candidates,meta,cal=build_candidates(arc,state,ids,s,screen,c,seedbase+seed,d);timing['local_profiles']=time.perf_counter()-a;dump(d/'candidate-metadata.json',meta);dump(d/'calibration.json',cal);states=candidates+[anchor];reference=len(candidates)
       a=time.perf_counter();source,checks=execute(rt,arc,g,C,states,drive,c,'candidate_source',6);heads,desc=fit_candidates(source,candidates,state,c);timing['candidate_source_and_fit']=time.perf_counter()-a;np.savez_compressed(d/'source.npz',**source);save_checks(d/'checks-source.npz',checks);delta=float(np.max(abs(screen['features'][0]-source['features'][0])));dump(d/'baseline-repeat-fidelity.json',dict(max_absolute_error=delta,rtol=2e-5,atol=2e-5));assert np.allclose(screen['features'][0],source['features'][0],rtol=2e-5,atol=2e-5)
       sealed={}
       for j,st in enumerate(candidates):
        save_state(d/f'candidate-{j}',st);np.savez_compressed(d/f'head-{j}.npz',**heads[j]);sealed[str(j)]=state_hash(st)
       dump(d/'construction-seal.json',dict(utc=datetime.now(timezone.utc).isoformat(),states=sealed,heads={str(j):sha(d/f'head-{j}.npz') for j in range(len(heads))},prior_library_hashes={str(k):sha(folder/f'library-{k}/head.npz') for k in range(len(libs))},fresh_exposed=False))
       feedback=[]
       for b in range(2):
        drv=np.random.default_rng(seedbase+10+b).uniform(-1,1,c['burn']+c['assessment_steps']);a=time.perf_counter();tr,checks=execute(rt,arc,g,C,states,drv,c,f'feedback-{b}',4);timing[f'feedback-{b}']=time.perf_counter()-a;feedback.append(evaluate(tr,states,heads,libs,c,multi,reference));np.savez_compressed(d/f'feedback-{b}.npz',**tr);save_checks(d/f'checks-feedback-{b}.npz',checks)
       dump(d/'feedback-scores.json',feedback);choice=select(feedback,libs,c);dump(d/'selection.json',choice);dump(d/'selection-seal.json',dict(utc=datetime.now(timezone.utc).isoformat(),sha256=sha(d/'selection.json'),construction_sha256=sha(d/'construction-seal.json'),fresh_validation_started=False));emit(dict(event='selected',context=key,round=step,slot=choice['selected_slot'],proposed=ids));print(json.dumps(dict(context=key,round=step,stage='selected',slot=choice['selected_slot'],ports=ids)),flush=True)
       fresh=[]
       for b in range(2):
        drv=np.random.default_rng(seedbase+20+b).uniform(-1,1,c['burn']+c['assessment_steps']);a=time.perf_counter();tr,checks=execute(rt,arc,g,C,states,drv,c,f'fresh-{b}',4);timing[f'fresh-{b}']=time.perf_counter()-a;fresh.append(evaluate(tr,states,heads,libs,c,multi,reference));np.savez_compressed(d/f'fresh-{b}.npz',**tr);save_checks(d/f'checks-fresh-{b}.npz',checks)
       dump(d/'fresh-scores.json',fresh);j=choice['selected_slot'];result=outcome(j,fresh,libs,c);accepted=result['strict_fresh_gain'];state_before=state_hash(state)
       if accepted:
        state=candidates[j];current_head=heads[j];old=np.min([q[j]['r2'] for q in feedback+fresh],0)>=.5;lib=dict(head=current_head,A=np.eye(24),b=np.zeros(24),ids=np.flatnonzero(old).tolist(),name=f'round{step}_acquired');libs.append(lib);save_library(folder/f'library-{len(libs)-1}',lib);history.append((state,current_head))
       for j0,st in enumerate(candidates):assert state_hash(st)==sealed[str(j0)]
       for k in range(len(libs)-(1 if accepted else 0)):assert sha(folder/f'library-{k}/head.npz')==json.loads((d/'construction-seal.json').read_text())['prior_library_hashes'][str(k)]
       bytes_=sum(a.nbytes for lib in libs for a in list(lib['head'].values())+[lib['A'],lib['b']])+payload(state)+sum(v.nbytes for v in load(parent/'runtime.npz').values());peak_payload=max(peak_payload,bytes_);assert bytes_<=c['retained_payload_cap'];assert len(state['edges'])<=c['retained_recruits']
       record=dict(context=key,model=model,seed=seed,arm=arm,round=step,**result,accepted=accepted,selected_metadata=meta[j],retained_port_ids=[e['id'] for e in state['edges']],retained_payload_bytes=bytes_,shared_map_values=state['A'].size+state['b'].size,protected_program_instances=sum(len(lib['ids']) for lib in libs),library_count=len(libs),parent_state_sha256=state_before,retained_state_sha256=state_hash(state),timing=timing,wall_seconds=time.perf_counter()-begin,acquisition_work={k:arc.counters()[k]-before[k] if k not in ['maximum_screen_workspace_bytes'] else arc.counters()[k] for k in before},base_runtime_work={k:rt.counts.get(k,0)-runtime_before.get(k,0) for k in rt.counts})
       dump(d/'result.json',record);emit(dict(event='round_result',**record));rounds.append(record);print(json.dumps({k:record[k] for k in ['context','round','accepted','relative_gains','retained_port_ids','retained_payload_bytes']}),flush=True)
      # Final source fits are common exposure only; controller remains sealed.
      d=folder/'terminal';d.mkdir();save_state(d/'final-state',state);np.savez_compressed(d/'current-head.npz',**current_head);dump(d/'controller-seal.json',dict(utc=datetime.now(timezone.utc).isoformat(),state=state_hash(state),retained_ports=[e['id'] for e in state['edges']],all_round_outcomes_are_prior_experience=True,fresh_terminal_outcomes_exposed=False));terminalstates=[anchor,state]+[v[0] for v in history[1:]];drv=np.random.default_rng(c['terminal_source_seed']).uniform(-1,1,c['burn']+c['terminal_training']);a=time.perf_counter();tr,checks=execute(rt,arc,g,C,terminalstates,drv,c,'terminal_source',4);Y,_=targets(drv,c['burn'],c['terminal_training']);terminalheads=[fit_head(canonical(tr['features'][k,c['burn']:].astype(float),st),Y,c['readout_ridge']) for k,st in enumerate(terminalstates)];np.savez_compressed(d/'source.npz',**tr);save_checks(d/'checks-source.npz',checks)
      for j0,h in enumerate(terminalheads):np.savez_compressed(d/f'head-{j0}.npz',**h)
      dump(d/'heads-seal.json',dict(utc=datetime.now(timezone.utc).isoformat(),heads={str(j0):sha(d/f'head-{j0}.npz') for j0 in range(len(terminalheads))},state=state_hash(state)))
      terminal=[]
      for b,seed0 in enumerate(c['terminal_seeds']):
       drv=np.random.default_rng(seed0).uniform(-1,1,c['burn']+c['terminal_steps']);z,checks=execute(rt,arc,g,C,terminalstates,drv,c,f'terminal-{b}',6);scores=evaluate(z,terminalstates,terminalheads,libs,c,multi,0);terminal.append(scores);np.savez_compressed(d/f'fresh-{b}.npz',**z);save_checks(d/f'checks-fresh-{b}.npz',checks)
      dump(d/'scores.json',terminal);gains=[(b[1]['quality']-b[0]['quality'])/max(b[0]['quality'],.25) for b in terminal];checks0=[contract0 for contract0 in []];oldOK=True
      from cycle import contract
      contracts=[contract(b[1],b[0],libs,c) for b in terminal];oldOK=all(q['programs_ok'] and q['quality_ok'] for q in contracts)
      result=dict(context=key,model=model,seed=seed,arm=arm,rounds=rounds,accepted_acquisitions=sum(r['accepted'] for r in rounds),terminal_relative_gains=gains,terminal_quality_changes=[b[1]['quality']-b[0]['quality'] for b in terminal],terminal_quality_root=[b[0]['quality'] for b in terminal],terminal_quality_final=[b[1]['quality'] for b in terminal],terminal_contracts=contracts,terminal_strict_gain=bool(min(gains)>=c['quality_acceptance'] and oldOK and all(b[1]['bounded'] and b[0]['bounded'] for b in terminal)),cumulative_repeated_gain=bool(sum(r['accepted'] for r in rounds)>=2 and min(gains)>=c['quality_acceptance'] and oldOK),final_retained_port_ids=[e['id'] for e in state['edges']],fixed_map_values=state['A'].size+state['b'].size,peak_retained_payload_bytes=peak_payload,source_and_online_work=arc.counters(),base_runtime_work=rt.counts,wall_seconds=time.perf_counter()-total_begin,terminal_seconds=time.perf_counter()-a)
      dump(folder/'result.json',result);all_results.append(result);emit(dict(event='terminal_result',**result));dump(out/'progress.json',dict(completed=len(all_results),expected=6*len(c['arms']),last=key));print(json.dumps({k:result[k] for k in ['context','accepted_acquisitions','terminal_relative_gains','terminal_strict_gain','cumulative_repeated_gain','wall_seconds']}),flush=True)
   dump(out/'summary.json',dict(config=c,results=all_results,wall_seconds=time.perf_counter()-started,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
  except BaseException as e:dump(out/'failure.json',dict(phase=phase,type=type(e).__name__,message=str(e),elapsed_seconds=time.perf_counter()-started,base_work=rt.counts if 'rt' in locals() else {},recruitment_work=arc.counters() if 'arc' in locals() else {}));raise
if __name__=='__main__':main()
