"""P54: full-model nonlinear computational capacity after unlabeled stimulation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import resource
import shutil
import time
import gc
import numpy as np
from native import NativeField
from capacity_engine import build_candidates,source_selection,selected_systems,run_systems,assess

HERE=Path(__file__).resolve().parent;LAB=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    plan=json.loads((HERE/'protocol-capacity.json').read_text());c=plan['planned_config']['config']
    out=LAB/'results/p55-neural-stimulus-capacity-002';out.mkdir(exist_ok=False);(out/'helpers').mkdir();(out/'models').mkdir()
    shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(HERE/'protocol-capacity.json',out/'protocol.json')
    helpers=['native.py','stimulus.py','capacity_engine.py']
    for n in helpers:shutil.copy2(HERE/n,out/'helpers'/n)
    dump(out/'manifest.json',dict(plan_id=plan['id'],config=c,created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),helper_sha256={n:sha(HERE/n) for n in helpers}))
    raw=(out/'runs.jsonl').open('x');began=time.perf_counter();results=[];model_records=[]
    def emit(v):raw.write(json.dumps(v,allow_nan=False)+'\n');raw.flush()
    try:
        for spec in c['models']:
            model_started=time.perf_counter();directory=out/'models'/spec['label'];directory.mkdir()
            receipt=json.loads((LAB/spec['receipt']).read_text());assert receipt['revision']==spec['revision'];dump(directory/'checkpoint.json',receipt)
            native=NativeField(receipt['local_path'],emit,c['tokens'],c['native_batch_size']);systems=[];source_records=[]
            for seed in c['seeds']:
                prior=dict(native.counts);ts=time.perf_counter();native.stage=f"{spec['label']}-source-{seed}"
                setup=build_candidates(native,seed,c);emit(dict(kind='geometry_acquired',model=spec['label'],seed=seed))
                for i,g in enumerate(setup['geometries']):
                    np.savez_compressed(directory/f'geometry-{seed}-{i}.npz',**{k:v for k,v in g.items() if isinstance(v,np.ndarray)})
                records,selection,trace,bound=source_selection(native,setup,c,seed)
                np.savez_compressed(directory/f'source-{seed}.npz',trace=trace,frame=setup['frame'],actuator=setup['U'],token_ids=setup['token_ids'])
                dump(directory/f'source-{seed}.json',dict(candidates=records,selection=selection,controller_array_bound_bytes=bound))
                selected=selected_systems(setup,selection,c,seed)
                for local_index,system in enumerate(selected):
                    name=f'selected-{seed}-{local_index}.npz';np.savez_compressed(directory/name,**{k:v for k,v in system.items() if isinstance(v,np.ndarray)})
                    meta={k:v for k,v in system.items() if not isinstance(v,np.ndarray)};meta['array_file']=name;systems.append(meta)
                source_record=dict(seed=seed,seconds=time.perf_counter()-ts,native_work={k:native.counts[k]-prior[k] for k in prior},selection=selection,
                    unstable_candidates=sum(not r['valid'] for r in records),candidates=len(records),controller_array_bound_bytes=bound)
                source_records.append(source_record);emit(dict(kind='source_selection',model=spec['label'],**source_record))
                print(json.dumps(dict(model=spec['label'],seed=seed,selected={k:None if j is None else records[j] for k,j in selection.items()},systems=len(selected))),flush=True)
                del setup,trace,selected;gc.collect()
            # All source decisions freeze before assessment labels are constructed.
            for meta in systems:
                with np.load(directory/meta['array_file']) as data:
                    meta.update({k:data[k].copy() for k in data.files})
            for i,s in enumerate(systems):
                np.savez_compressed(directory/f'system-{i}.npz',**{k:v for k,v in s.items() if isinstance(v,np.ndarray)})
            descriptors=[{k:v for k,v in s.items() if not isinstance(v,np.ndarray)} for s in systems]
            dump(directory/'systems.json',descriptors)
            dump(directory/'sealed.json',dict(files={p.name:sha(p) for p in directory.glob('system-*.npz')},assessment_labels_exposed=False))
            rng=np.random.default_rng(94001);T=c['evaluation_washout']+c['evaluation_training']+c['evaluation_test'];drive=rng.uniform(-1,1,T)
            native.stage=f"{spec['label']}-assessment";before=dict(native.counts);ts=time.perf_counter()
            states,features,valid,checks,bound=run_systems(native,systems,c,drive,emit)
            np.savez_compressed(directory/'assessment.npz',drive=drive,states=states,features=features,valid=valid)
            np.savez_compressed(directory/'native-checks.npz',times=np.array([r['time'] for r in checks]),prior=np.array([r['prior'] for r in checks]),
                inputs=np.array([r['input'] for r in checks]),outputs=np.array([r['native_output'] for r in checks]),states=np.array([r['state'] for r in checks]),valid=np.array([r['valid'] for r in checks]))
            assess_seconds=time.perf_counter()-ts;assessment_work={k:native.counts[k]-before[k] for k in before}
            for i,s in enumerate(systems):
                score,readout,targets=assess(features[i],drive,c)
                if not valid[i]:score.update(total_capacity=0.,nonlinear_temporal_capacity=0.,reliable_functions=0,reliable_nonlinear_temporal_functions=0,capacity_by_degree={str(d):0. for d in range(1,5)})
                np.savez_compressed(directory/f'readout-{i}.npz',**readout)
                row=dict(model=spec['label'],system=i,**descriptors[i],stable_on_assessment=bool(valid[i]),**score)
                results.append(row);emit(dict(event='capacity_result',**row))
                print(json.dumps({k:v for k,v in row.items() if k!='r2'}),flush=True)
            dump(directory/'targets.json',targets)
            assert native.digest()==native.before
            rec=dict(model=spec['label'],parameter_sha256=native.before[0],stored_parameter_bytes=native.before[1],width=native.width,layers=native.layers,
                source_records=source_records,native_work=native.counts,native_seconds=native.seconds,assessment_seconds=assess_seconds,
                assessment_native_work=assessment_work,assessment_controller_array_bound_bytes=bound,wall_seconds=time.perf_counter()-model_started,
                weights_unchanged=True,mlx_peak_bytes=native.mx.get_peak_memory(),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            model_records.append(rec);dump(directory/'model-summary.json',rec)
            del native,systems,states,features,checks;gc.collect()
            import mlx.core as mx
            mx.clear_cache()
            dump(out/'progress.json',dict(completed_models=len(model_records),models=len(c['models'])))
        dump(out/'summary.json',dict(config=c,results=results,models=model_records,wall_seconds=time.perf_counter()-began))
        print(json.dumps(dict(complete=True,models=len(model_records),systems=len(results))),flush=True)
    except Exception as e:dump(out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise
    finally:raw.close()

if __name__=='__main__':main()
