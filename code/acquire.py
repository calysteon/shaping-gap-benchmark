"""P46 central study: acquire E/K -> seal -> classify compositions -> execute."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import importlib.metadata
import json
import resource
import shutil
import time
import numpy as np
from substrate import array_sha
from native import Native,collect,analytic_reference,state,work,difference,PROJECTION_WEIGHTS
from interfaces import QuadraticInterface
from planner import Planner


def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def gaussian(seed,width,rms):
    x=np.random.default_rng(seed).normal(size=width).astype(np.float32)
    return x*(rms/np.sqrt(np.mean(x*x)))


def validation(model,data):
    predicted=[];gradients=[]
    if isinstance(model,QuadraticInterface):
        for z in data['z']:
            y,g=model.evaluate(z);predicted.append(y);gradients.append(g.transpose(2,0,1))
        yp=np.array(predicted);gp=np.array(gradients)
        return dict(amplitude_rmse=float(np.sqrt(np.mean((yp[:,:,0]-data['y'][:,:,0])**2))),
            effect_rmse=float(np.sqrt(np.mean((yp[:,:,1:]-data['y'][:,:,1:])**2))),
            effect_derivative_rmse=float(np.sqrt(np.mean((gp[:,:,:,1:]-data['g'][:,:,:,1:])**2))),
            effect_derivative_rms=float(np.sqrt(np.mean(data['g'][:,:,:,1:]**2))))
    # Reference values/effects are independently checked even though its known
    # formulas are supplied; no reference is presumed exact from its name.
    for z in data['z']:
        amp,q0=model.values(z);d=len(amp[0]);za=np.repeat(z[None],d,axis=0)
        za[np.arange(d),np.arange(d)]+=amp[0];_,qa=model.values(za)
        predicted.append(np.column_stack([amp[0],qa-q0]))
    yp=np.array(predicted)
    return dict(amplitude_rmse=float(np.sqrt(np.mean((yp[:,:,0]-data['y'][:,:,0])**2))),
        effect_rmse=float(np.sqrt(np.mean((yp[:,:,1:]-data['y'][:,:,1:])**2))))


def main():
    here=Path(__file__).resolve().parent;root=here.parent
    parser=argparse.ArgumentParser(description='Acquire source interfaces using the paper protocol.')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=root/'runs/source')
    args=parser.parse_args()
    plan=json.loads((root/'protocols/source.json').read_text());c=plan['planned_config']['config']
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    (out/'helpers').mkdir();(out/'scopes').mkdir();(out/'cases').mkdir()
    shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(root/'protocols/source.json',out/'protocol.json')
    helpers=['substrate.py','geometry_backend.py','interfaces.py','native.py','planner.py']
    for n in helpers:shutil.copy2(here/n,out/'helpers'/n)
    dump(out/'manifest.json',dict(plan_id=plan['id'],config=c,created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(Path(__file__)),helper_sha256={n:sha(here/n) for n in helpers}))
    raw=(out/'runs.jsonl').open('x');native=None
    def emit(row):
        if 'stage' not in row:row['stage']=native.stage if native is not None else 'setup'
        raw.write(json.dumps(row,allow_nan=False)+'\n');raw.flush()
    receipt=json.loads((root/'data/source/checkpoint.json').read_text());receipt['local_path']=str(args.checkpoint.resolve());dump(out/'checkpoint.json',receipt)
    started=time.perf_counter()
    try:
        native=Native(receipt['local_path'],emit);assert native.before[0]==c['parameter_sha256']
        native.stage='physical-inventory';before=dict(native.counts);ts=time.perf_counter()
        units=[tuple(v) for v in c['source_units']]
        D=np.column_stack([native.acquire_direction(l,j) for l,j in units]).astype(np.float64)
        assert np.linalg.matrix_rank(D)==len(units)
        P=np.linalg.solve(D.T@D,D.T)
        np.savez_compressed(out/'physical-coordinates.npz',D=D,P=P)
        common=dict(native_work=difference(before,native.counts),lpe=work(difference(before,native.counts)),seconds=time.perf_counter()-ts)
        scopes=[];models=[];source_costs=[];scores=[]
        for index,sc in enumerate(c['scopes']):
            folder=out/'scopes'/str(index);folder.mkdir()
            native.stage=f'scope{index}-anchor';before=dict(native.counts);ts=time.perf_counter()
            anchor=gaussian(sc['anchor_seed'],native.width,sc['anchor_rms'])
            q=native.consumer(anchor,sc['consumer_layer'],list(range(native.hidden)))
            pool=np.flatnonzero((abs(q)>=c['coordinate_min_value'])&(abs(q)<=c['coordinate_max_value']))
            indices=np.random.default_rng(sc['readout_seed']).choice(pool,c['readout_size'],replace=False).tolist()
            scope=dict(anchor=anchor,readout=dict(layer=sc['consumer_layer'],indices=indices),definition=sc)
            np.savez_compressed(folder/'domain.npz',anchor=anchor)
            dump(folder/'domain.json',dict(readout=scope['readout'],definition=sc))
            domain_cost=dict(native_work=difference(before,native.counts),seconds=time.perf_counter()-ts)
            rng=np.random.default_rng(sc['source_seed'])
            # Values are selected before outcomes. Cost-matched sample count is
            # determined only by the declared native-query ledger.
            all_z=rng.uniform(-c['domain_radius'],c['domain_radius'],size=(c['max_endpoint_source_contexts'],len(units)))
            ts=time.perf_counter()
            training,tw=collect(native,scope,D,P,units,all_z[:c['source_contexts']],True,emit,f's{index}-EK')
            train_time=time.perf_counter()-ts;np.savez_compressed(folder/'EK-source.npz',**training)
            # Determine exact endpoint-only native cost using its first context;
            # retain that observation and continue rather than wasting the query.
            ts=time.perf_counter()
            first,fw=collect(native,scope,D,P,units,all_z[:1],False,emit,f's{index}-E-first')
            first_time=time.perf_counter()-ts;endpoint_unit=work(fw)
            n_endpoint=work(tw)//endpoint_unit
            assert c['source_contexts']<=n_endpoint<=c['max_endpoint_source_contexts']
            ts=time.perf_counter()
            rest,rw=collect(native,scope,D,P,units,all_z[1:n_endpoint],False,emit,f's{index}-E-rest')
            endpoint_time=first_time+time.perf_counter()-ts
            endpoint={k:np.concatenate([first[k],rest[k]]) for k in first}
            np.savez_compressed(folder/'E-source-budget.npz',**endpoint)
            assert np.array_equal(endpoint['y'][:c['source_contexts']],training['y'])
            ew={k:fw.get(k,0)+rw.get(k,0) for k in set(fw)|set(rw)}
            assert work(ew)<=work(tw)
            trained={};fits={}
            for label,data,derivative in [('differential',training,True),('value_same',training,False),('value_budget',endpoint,False)]:
                ts=time.perf_counter();m=QuadraticInterface.fit(data,c['domain_radius'],derivative,c['ridge'],c['gradient_weight'])
                fits[label]=time.perf_counter()-ts;m.save(folder/f'{label}.npz');trained[label]=m
            native.stage=f's{index}-analytic-calibration';before=dict(native.counts);ts=time.perf_counter()
            reference=analytic_reference(native,scope,D,units);reference.save(folder/'analytic.npz');trained['analytic']=reference
            analytic_cost=dict(native_work=difference(before,native.counts),seconds=time.perf_counter()-ts)
            # Validation is diagnostic, not prefix/ridge/model selection.
            zv=np.random.default_rng(sc['validation_seed']).uniform(-c['domain_radius'],c['domain_radius'],size=(c['validation_contexts'],len(units)))
            ts=time.perf_counter();valid,vw=collect(native,scope,D,P,units,zv,True,emit,f's{index}-validation')
            val_time=time.perf_counter()-ts;np.savez_compressed(folder/'validation.npz',**valid)
            diagnostic={name:validation(model,valid) for name,model in trained.items()};dump(folder/'validation.json',diagnostic)
            cost=dict(domain=domain_cost,EK_source=dict(native_work=tw,lpe=work(tw),seconds=train_time),
                endpoint_source=dict(native_work=ew,lpe=work(ew),contexts=n_endpoint,seconds=endpoint_time,per_context_work=fw),
                fitting_seconds=fits,analytic=analytic_cost,validation=dict(native_work=vw,seconds=val_time))
            dump(folder/'costs.json',cost)
            scopes.append(scope);models.append(trained);source_costs.append(cost);scores.append(diagnostic)
            print(json.dumps(dict(scope=index,sealed_source=True,endpoint_contexts=n_endpoint,validation=diagnostic)),flush=True)
        sealed=dict(model_hashes={str(p.relative_to(out)):sha(p) for p in sorted((out/'scopes').glob('*/*.npz'))},
            later_conditions=c['later_conditions'],later_outcomes_exposed=False,
            validation_used_for_selection=False,projection_equivalent_weights=PROJECTION_WEIGHTS)
        dump(out/'sealed.json',sealed);emit(dict(kind='interfaces_sealed',**sealed))
        model_payloads={name:sum(m[name].payload() for m in models)+D.nbytes+P.nbytes+sum(s['anchor'].nbytes for s in scopes)
            for name in ['differential','value_same','value_budget','analytic']}
        assert max(model_payloads.values())<=c['controller_payload_cap_bytes']
        assert native.parameter_digest()==native.before
        dump(out/'summary.json',dict(acquisition_only=True,source_costs=source_costs,
            common_acquisition=common,validation=scores,sealed=sealed,
            model_payloads=model_payloads,config=c,weights_unchanged=True,
            parameter_sha256=native.before[0],stored_parameter_bytes=native.before[1],
            native_work_total=native.counts,wall_seconds=time.perf_counter()-started,
            versions={k:importlib.metadata.version(k) for k in ['numpy','mlx','mlx-lm']}))
        print(json.dumps(dict(acquisition_complete=True,model_payloads=model_payloads)),flush=True)
    except Exception as error:
        dump(out/'failure.json',dict(type=type(error).__name__,message=str(error)));raise
    finally:
        raw.close()
        if native is not None:native.close()

if __name__=='__main__':main()
