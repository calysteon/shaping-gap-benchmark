"""P48: source acquisition, exact forecast scaling and common native construction."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import resource
import argparse
import shutil
import time
import numpy as np
from native import Native,work,difference
from substrate import array_sha
from geometry_backend import Geometry
from factored import Bank,ScaledPlanner

HERE=Path(__file__).resolve().parent;LAB=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,default=LAB/'runs/scaling');parser.add_argument('--scopes',type=Path,default=LAB/'data/inherited');args=parser.parse_args();plan=json.loads((LAB/'protocols/scaling.json').read_text());c=plan['planned_config']['config']
    out=args.output;out.mkdir(parents=True,exist_ok=False);(out/'helpers').mkdir();(out/'cases').mkdir()
    shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(LAB/'protocols/scaling.json',out/'protocol.json')
    helpers=['substrate.py','geometry_backend.py','native.py','interfaces.py','base_planner.py','factored.py']
    for n in helpers:shutil.copy2(HERE/n,out/'helpers'/n)
    dump(out/'manifest.json',dict(plan_id=plan['id'],config=c,created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(Path(__file__)),helper_sha256={n:sha(HERE/n) for n in helpers}))
    parent=args.scopes;scopes=[]
    for i in range(3):
        d=json.loads((parent/'scopes'/str(i)/'domain.json').read_text())
        scopes.append(dict(anchor=np.load(parent/'scopes'/str(i)/'domain.npz')['anchor'],readout=d['readout']))
    receipt=json.loads((parent/'checkpoint.json').read_text());receipt['local_path']=str(args.checkpoint.resolve());dump(out/'checkpoint.json',receipt)
    stream=(out/'runs.jsonl').open('x');native=None;started=time.perf_counter()
    def emit(v):
        if 'stage' not in v:v['stage']=native.stage if native is not None else 'setup'
        stream.write(json.dumps(v,allow_nan=False)+'\n');stream.flush()
    try:
        native=Native(receipt['local_path'],emit);assert native.before[0]==c['parameter_sha256']
        units=[tuple(v) for v in c['source_units']];rng=np.random.default_rng(c['bank_seed'])
        for flat in rng.permutation(len(native.layers)*native.hidden):
            unit=(int(flat)//native.hidden,int(flat)%native.hidden)
            if unit not in units:units.append(unit)
            if len(units)==max(c['bank_sizes']):break
        dump(out/'units.json',units)
        x0=scopes[c['source_anchor_scope']]['anchor'];n=native.width;eps=float(native.config['rms_norm_eps'])
        radius=np.sqrt(float(x0.astype(np.float64)@x0)/n+eps);geometry=Geometry(native,{})
        native.stage='source-readouts';before=dict(native.counts);ts=time.perf_counter();Q=[]
        for scope in scopes:
            q,j=geometry.qj(x0,scope['readout']);Q.append(radius*j+np.outer(q,x0)/(n*radius))
        common_source=dict(native_work=difference(before,native.counts),seconds=time.perf_counter()-ts)
        D=[];G=[];U=[];costs=[];prefix={};physical_seconds=behavior_seconds=0.
        for index,(layer,unit) in enumerate(units):
            native.stage=f'source-component-{index}';before=dict(native.counts);ts=time.perf_counter()
            D.append(native.acquire_direction(layer,unit).astype(np.float64));pt=time.perf_counter()-ts
            physical_work=difference(before,native.counts);before=dict(native.counts);ts=time.perf_counter()
            for kind,rows in [('gate',G),('up',U)]:
                value,gradient=native.pre_row(x0,layer,unit,kind)
                rows.append(radius*gradient+value*x0/(n*radius))
            bt=time.perf_counter()-ts;behavior_work=difference(before,native.counts)
            physical_seconds+=pt;behavior_seconds+=bt
            costs.append(dict(unit=[layer,unit],physical_native_work=physical_work,physical_seconds=pt,behavior_native_work=behavior_work,behavior_seconds=bt))
            if index+1 in c['bank_sizes']:
                prefix[str(index+1)]=dict(physical_lpe=index+1,behavior_lpe=4*(index+1)+work(common_source['native_work']),
                    physical_seconds=physical_seconds,behavior_seconds=behavior_seconds+common_source['seconds'])
                print(json.dumps(dict(acquired_components=index+1)),flush=True)
        np.savez_compressed(out/'source.npz',D=np.column_stack(D),G=np.array(G),U=np.array(U),Q=np.array(Q),eps=eps,
                            anchors=np.array([v['anchor'] for v in scopes]))
        dump(out/'source-costs.json',dict(prefix=prefix,common_readouts=common_source,components=costs))
        source_hash=sha(out/'source.npz');dump(out/'sealed.json',dict(source_sha256=source_hash,units_sha256=sha(out/'units.json'),later_outcomes_exposed=False))
        del D,G,U,Q,geometry
        conditions=[]
        for demand in c['demands']:
            for scope_id in range(3):
                for repeat in range(c['conditions_per_domain_per_demand']):
                    task=len(conditions);conditions.append(dict(task=task,scope=scope_id,demand=demand,seed=c['root_seed_start']+task))
        validation=[];micro=[];results=[];preparation=[]
        for k in c['bank_sizes']:
            ts=time.perf_counter();bank=Bank(out/'source.npz',k);prep=time.perf_counter()-ts
            assert np.isfinite(bank.P).all();assert bank.payload()+8*1024**2<=c['controller_payload_cap_bytes']
            preparation.append(dict(bank_size=k,seconds=prep,physical_seconds=bank.physical_preparation_seconds,
                behavior_seconds=bank.behavior_preparation_seconds,payload_bytes=bank.payload(),basis_condition=float(np.linalg.cond(bank.gram))))
            # Diagnostics use fixed states before later contracts; no fitting or selection.
            if k==max(c['bank_sizes']):
                native.stage='source-validation';before=dict(native.counts);ts=time.perf_counter()
                for scope_id,scope in enumerate(scopes):
                    for j in range(c['validation_contexts_per_domain']):
                        z=np.random.default_rng(73001+100*scope_id+j).uniform(-.5,.5,size=22)
                        x=scope['anchor']+np.asarray(bank.D[:,:22]@z,dtype=np.float32)
                        a,qhat=bank.values(x,scope_id);q=native.consumer(x,scope['readout']['layer'],scope['readout']['indices']);activity={}
                        actual=[]
                        for layer,unit in units:
                            if layer not in activity:activity[layer]=native.activity(x,layer)
                            actual.append(float(activity[layer][unit]))
                        validation.append(dict(scope=scope_id,index=j,max_amplitude_error=float(np.max(abs(a-actual))),max_readout_error=float(np.max(abs(qhat-q)))))
                dump(out/'source-validation.json',dict(rows=validation,native_work=difference(before,native.counts),seconds=time.perf_counter()-ts))
            for scope_id in range(c['micro_contexts_per_bank']):
                scope=scopes[scope_id];seed=74001+scope_id
                z0=np.random.default_rng(seed).uniform(-.5,.5,size=22)
                x=scope['anchor']+np.asarray(bank.D[:,:22]@z0,dtype=np.float32);z=bank.P@(x-scope['anchor'])
                native.stage=f'k{k}-micro{scope_id}';q=native.consumer(x,scope['readout']['layer'],scope['readout']['indices'])
                mask=np.arange(c['contract_size']);base=q[mask];contract=dict(mask=mask,lower=np.r_[base[:3]+.075,base[3:]-.05],upper=np.r_[np.full(3,np.inf),base[3:]+.05])
                times={};ranks={}
                for name,direct,rows in [('direct',True,c['direct_tile_rows']),('factored',False,c['factored_tile_rows'])]:
                    bound=bank.payload()+bank.workspace_bound(rows,len(mask),direct)+8*1024**2
                    assert bound<=c['controller_payload_cap_bytes'],(k,name,bound)
                    ts=time.perf_counter();ranks[name]=bank.rank(x,q,scope_id,mask,z,contract,c,rows,direct)
                    times[name]=time.perf_counter()-ts
                diffs={field:0. for field in ['q1','q2','amplitude']}
                for first,second in zip(bank.tiles(x,q,scope_id,mask,c['direct_tile_rows'],True),bank.tiles(x,q,scope_id,mask,c['direct_tile_rows'])):
                    for field in ['q1','q2']:diffs[field]=max(diffs[field],float(np.max(abs(first[field]-second[field]))))
                    diffs['amplitude']=max(diffs['amplitude'],float(np.max(abs(first['successor_amplitude']-second['successor_amplitude']))))
                assert all(v<1e-7 for v in diffs.values()),(k,scope_id,diffs)
                direct_rank=ranks['direct'];factor_rank=ranks['factored']
                micro.append(dict(bank_size=k,scope=scope_id,seconds=times,max_abs_discrepancy=diffs,
                    first_six_choices_equal=direct_rank['order'][:6]==factor_rank['order'][:6],
                    direct_workspace_bound=bank.workspace_bound(c['direct_tile_rows'],len(mask),True),
                    factored_workspace_bound=bank.workspace_bound(c['factored_tile_rows'],len(mask))))
                emit(dict(kind='micro',**micro[-1]));del ranks
            for condition in conditions:
                task=condition['task'];i=condition['scope'];scope=scopes[i]
                z0=np.random.default_rng(condition['seed']).uniform(-c['initial_radius'],c['initial_radius'],size=22)
                x=scope['anchor']+np.asarray(bank.D[:,:22]@z0,dtype=np.float32)
                native.stage=f'k{k}-t{task}-condition';before=dict(native.counts);ts=time.perf_counter()
                q=native.consumer(x,scope['readout']['layer'],scope['readout']['indices'])
                mask=np.random.default_rng(condition['seed']+900000).choice(c['readout_size'],c['contract_size'],replace=False)
                base=q[mask];lo=np.r_[base[:3]+condition['demand'],base[3:]-.05].astype(np.float32)
                hi=np.r_[np.full(3,np.inf),base[3:]+.05].astype(np.float32);contract=dict(mask=mask,lower=lo,upper=hi)
                generation=dict(native_work=difference(before,native.counts),seconds=time.perf_counter()-ts)
                if k==c['bank_sizes'][0]:np.savez_compressed(out/'cases'/f't{task}.npz',root=x,q=q)
                else:assert np.array_equal(x,np.load(out/'cases'/f't{task}.npz')['root'])
                for method in c['methods'][task%2:]+c['methods'][:task%2]:
                    native.stage=f'k{k}-t{task}-{method}';before=dict(native.counts);ts=time.perf_counter()
                    emit(dict(kind='case_start',bank_size=k,condition=condition,method=method,mask=mask.tolist(),lower=lo.tolist(),upper=[None if not np.isfinite(v) else float(v) for v in hi],root_sha256=array_sha(x)))
                    planner=ScaledPlanner(native,scope,bank,units[:k],method,c,emit,i);result=planner.search(x,q,contract)
                    search_seconds=time.perf_counter()-ts;search_work=difference(before,native.counts)
                    final=result.pop('final_x');final_q=result.pop('final_q');native.stage+= '-deployment';before=dict(native.counts);ts=time.perf_counter()
                    replay=x.copy()
                    for action in result['graph']:
                        amp=native.activity(replay,action['layer']);replay+=native.write(action['layer'],action['unit'],float(amp[action['unit']]))
                    qr=native.consumer(replay,scope['readout']['layer'],scope['readout']['indices'])
                    assert np.array_equal(replay,final) and np.array_equal(qr,final_q)
                    result.update(bank_size=k,condition=condition,method=method,search_seconds=search_seconds,search_native_work=search_work,
                        deploy_seconds=time.perf_counter()-ts,deploy_native_work=difference(before,native.counts),generation=generation,
                        final_q=qr.tolist(),native_replay_exact=True)
                    emit(dict(kind='case_result',**result));results.append(result)
                if (task+1)%12==0:print(json.dumps(dict(bank_size=k,completed=task+1,solved={m:sum(r['metrics']['satisfied'] for r in results if r['bank_size']==k and r['method']==m) for m in c['methods']})),flush=True)
                dump(out/'progress.json',dict(bank_size=k,completed_conditions=task+1,conditions=len(conditions)))
            del bank
        assert sha(out/'source.npz')==source_hash and native.parameter_digest()==native.before
        summary=dict(config=c,results=results,micro=micro,source_prefix_costs=prefix,preparation=preparation,source_validation=validation,
            weights_unchanged=True,source_unchanged=True,parameter_sha256=native.before[0],stored_parameter_bytes=native.before[1],
            wall_seconds=time.perf_counter()-started,native_work_total=native.counts,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mlx_peak_bytes=native.mx.get_peak_memory())
        dump(out/'summary.json',summary);print(json.dumps(dict(complete=True,conditions=len(conditions),rows=len(results),micro_checks=len(micro))),flush=True)
    except Exception as error:
        dump(out/'failure.json',dict(type=type(error).__name__,message=str(error)));raise
    finally:
        stream.close()
        if native is not None:native.close()

if __name__=='__main__':main()
