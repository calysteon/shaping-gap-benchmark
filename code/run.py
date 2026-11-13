"""Fresh central validation after a symmetric stop-aware planner correction.

All source model coefficients, domain definitions and acquisition costs are
inherited unchanged from P46. No refitting or later-outcome source update.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import resource
import shutil
import time
import numpy as np
from substrate import array_sha
from native import Native,state,work,difference
from interfaces import QuadraticInterface,AnalyticReference
from planner import Planner


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    here=Path(__file__).resolve().parent;root=here.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=root/'data/source')
    parser.add_argument('--output',type=Path,default=root/'runs/benchmark')
    args=parser.parse_args();parent=args.source
    plan=json.loads((root/'protocols/benchmark.json').read_text());c=plan['planned_config']['config']
    if not json.loads((parent/'summary.json').read_text()).get('acquisition_only'):
        assert sha(parent/'summary.json')==c['source_summary_sha256']
    for name,h in c['inherited_model_sha256'].items():assert sha(parent/name)==h
    inherited=json.loads((parent/'summary.json').read_text())
    out=args.output;out.mkdir(parents=True,exist_ok=False);(out/'helpers').mkdir();(out/'cases').mkdir()
    shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(root/'protocols/benchmark.json',out/'protocol.json')
    helpers=['substrate.py','geometry_backend.py','interfaces.py','native.py','planner.py']
    for n in helpers:shutil.copy2(here/n,out/'helpers'/n)
    dump(out/'manifest.json',dict(plan_id=plan['id'],config=c,created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(Path(__file__)),helper_sha256={n:sha(here/n) for n in helpers}))
    physical=np.load(parent/'physical-coordinates.npz');D,P=physical['D'],physical['P']
    units=[tuple(v) for v in c['source_units']];scopes=[];models=[]
    for i in range(len(c['scopes'])):
        folder=parent/'scopes'/str(i);domain=json.loads((folder/'domain.json').read_text())
        scopes.append(dict(anchor=np.load(folder/'domain.npz')['anchor'],readout=domain['readout']))
        pack={}
        for name in ['differential','value_same','value_budget']:
            a=np.load(folder/f'{name}.npz');pack[name]=QuadraticInterface(a['c'],a['L'],a['H'],a['shape'])
        a=np.load(folder/'analytic.npz');pack['analytic']=AnalyticReference(**{n:a[n] for n in a.files});models.append(pack)
    dump(out/'sealed.json',dict(inherited_model_sha256=c['inherited_model_sha256'],models_refit=False,
        parent_run='p46-keystone-001',later_conditions=c['later_conditions']))
    receipt=json.loads((parent/'checkpoint.json').read_text());receipt['local_path']=str(args.checkpoint.resolve());dump(out/'checkpoint.json',receipt)
    raw=(out/'runs.jsonl').open('x');native=None;started=time.perf_counter()
    def emit(row):
        if 'stage' not in row:row['stage']=native.stage if native is not None else 'setup'
        raw.write(json.dumps(row,allow_nan=False)+'\n');raw.flush()
    try:
        native=Native(receipt['local_path'],emit);assert native.before[0]==c['parameter_sha256']
        results=[]
        for task,condition in enumerate(c['later_conditions']):
            i=condition['scope'];scope=scopes[i]
            z=np.random.default_rng(condition['seed']).uniform(-c['initial_radius'],c['initial_radius'],size=len(units))
            x=state(scope['anchor'],D,z);native.stage=f't{task}-condition';before=dict(native.counts);ts=time.perf_counter()
            q=native.consumer(x,scope['readout']['layer'],scope['readout']['indices'])
            mask=np.random.default_rng(condition['seed']+900000).choice(c['readout_size'],c['contract_size'],replace=False)
            base=q[mask];n=c['target_count']
            lo=np.r_[base[:n]+condition['target_increment'],base[n:]-c['protected_tolerance']].astype(np.float32)
            hi=np.r_[np.full(n,np.inf),base[n:]+c['protected_tolerance']].astype(np.float32)
            contract=dict(mask=mask,lower=lo,upper=hi)
            generation=dict(native_work=difference(before,native.counts),seconds=time.perf_counter()-ts)
            case_start=dict(kind='case_start',task=task,condition=condition,mask=mask.tolist(),
                lower=lo.tolist(),upper=[None if not np.isfinite(v) else float(v) for v in hi],root_sha256=array_sha(x))
            np.savez_compressed(out/'cases'/f't{task}.npz',root=x,q=q,z=z)
            methods=c['methods'][task%len(c['methods']):]+c['methods'][:task%len(c['methods'])]
            for method in methods:
                native.stage=f't{task}-{method}';emit(dict(**case_start,method=method))
                label='value_same' if method=='value_same_pair' else 'value_budget' if method=='value_budget_pair' else 'analytic' if method=='analytic_pair' else 'differential'
                before=dict(native.counts);ts=time.perf_counter()
                planner=Planner(native,scope,D,P,units,models[i][label],method,c,emit)
                r=planner.search(x,q,contract);search_seconds=time.perf_counter()-ts;search_work=difference(before,native.counts)
                assert r['query_lpe']<=c['query_budget_lpe']
                final=r.pop('final_x');final_q=r.pop('final_q')
                native.stage=f't{task}-{method}-deployment';before=dict(native.counts);ts=time.perf_counter()
                replay=x.copy()
                for b in r['graph']:
                    a=native.activity(replay,b['layer']);replay+=native.write(b['layer'],b['unit'],float(a[b['unit']]))
                qr=native.consumer(replay,scope['readout']['layer'],scope['readout']['indices'])
                assert np.array_equal(replay,final) and np.array_equal(qr,final_q)
                r.update(task=task,method=method,condition=condition,source_model=None if method=='native_local' else label,
                    search_native_work=search_work,search_seconds=search_seconds,generation=generation,
                    deploy_native_work=difference(before,native.counts),deploy_seconds=time.perf_counter()-ts,
                    deploy_exact=True,final_q=qr.tolist(),payload_bytes=(D.nbytes+P.nbytes+sum(v['anchor'].nbytes for v in scopes))
                    if method=='native_local' else inherited['model_payloads'][label])
                emit(dict(kind='case_result',**r));results.append(r)
            by={r['method']:r for r in results if r['task']==task}
            assert by['differential_pair']['graph']==by['differential_restored']['graph']
            assert by['differential_pair']['final_sha256']==by['differential_restored']['final_sha256']
            if (task+1)%6==0:print(json.dumps(dict(completed_conditions=task+1,
                solved={m:sum(r['metrics']['satisfied'] for r in results if r['method']==m) for m in c['methods']})),flush=True)
            dump(out/'progress.json',dict(completed_conditions=task+1,total_conditions=len(c['later_conditions'])))
        for name,h in c['inherited_model_sha256'].items():assert sha(parent/name)==h
        assert native.parameter_digest()==native.before
        totals={}
        for method in c['methods']:
            rs=[r for r in results if r['method']==method]
            totals[method]=dict(satisfied=sum(r['metrics']['satisfied'] for r in rs),conditions=len(rs),
                query_lpe=sum(r['query_lpe'] for r in rs),branches=sum(r['branches'] for r in rs),
                search_seconds=sum(r['search_seconds'] for r in rs),deploy_seconds=sum(r['deploy_seconds'] for r in rs),
                by_scope={str(i):dict(satisfied=sum(r['metrics']['satisfied'] for r in rs if r['condition']['scope']==i),
                    conditions=sum(r['condition']['scope']==i for r in rs)) for i in range(len(scopes))})
        dump(out/'summary.json',dict(plan_id=plan['id'],config=c,source_costs=inherited['source_costs'],
            common_acquisition=inherited['common_acquisition'],validation=inherited['validation'],
            model_payloads=inherited['model_payloads'],source_parent='p46-keystone-001',models_refit=False,
            results=results,totals=totals,restoration_exact_conditions=len(c['later_conditions']),weights_unchanged=True,
            parameter_sha256=native.before[0],stored_parameter_bytes=native.before[1],native_work_total=native.counts,
            wall_seconds=time.perf_counter()-started,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            mlx_peak_bytes=native.mx.get_peak_memory(),versions=inherited['versions'],limitations=plan['limitations']))
        print(json.dumps(dict(complete=True,totals=totals)),flush=True)
    except Exception as error:
        dump(out/'failure.json',dict(type=type(error).__name__,message=str(error)));raise
    finally:
        raw.close()
        if native is not None:native.close()


if __name__=='__main__':main()
