"""Independent native replay for every P48 selected graph and sampled branches."""
from pathlib import Path
import hashlib
import argparse
from trace_io import open_trace
import json
import time
import re
import numpy as np
from substrate import Substrate,array_sha
from base_planner import metrics
from factored import Bank

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    lab=Path(__file__).resolve().parent.parent;parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--run',type=Path,default=lab/'data/scaling');parser.add_argument('--output',type=Path,default=lab/'runs/scaling-verification.json');args=parser.parse_args();out=args.run
    s=json.loads((out/'summary.json').read_text());c=s['config'];manifest=json.loads((out/'manifest.json').read_text())
    assert sha(out/'runner_snapshot.py')==manifest['source_sha256']
    for n,h in manifest['helper_sha256'].items():assert sha(out/'helpers'/n)==h
    sealed=json.loads((out/'sealed.json').read_text());assert sha(out/'source.npz')==sealed['source_sha256']
    units=[tuple(v) for v in json.loads((out/'units.json').read_text())]
    parent=lab/c['source_directory'];scopes=[]
    for i in range(3):
        p=parent/'scopes'/str(i);d=json.loads((p/'domain.json').read_text())
        scopes.append(dict(anchor=np.load(p/'domain.npz')['anchor'],readout=d['readout']))
    starts={};tree={};counts={};sample=[];rng=np.random.default_rng(76001);branches=0
    with open_trace(out) as f:
        for line in f:
            r=json.loads(line)
            if r['kind']=='primitive':counts[r['operation']]=counts.get(r['operation'],0)+r['vectors']
            elif r['kind']=='case_start':starts[r['stage']]=r
            elif r['kind']=='branch':
                branches+=1;tree[(r['stage'],r['node'])]=(r['parent'],r['action']['unit_index'])
                if len(sample)<512:sample.append(r)
                else:
                    j=int(rng.integers(branches))
                    if j<len(sample):sample[j]=r
    assert counts==s['native_work_total']
    native=Substrate(args.checkpoint,lambda r:None)
    started=time.perf_counter();replayed=0;checked=0
    try:
        assert native.before[0]==c['parameter_sha256']
        for k in c['bank_sizes']:
            bank=Bank(out/'source.npz',k)
            def apply(x,j):
                layer,unit=units[j];activity=native.activity(x,layer);amp=float(activity[unit])
                h=np.zeros(native.hidden,dtype=np.float32);h[unit]=activity[unit]
                w=native.down(layer,h,'verification_masked_down')
                return x+w,amp,w
            for result in [v for v in s['results'] if v['bank_size']==k]:
                condition=result['condition'];scope=scopes[condition['scope']];task=condition['task']
                stage=f'k{k}-t{task}-{result["method"]}';start=starts[stage]
                x=np.load(out/'cases'/f't{task}.npz')['root'].copy();assert array_sha(x)==start['root_sha256']
                for action in result['graph']:
                    assert units[action['unit_index']]==(action['layer'],action['unit'])
                    x,amp,w=apply(x,action['unit_index'])
                    assert abs(amp)<=c['max_atom_amplitude']
                    assert np.sqrt(np.mean(w*w))<=c['max_write_rms']
                    assert np.sqrt(np.mean(x*x))<=c['max_live_rms']
                    assert np.max(abs(bank.P@(x-scope['anchor'])))<=c['domain_radius']+1e-6
                cs=scope['readout'];q=native.consumer(x,cs['layer'],cs['indices'])
                assert array_sha(x)==result['final_sha256'] and np.array_equal(q,np.asarray(result['final_q'],dtype=np.float32))
                lo=np.array(start['lower'],dtype=np.float32);hi=np.array([np.inf if v is None else v for v in start['upper']],dtype=np.float32)
                assert metrics(q[start['mask']],lo,hi,c['target_count'])==result['metrics'];replayed+=1
            for r in sample:
                match=re.match(r'k(\d+)-t(\d+)-',r['stage'])
                if int(match[1])!=k:continue
                start=starts[r['stage']];scope=scopes[start['condition']['scope']]
                x=np.load(out/'cases'/f't{start["condition"]["task"]}.npz')['root'].copy();path=[];node=r['parent']
                while node:
                    parent_id,j=tree[(r['stage'],node)];path.append(j);node=parent_id
                for j in reversed(path):x,_,_=apply(x,j)
                j=r['action']['unit_index'];after,amp,w=apply(x,j)
                assert amp==r['amplitude']
                if r.get('reason')=='amplitude':assert abs(amp)>c['max_atom_amplitude'];checked+=1;continue
                assert array_sha(w)==r['write_sha256']
                legal=bool(np.isfinite(after).all() and np.max(abs(bank.P@(after-scope['anchor'])))<=c['domain_radius']+1e-6 and
                           np.sqrt(np.mean(w*w))<=c['max_write_rms'] and np.sqrt(np.mean(after*after))<=c['max_live_rms'])
                assert legal==r['legal']
                if legal:
                    cs=scope['readout'];q=native.consumer(after,cs['layer'],cs['indices'])
                    assert array_sha(after)==r['after_sha256'] and array_sha(q)==r['q_sha256']
                checked+=1
            print(json.dumps(dict(bank_size=k,graphs_verified=replayed)),flush=True)
        assert replayed==len(s['results']) and checked==512
        assert native.parameter_digest()==native.before and sha(out/'source.npz')==sealed['source_sha256']
        result=dict(passed=True,selected_graphs_native_bit_exact=replayed,branches_sampled=checked,total_recorded_branches=branches,
            native_primitive_ledger_matches=True,weights_unchanged=True,source_unchanged=True,
            verification_seconds=time.perf_counter()-started,native_work=native.counts)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    finally:native.close()

if __name__=='__main__':main()
