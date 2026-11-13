"""Fresh native graph replay, sampled branch replay and source derivative checks."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from substrate import Substrate,array_sha
from planner import metrics


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))


def main():
    root=Path(__file__).resolve().parent.parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source',type=Path,default=root/'data/source')
    p.add_argument('--run',type=Path,default=root/'data/benchmark')
    p.add_argument('--output',type=Path,default=root/'runs/native-verification.json')
    args=p.parse_args();out=args.run;parent=args.source
    s=json.loads((out/'summary.json').read_text());c=s['config']
    if (out/'runner_snapshot.py').exists():
        m=json.loads((out/'manifest.json').read_text())
        assert sha(out/'runner_snapshot.py')==m['source_sha256']
        for n,h in m['helper_sha256'].items():assert sha(out/'helpers'/n)==h
    for n,h in (c.get('inherited_model_sha256') or s['sealed']['model_hashes']).items():assert sha(parent/n)==h
    units=[tuple(v) for v in c['source_units']];physical=np.load(parent/'physical-coordinates.npz');D,P=physical['D'],physical['P']
    scopes=[]
    for i in range(3):
        folder=parent/'scopes'/str(i);j=json.loads((folder/'domain.json').read_text())
        scopes.append(dict(anchor=np.load(folder/'domain.npz')['anchor'],readout=j['readout']))
    # Reservoir samples from all arms/demands, not just successful constructions.
    rng=np.random.default_rng(67001);sample=[];tree={};starts={};native_counts={};nbranches=0
    if (out/'runs.jsonl').exists():
        with (out/'runs.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                if r['kind']=='primitive':native_counts[r['operation']]=native_counts.get(r['operation'],0)+r['vectors']
                elif r['kind']=='case_start':starts[r['stage']]=r
                elif r['kind']=='branch':
                    key=(r['stage'],r['node']);a=r['action'];tree[key]=(r['parent'],a['unit_index'])
                    nbranches+=1
                    if len(sample)<512:sample.append(r)
                    else:
                        j=int(rng.integers(nbranches))
                        if j<512:sample[j]=r
        assert native_counts==s['native_work_total']
    else:
        contracts=json.loads((out/'contracts.json').read_text())
        starts={f't{row["task"]}-{row["method"]}':contracts[str(row['task'])] for row in s['results']}
    native=Substrate(str(args.checkpoint.resolve()),lambda r:None)
    started=time.perf_counter()
    try:
        assert native.before[0]==s['parameter_sha256']
        def apply(x,column):
            layer,j=units[column];h=native.activity(x,layer);amp=float(h[j])
            mask=np.zeros(native.hidden,dtype=np.float32);mask[j]=h[j]
            return x+native.down(layer,mask,'verification_masked_down'),amp
        graphs=0
        for r in s['results']:
            x=np.load(out/'cases'/f't{r["task"]}.npz')['root'].copy()
            for binding in r['graph']:x,_=apply(x,binding['unit_index'])
            sc=scopes[r['condition']['scope']];q=native.consumer(x,sc['readout']['layer'],sc['readout']['indices'])
            assert array_sha(x)==r['final_sha256'] and np.array_equal(q,np.array(r['final_q'],dtype=np.float32))
            start=starts[f't{r["task"]}-{r["method"]}'];mask=start['mask']
            lower=np.array(start['lower'],dtype=np.float32);upper=np.array([np.inf if v is None else v for v in start['upper']],dtype=np.float32)
            assert metrics(q[mask],lower,upper,c['target_count'])==r['metrics']
            graphs+=1
            if graphs%432==0:print(json.dumps(dict(native_graphs_replayed=graphs)),flush=True)
        for r in sample:
            start=starts[r['stage']];x=np.load(out/'cases'/f't{start["task"]}.npz')['root'].copy()
            path=[];node=r['parent']
            while node:
                parent_id,column=tree[(r['stage'],node)];path.append(column);node=parent_id
            for column in reversed(path):x,_=apply(x,column)
            if 'before_sha256' in r:assert array_sha(x)==r['before_sha256']
            column=r['action']['unit_index'];layer,j=units[column];h=native.activity(x,layer)
            assert float(h[j])==r['amplitude']
            if r.get('reason')=='amplitude':assert abs(float(h[j]))>c['max_atom_amplitude'];continue
            mask=np.zeros(native.hidden,dtype=np.float32);mask[j]=h[j]
            w=native.down(layer,mask,'verification_masked_down');assert array_sha(w)==r['write_sha256']
            after=x+w;sc=scopes[start['condition']['scope']]
            z=P@(after-sc['anchor'])
            legal=bool(np.isfinite(after).all() and np.max(abs(z))<=c['domain_radius']+1e-6 and
                np.sqrt(np.mean(w*w))<=c['max_write_rms'] and np.sqrt(np.mean(after*after))<=c['max_live_rms'])
            assert legal==r['legal']
            if legal:
                q=native.consumer(after,sc['readout']['layer'],sc['readout']['indices'])
                assert array_sha(after)==r['after_sha256'] and array_sha(q)==r['q_sha256']
        fd=[];direct=[];frng=np.random.default_rng(67002)
        for scope_id in range(3):
            sc=scopes[scope_id];data=np.load(parent/'scopes'/str(scope_id)/'EK-source.npz')
            for index in [0,7,15,23]:
                for column in [0,13]:
                    v=frng.normal(size=len(units));v/=np.linalg.norm(v);eps=.005
                    def observe(z):
                        x=sc['anchor']+np.asarray(D@z,dtype=np.float32)
                        q0=native.consumer(x,sc['readout']['layer'],sc['readout']['indices'])
                        xa,amp=apply(x,column)
                        qa=native.consumer(xa,sc['readout']['layer'],sc['readout']['indices'])
                        return np.r_[amp,qa.astype(np.float64)-q0]
                    # Use original nominal coordinate reconstructed from stored x
                    # through P; the sub-float32 difference is included in tolerance.
                    z=data['z'][index]
                    finite=(observe(z+eps*v)-observe(z-eps*v))/(2*eps)
                    expected=v@data['g'][index,:,column,:]
                    effect_error=rms(finite[1:]-expected[1:]);amp_error=abs(float(finite[0]-expected[0]))
                    passed=effect_error<=1e-5+.05*rms(expected[1:]) and amp_error<=1e-4+.05*abs(expected[0])
                    fd.append(dict(scope=scope_id,source_index=index,unit_index=column,effect_error_rms=effect_error,
                        effect_derivative_rms=rms(expected[1:]),amplitude_error=amp_error,passed=bool(passed)))
                    # Differentiate the complete native component/readout program
                    # directly. This avoids differencing two already small effects
                    # and independently checks the factored K construction.
                    mx=native.mx;layer,unit=units[column];l=native.layers[layer]
                    original=mx.array(data['x'][index],dtype=mx.float32)
                    direction=mx.array(D@v,dtype=mx.float32)
                    hot=np.zeros((1,native.hidden),dtype=np.float32);hot[0,unit]=1
                    hot=mx.array(hot);actual=[]
                    for coordinate in range(-1,c['readout_size']):
                        def f(t):
                            xx=original+t*direction;normal=l.post_attention_layernorm(xx[None])
                            native.meter('verification_direct_producer_norm',layer)
                            g=l.mlp.gate_proj(normal);u=l.mlp.up_proj(normal)
                            native.meter('verification_direct_gate_projection',layer);native.meter('verification_direct_up_projection',layer)
                            amp=native.swiglu(g,u)[0,unit]
                            if coordinate<0:return amp
                            w=l.mlp.down_proj(hot*amp)[0];native.meter('verification_direct_down_projection',layer)
                            after=native.consumer_mx(xx+w,sc['readout']['layer'],sc['readout']['indices'])
                            before=native.consumer_mx(xx,sc['readout']['layer'],sc['readout']['indices'])
                            return (after-before)[coordinate]
                        derivative=mx.grad(f)(mx.array(0.,dtype=mx.float32));mx.eval(derivative)
                        native.meter('verification_direct_program_vjp',layer);actual.append(float(derivative))
                    actual=np.array(actual);err=rms(actual[1:]-expected[1:])
                    check=dict(scope=scope_id,source_index=index,unit_index=column,
                        expected=expected.tolist(),direct_native=actual.tolist(),effect_error_rms=err,
                        amplitude_error=abs(float(actual[0]-expected[0])),
                        passed=bool(err<=1e-6+.001*rms(expected[1:]) and abs(actual[0]-expected[0])<=1e-6+.001*abs(expected[0])))
                    direct.append(check)
        assert native.parameter_digest()==native.before
        result=dict(passed=all(r['passed'] for r in direct),selected_graphs_native_bit_exact=graphs,stratified_reservoir_branches_checked=len(sample),
            total_recorded_branches=nbranches,source_directional_derivative_checks=fd,
            original_fd_checks_passed=sum(r['passed'] for r in fd),source_direct_autodiff_checks=direct,
            direct_autodiff_checks_passed=sum(r['passed'] for r in direct),
            fd_convergence_record='source-fd-convergence.json',
            restoration_conditions=s['restoration_exact_conditions'],model_hashes_unchanged=True,
            weights_unchanged=True,native_work=native.counts,verification_seconds=time.perf_counter()-started,
            scope='Every selected graph and24 source directions checked;512 branches additionally sampled when a complete trace is supplied. Original small-step finite-difference checks are retained.')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['source_directional_derivative_checks','source_direct_autodiff_checks','native_work']}),flush=True)
        assert result['passed']
    finally:native.close()


if __name__=='__main__':main()
