"""Replay the fixed native programs using only the acquired structured interface.

This process imports NumPy, not MLX, and does not open the pretrained checkpoint.
It checks a finite component bank on the existing three reachable domains.
"""
from pathlib import Path
import argparse
import json
import resource
import sys
import time
import numpy as np
from interfaces import AnalyticReference


def main():
    root=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=root/'data/source')
    parser.add_argument('--run',type=Path,default=root/'data/benchmark')
    parser.add_argument('--output',type=Path,default=root/'runs/compiled-replay.json')
    args=parser.parse_args();parent=args.source;out=args.run
    s=json.loads((out/'summary.json').read_text());c=s['config']
    physical=np.load(parent/'physical-coordinates.npz');P=physical['P']
    refs=[];anchors=[]
    for i in range(3):
        p=parent/'scopes'/str(i);a=np.load(p/'analytic.npz')
        refs.append(AnalyticReference(**{k:a[k] for k in a.files}));anchors.append(np.load(p/'domain.npz')['anchor'])
    assert 'mlx.core' not in sys.modules and 'mlx_lm' not in sys.modules
    starts={}
    if (out/'contracts.json').exists():
        starts={int(k):v for k,v in json.loads((out/'contracts.json').read_text()).items()}
    else:
        with (out/'runs.jsonl').open() as f:
            for line in f:
                if '"kind": "case_start"' in line:
                    row=json.loads(line);starts[row['task']]=row
    started=time.perf_counter();rows=[]
    for result in s['results']:
        scope=result['condition']['scope'];model=refs[scope]
        root=np.load(out/'cases'/f't{result["task"]}.npz')['root']
        z=P@(root-anchors[scope])
        for b in result['graph']:
            amp,_=model.values(z);z=z.copy();z[b['unit_index']]+=amp[0,b['unit_index']]
        _,q=model.values(z);q=q[0];native_q=np.array(result['final_q'])
        st=starts[result['task']];mask=st['mask']
        lo=np.array(st['lower']);hi=np.array([np.inf if v is None else v for v in st['upper']])
        error=np.maximum(lo-q[mask],0)+np.maximum(q[mask]-hi,0)
        solved=bool(np.all(error<=1e-5))
        rows.append(dict(task=result['task'],method=result['method'],max_abs_readout_error=float(np.max(abs(q-native_q))),
            compiled_satisfied=solved,native_satisfied=result['metrics']['satisfied'],q=q.tolist()))
    summary=dict(checkpoint_loaded=False,mlx_imported=False,programs=len(rows),
        quality_agreement=sum(r['compiled_satisfied']==r['native_satisfied'] for r in rows),
        max_abs_readout_error=max(r['max_abs_readout_error'] for r in rows),
        structured_reference_agreement=sum(r['compiled_satisfied']==r['native_satisfied'] for r in rows if r['method']=='analytic_pair'),
        structured_reference_programs=sum(r['method']=='analytic_pair' for r in rows),
        acquired_payload_bytes=s['model_payloads']['analytic'],
        structured_formula_payload_bytes=sum(r.payload() for r in refs),
        replay_seconds=time.perf_counter()-started,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rows=rows,scope='Approximate real-arithmetic replay of the finite22-unit bank and fixed monitored readouts on the existing three domains; original native checks remain authoritative.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


if __name__=='__main__':main()
