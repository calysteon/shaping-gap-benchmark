"""Replay all recorded programs using the acquired bank, NumPy and no model."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from factored import Bank
from trace_io import open_trace


def main():
    root=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=root/'runs/interface-replay.json')
    args=parser.parse_args();source=root/'data/scaling/source.npz'
    started=time.perf_counter();rows=[]
    for study in ['scaling','functions']:
        directory=root/'data'/study;summary=json.loads((directory/'summary.json').read_text());c=summary['config']
        starts={}
        with open_trace(directory) as stream:
            for line in stream:
                if '"kind": "case_start"' in line:
                    r=json.loads(line);starts[r['stage']]=r
        banks={}
        for result in summary['results']:
            k=result.get('bank_size',512)
            if k not in banks:banks[k]=Bank(source,k)
            bank=banks[k]
            if study=='scaling':
                case=result['condition']['task'];scope=result['condition']['scope']
                frames=np.load(directory/'cases'/f't{case}.npz')['root'][None].astype(np.float64)
            else:
                case=result['case'];scope=1
                sample=np.load(directory/'cases'/f't{case}.npz');frames=sample['roots'].astype(np.float64);labels=sample['labels']
            for action in result['graph']:
                j=action['unit_index']
                for i,x in enumerate(frames):
                    amplitude,_=bank.values(x,scope);frames[i]=x+amplitude[j]*bank.D[:,j]
            q=np.array([bank.values(x,scope)[1] for x in frames]);actual=np.asarray(result['final_q'])
            if study=='scaling':
                actual=actual[None];st=starts[f'k{k}-t{case}-{result["method"]}'];mask=st['mask']
                lo=np.array(st['lower']);hi=np.array([np.inf if v is None else v for v in st['upper']])
                error=np.maximum(lo-q[0,mask],0)+np.maximum(q[0,mask]-hi,0)
                native=result['metrics']['satisfied'];replayed=bool(np.all(error<=1e-5));exact=None
            else:
                error=np.maximum(c['output_margin']-(2*labels.astype(int)-1)*q[:,0],0)
                native=result['metrics']['robust_satisfied'];replayed=bool(np.all(error<=1e-5));exact=bool(np.array_equal(q[:,0]>0,labels))
                assert exact==result['metrics']['boolean_correct']
            rows.append(dict(study=study,case=case,bank_size=k,method=result['method'],
                max_abs_readout_error=float(np.max(abs(q-actual))),native_satisfied=native,
                interface_satisfied=replayed,boolean_exact=exact))
    assert 'mlx.core' not in sys.modules and 'mlx_lm' not in sys.modules
    report=dict(checkpoint_loaded=False,mlx_imported=False,programs=len(rows),
        quality_agreement=sum(r['native_satisfied']==r['interface_satisfied'] for r in rows),
        max_abs_readout_error=max(r['max_abs_readout_error'] for r in rows),
        replay_seconds=time.perf_counter()-started,
        scope='Approximate finite-bank replay; independent native float32 verification remains authoritative.',rows=rows)
    assert report['quality_agreement']==len(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
