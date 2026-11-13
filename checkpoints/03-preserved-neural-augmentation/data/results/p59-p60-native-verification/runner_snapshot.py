"""Original-weight re-execution of P59/P60 held-out native observations."""
from pathlib import Path
import json,time,gc,shutil,hashlib
import numpy as np
from native import NativeField

LAB=Path(__file__).resolve().parents[1]
def load(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def main():
    p59=LAB/'results/p59-preserved-return-augmentation-001';p60=LAB/'results/p60-shared-observer-transport-001';assert (p60/'summary.json').exists()
    out=LAB/'results/p59-p60-native-verification';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(Path(__file__).parent/'native.py',out/'native.py');records=[]
    with (out/'native-work.jsonl').open('w') as log:
      def emit(v):log.write(json.dumps(v)+'\n')
      for label in ['0.6B','1.7B']:
        receipt=json.loads((p59/'models'/label/'checkpoint.json').read_text());native=NativeField(receipt['local_path'],emit,2,32);expected=json.loads((p59/'models'/label/'model-summary.json').read_text())['parameter_sha256'];assert native.before[0]==expected
        for run in [p59,p60]:
            gold=load(run/'models'/label/'native-checks.npz');n=native.n;native.stage=f'{run.name}-saved-step-verification';count=0
            for X,Y in zip(gold['inputs'],gold['outputs']):
                flat=X.reshape(-1,n);pad=(-len(flat))%32;test=np.concatenate([flat,np.repeat(flat[-1:],pad,axis=0)]) if pad else flat
                result=native.run(test)[:len(flat)].reshape(Y.shape);assert np.array_equal(result,Y),(label,run.name,float(np.max(abs(result-Y))))
                count+=len(flat)
            rec=dict(model=label,run=run.name,native_condition_steps=count,native_outputs_bitexact=True);records.append(rec);print(json.dumps(rec),flush=True)
        assert native.digest()==native.before;records.append(dict(model=label,weights_unchanged=True,parameter_sha256=native.before[0],native_work=native.counts,native_seconds=native.seconds));del native;gc.collect()
        import mlx.core as mx;mx.clear_cache()
    report=dict(records=records,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    dest=LAB.parent.parent/'outputs/neural-stimulus/preserved-augmentation';dest.mkdir(parents=True,exist_ok=True);(dest/'native-verification.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
