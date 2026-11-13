"""Fixed pretrained recruitment inventory, streamed in16-row chunks at acquisition."""
from pathlib import Path
import sys,time,json,hashlib,resource
import numpy as np
HERE=Path(__file__).resolve().parent;LAB=HERE.parent;sys.path.insert(0,str(LAB/'keystone-study'))
from substrate import Substrate

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=HERE/'pools';out.mkdir(exist_ok=False);records=[];start=time.perf_counter()
 for label,receiptfile in [('0.6B','checkpoint-receipt.json'),('1.7B','checkpoint-1.7b-receipt.json')]:
  receipt=json.loads((LAB/'open-weight'/receiptfile).read_text());events=[];native=Substrate(receipt['local_path'],events.append);mx=native.mx
  def dense(m):
   return mx.dequantize(m.weight,m.scales,m.biases,group_size=m.group_size,bits=m.bits).astype(mx.float32)
  for seed in [151001,151002,151003]:
   began=time.perf_counter();before=dict(native.counts);parent=np.load(LAB/'component-cycle-study/banks'/f'{label}-{seed}.npz');excluded=set(zip(parent['layers'].tolist(),parent['units'].tolist()));rng=np.random.default_rng(seed+4100000);units=[]
   while len(units)<512:
    pair=(int(rng.integers(0,len(native.layers))),int(rng.integers(0,native.hidden)))
    if pair not in excluded and pair not in units:units.append(pair)
   L,J=np.array(units).T;G={};U={};D={};E={}
   for layer in sorted(set(L.tolist())):
    l=native.layers[layer];gg=dense(l.mlp.gate_proj);uu=dense(l.mlp.up_proj);dd=dense(l.mlp.down_proj);mx.eval(gg,uu,dd);g=np.asarray(gg);u=np.asarray(uu);d=np.asarray(dd);norm=np.asarray(l.post_attention_layernorm.weight.astype(mx.float32))
    for i in np.flatnonzero(L==layer):G[i]=g[J[i]].copy()*norm;U[i]=u[J[i]].copy()*norm;D[i]=d[:,J[i]].copy();E[i]=float(l.post_attention_layernorm.eps)
    del gg,uu,dd
   full=dict(G=np.stack([G[i] for i in range(512)]),U=np.stack([U[i] for i in range(512)]),D=np.stack([D[i] for i in range(512)]),eps=np.array([E[i] for i in range(512)]),layers=L,units=J)
   fidelity=[]
   for k in range(3):
    x=rng.normal(size=native.width).astype(np.float32)*[.12,.25,.5][k];g=full['G']@x/np.sqrt(np.mean(x*x)+full['eps']);u=full['U']@x/np.sqrt(np.mean(x*x)+full['eps']);amp=g/(1+np.exp(-g))*u;truth=np.empty(512)
    for layer in sorted(set(L.tolist())):
     a=native.activity(x,int(layer));idx=np.flatnonzero(L==layer);truth[idx]=a[J[idx]]
    fidelity.append(dict(relative_error=float(np.linalg.norm(amp-truth)/max(np.linalg.norm(truth),1e-20)),max_absolute_error=float(np.max(abs(amp-truth)))))
   folder=out/f'{label}-{seed}';folder.mkdir();chunks=[]
   for i in range(0,512,16):
    p=folder/f'chunk-{i//16:03d}.npz';np.savez_compressed(p,**{k:v[i:i+16] for k,v in full.items()});chunks.append(dict(file=p.name,sha256=sha(p),start=i,count=16,bytes=p.stat().st_size))
   record=dict(model=label,seed=seed,fragments=512,width=native.width,folder=folder.name,chunks=chunks,parameter_sha256=native.before[0],original_module_fidelity=fidelity,packed_array_bytes=sum(v.nbytes for v in full.values()),native_primitives={k:native.counts.get(k,0)-before.get(k,0) for k in native.counts},dequantized_coefficients=3*len(set(L.tolist()))*native.width*native.hidden,wall_seconds=time.perf_counter()-began)
   records.append(record);dump(folder/'manifest.json',record);print(json.dumps(dict(model=label,seed=seed,fragments=512,fidelity=max(v['relative_error'] for v in fidelity),seconds=record['wall_seconds'])),flush=True)
  assert native.parameter_digest()==native.before;native.close();del native;mx.clear_cache()
 dump(out/'manifest.json',dict(records=records,wall_seconds=time.perf_counter()-start,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sha(Path(__file__)),scope='Candidate inventory extraction before any new task outcome. Original modules only for fidelity, source graph guarded. Pool is an immutable substrate archive, not retained active controller state.'))
if __name__=='__main__':main()
