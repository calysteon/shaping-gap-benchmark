"""Extract original parameter-backed SwiGLU fragments; never invoke transformer."""
from pathlib import Path
import json,sys,time,hashlib,argparse
import numpy as np
HERE=Path(__file__).resolve().parent;LAB=HERE.parent
from native_substrate import Substrate

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model-06',type=Path,required=True);parser.add_argument('--model-17',type=Path,required=True);parser.add_argument('--output',type=Path,default=HERE.parent/'runs/extraction');args=parser.parse_args();out=args.output;out.parent.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=False);records=[];all_started=time.perf_counter()
 for label,receiptfile in [('0.6B','checkpoint-receipt.json'),('1.7B','checkpoint-1.7b-receipt.json')]:
  receipt=next(r['receipt'] for r in json.loads((HERE.parent/'data/banks/manifest.json').read_text())['records'] if r['model']==label);receipt['local_path']=str(args.model_06 if label=='0.6B' else args.model_17);events=[];ts=time.perf_counter();native=Substrate(receipt['local_path'],events.append);mx=native.mx
  def dense(m):
   if hasattr(m,'scales'):
    return mx.dequantize(m.weight,m.scales,m.biases,group_size=m.group_size,bits=m.bits).astype(mx.float32)
   return m.weight.astype(mx.float32)
  print(json.dumps(dict(stage='loaded',model=label,width=native.width,digest=native.before[0])),flush=True)
  for seed in [151001,151002,151003]:
   bank_started=time.perf_counter();before_counts=dict(native.counts)
   rng=np.random.default_rng(seed);L=rng.integers(0,len(native.layers),24);J=rng.integers(0,native.hidden,24)
   G=[];U=[];D=[];eps=[];checks=[]
   # Dequantize one layer at a time; frozen arrays are copied, not fitted.
   for layer in sorted(set(L.tolist())):
    mod=native.layers[layer];g=dense(mod.mlp.gate_proj);u=dense(mod.mlp.up_proj);d=dense(mod.mlp.down_proj);mx.eval(g,u,d)
    gn=np.asarray(g);un=np.asarray(u);dn=np.asarray(d);norm=np.asarray(mod.post_attention_layernorm.weight.astype(mx.float32));e=mod.post_attention_layernorm.eps
    for i in np.flatnonzero(L==layer):
     unit=int(J[i]);G.append((i,gn[unit].copy()*norm));U.append((i,un[unit].copy()*norm));D.append((i,dn[:,unit].copy()));eps.append((i,float(e)))
    del g,u,d
   G=np.stack([v for i,v in sorted(G)]);U=np.stack([v for i,v in sorted(U)]);D=np.stack([v for i,v in sorted(D)]);eps=np.array([v for i,v in sorted(eps)])
   # Check extracted responses and writes against original guarded modules.
   for k in range(3):
    x=rng.normal(size=native.width).astype(np.float32)*[.12,.25,.5][k];g=(G@x)/np.sqrt(np.mean(x*x)+eps);u=(U@x)/np.sqrt(np.mean(x*x)+eps);a=g/(1+np.exp(-g))*u
    truth=np.empty(24)
    for layer in sorted(set(L.tolist())):
     values=native.activity(x,int(layer))
     for i in np.flatnonzero(L==layer):truth[i]=values[int(J[i])]
    error=float(np.max(np.abs(a-truth)));relative=float(np.linalg.norm(a-truth)/max(np.linalg.norm(truth),1e-12))
    # MLX original weights may be bfloat16-rounded relative to float32 packing.
    checks.append(dict(kind='native_amplitudes',absolute_error=error,relative_error=relative))
    for i in [0,7,15,23]:
     w=native.write(int(L[i]),int(J[i]),float(truth[i]));packed=D[i]*truth[i]
     checks.append(dict(kind='native_write',relative_error=float(np.linalg.norm(w-packed)/max(np.linalg.norm(w),1e-12)),absolute_error=float(np.max(abs(w-packed)))))
   assert max(q['relative_error'] for q in checks)<.02,checks
   path=out/f'{label}-{seed}.npz';np.savez_compressed(path,G=G,U=U,D=D,eps=eps,layers=L,units=J)
   records.append(dict(model=label,seed=seed,file=path.name,sha256=sh(path),width=native.width,fragments=24,checks=checks,extraction_seconds=time.perf_counter()-bank_started,native_primitives={k:native.counts.get(k,0)-before_counts.get(k,0) for k in native.counts},full_matrix_dequantizations=3*len(set(L.tolist())),dequantized_coefficients=3*len(set(L.tolist()))*native.width*native.hidden,guarded_original_calls=['model','body','block','attention','embedding','unembedding'],parameter_sha256=native.before[0],receipt=receipt,packed_bytes=sum(a.nbytes for a in [G,U,D,eps,L,J])))
   print(json.dumps(dict(stage='bank_extracted',model=label,seed=seed,max_relative_error=max(q['relative_error'] for q in checks))),flush=True)
  assert native.parameter_digest()==native.before
  native.close();del native;mx.clear_cache()
 dump(out/'manifest.json',dict(records=records,wall_seconds=time.perf_counter()-all_started,extraction_source_sha256=sh(Path(__file__)),execution_note='Original modules used only for guarded primitive fidelity checks. Transformer, attention and token execution forbidden. Native kernels retained as extracted float32 row/column arithmetic; original mixed precision rounding separately quantified.'))
if __name__=='__main__':main()
