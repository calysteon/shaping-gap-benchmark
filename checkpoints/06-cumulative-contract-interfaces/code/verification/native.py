"""Independent original-coordinate checks of recruited native response/partials."""
from pathlib import Path
import json,numpy as np,time
CHECKPOINT=Path(__file__).resolve().parents[2];DATA=CHECKPOINT/'data/experiments';PREVIOUS=CHECKPOINT.parent/'05-component-learning-cycle';OUTPUT=CHECKPOINT/'runs/replay';OUTPUT.mkdir(parents=True,exist_ok=True)

def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def original(model,seed,index):
 if index<0:
  p=load(PREVIOUS/'data/banks'/f'{model}-{seed}.npz');j=-index-1
 else:
  q,j=divmod(index,16);p=load(CHECKPOINT/'data/pools'/f'{model}-{seed}'/f'chunk-{q:03d}.npz')
 return {k:v[j] for k,v in p.items()}
def full_native(raw,root,S):
 U=.8*root['Ux'].T;X=root['anchor']+S@U.T;h=X.shape[1];r=np.sqrt(np.mean(X*X,axis=1)+raw['eps']);g=X@raw['G']/r;u=X@raw['U']/r;sig=1/(1+np.exp(np.clip(-g,-60,60)));a=g*sig*u;dg=raw['G'][None]/r[:,None]-g[:,None]*X/(h*r[:,None]**2);du=raw['U'][None]/r[:,None]-u[:,None]*X/(h*r[:,None]**2);da=((sig+g*sig*(1-sig))*u)[:,None]*dg+(g*sig)[:,None]*du;return a,da@U

def main():
 started=time.perf_counter();wvala=0.;wgrad=0.;count=0;perstudy={}
 for name in ['p64-bounded-recruitment-cumulative-002','p65-program-contract-interfaces-001']:
  n=0
  for ctx in (DATA/name/'contexts').iterdir():
   model,seed=ctx.name.split('-')[:2];seed=int(seed);root=load(PREVIOUS/'data/experiment/contexts'/f'{model}-{seed}-integrated/runtime.npz')
   for p in ctx.glob('round-*/port-*-source.npz'):
    index=int(p.name[len('port-'):-len('-source.npz')]);raw=original(model,seed,index);z=load(p);A,G=full_native(raw,root,z['samples']);truth=z['values'][:,0];grad=z['partials'][:,:,0];ev=float(np.linalg.norm(A-truth)/max(np.linalg.norm(truth),1e-20));eg=float(np.linalg.norm(G-grad)/max(np.linalg.norm(grad),1e-20));wvala=max(wvala,ev);wgrad=max(wgrad,eg);count+=len(A);n+=1
  perstudy[name]=n
 assert wvala<1e-8 and wgrad<1e-8
 result=dict(passed=True,local_port_interfaces=sum(perstudy.values()),native_response_points=count,maximum_relative_value_error=wvala,maximum_relative_partial_error=wgrad,per_study=perstudy,wall_seconds=time.perf_counter()-started,scope='Direct full original-width SwiGLU/normalization arithmetic on original materialized rows; independent of the compiled9-coordinate kernel. Original mixed-precision materialization error is separately in pool provenance.')
 for name in perstudy:(OUTPUT/(name+'-native.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(result)
if __name__=='__main__':main()
