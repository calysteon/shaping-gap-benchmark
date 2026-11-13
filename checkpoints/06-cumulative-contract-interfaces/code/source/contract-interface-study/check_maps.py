"""Independent stationarity check of every acquired program-contract map."""
from pathlib import Path
import json,time,numpy as np
from independent_targets import targets

def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
 root=Path(__file__).resolve().parent.parent/'results/p65-program-contract-interfaces-001';c=json.loads((root/'summary.json').read_text())['config'];desc=json.loads((root.parent/'p63-component-acquisition-reuse-001/contexts/0.6B-151003-integrated/targets.json').read_text());worst=0.;count=0;started=time.perf_counter()
 for d in root.glob('contexts/*/round-*'):
  meta=json.loads((d/'candidate-metadata.json').read_text());seal=json.loads((d/'construction-seal.json').read_text());src=load(d/'source.npz');parentmap=load(d/'candidate-0/map.npz');PARENT=src['features'][0,c['burn']:].astype(float)@parentmap['A']+parentmap['b'];Y=targets(src['drive'],c['burn'],c['training_steps'],desc);mu=PARENT.mean(0);scale=np.maximum(PARENT.std(0),1e-8);Z=(PARENT-mu)/scale;ys=np.maximum(Y.std(0),1e-5);libs=[]
  for key in seal['prior_library_hashes']:
   folder=d.parent/f'library-{key}';libs.append(dict(h=load(folder/'head.npz'),m=load(folder/'input-map.npz'),ids=json.loads((folder/'program-ids.json').read_text())))
  for j,m in enumerate(meta):
   if m['interface']!='program_contract':continue
   raw=src['features'][j,c['burn']:].astype(float);nm=raw.mean(0);ns=np.maximum(raw.std(0),1e-8);X=np.column_stack([(raw-nm)/ns,np.ones(len(raw))]);a=load(d/f'candidate-{j}/map.npz');P=np.vstack([a['A']*ns[:,None]/scale[None],(a['b']-mu+nm@a['A'])/scale]);prior=np.vstack([np.eye(24),np.zeros((1,24))]);grad=m['feature_weight']*X.T@(X@P-Z);rhs=m['feature_weight']*X.T@Z;canon=(X@P)*scale+mu
   for lib in libs:
    h=lib['h'];A=lib['m']['A'];b=lib['m']['b'];W=scale[:,None]*(A@(h['readout']/h['feature_scale'][:,None]));bias=((mu@A+b-h['feature_mean'])/h['feature_scale'])@h['readout']+h['target_mean'];weights=np.ones(Y.shape[1]);weights[lib['ids']]=4.;weights/=ys**2*len(libs);out=(canon@A+b-h['feature_mean'])/h['feature_scale']@h['readout']+h['target_mean'];grad+=X.T@((out-Y)*weights)@W.T;rhs+=X.T@((Y-bias)*weights)@W.T
   lam=m['ridge']*len(raw);grad+=lam*(P-prior);rhs+=lam*prior;relative=float(np.linalg.norm(grad)/max(np.linalg.norm(rhs),1e-20));worst=max(worst,relative);count+=1
 assert worst<1e-7
 result=dict(passed=True,contract_maps=count,maximum_relative_stationarity_error=worst,criterion='First-order stationarity of actual source objective via independent forward program-output losses; no reused Sylvester/eigen solver.',wall_seconds=time.perf_counter()-started);(root/'map-objective-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
