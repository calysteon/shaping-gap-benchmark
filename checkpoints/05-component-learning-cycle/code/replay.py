"""Independent full-coordinate fragment replay, frozen-program scoring and seals."""
from pathlib import Path
from itertools import combinations
import hashlib,json,sys,time,argparse
import numpy as np
HERE=Path(__file__).resolve().parent;LAB=HERE.parent

def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def targets(drive,start,n,descriptors):
 idx=np.arange(start,start+n);u=np.stack([drive[idx-k] for k in range(16)])
 # Recurrence is independent of the closed-form formulas in the runner.
 polys={0:np.ones_like(u),1:u}
 for d in range(2,5):polys[d]=((2*d-1)*u*polys[d-1]-(d-1)*polys[d-2])/d
 return np.column_stack([np.prod([np.sqrt(2*d+1)*polys[d][lag] for lag,d in desc['parts']],axis=0) for desc in descriptors])
def r2(pred,y):return 1-((pred-y)**2).sum(0)/np.maximum(((y-y.mean(0))**2).sum(0),1e-30)
def predict(X,h):return ((X-h['feature_mean'])/h['feature_scale'])@h['readout']+h['target_mean']
def profile_value(p,z):return p['profile_c']+z@p['profile_L'].T+.5*np.einsum('ni,oij,nj->no',z,p['profile_H'],z)
def native_full(p,X):
 G,U=p['bank_G'],p['bank_U'];r=np.sqrt(np.mean(X*X,axis=1)[:,None]+p['bank_eps'][None]);gate=X@G.T/r;up=X@U.T/r
 return gate/(1+np.exp(np.clip(-gate,-60,60)))*up

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=HERE.parent/'data/experiment');parser.add_argument('--output',type=Path,default=HERE.parent/'runs/replay.json');args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter();out=args.data;s=json.loads((out/'summary.json').read_text());c=s['config'] if 'config' in s else json.loads((out/'protocol.json').read_text())['config'];worst=0.;native_err=0.;fragment_amplitude_err=0.;integral_error=0.;checked_fields=0;checked_routes=0;score_vectors=0;seals=0;rows=[];partial_error=0;native_relative_error=0.;integral_relative_error=0.
 for res in s['results']:
  d=out/'contexts'/res['context'];p=load(d/'runtime.npz');assert np.all(p['bank_eps']==p['bank_eps'][0]);desc=json.loads((d/'targets.json').read_text());multi=np.array([v['degree']>=2 and v['distinct_lags']>=2 for v in desc]);Cs=load(d/'returns.npz')['returns'];heads=[load(d/f'head-{j}.npz') for j in range(len(Cs))];maps=[load(d/f'interface-{j}.npz') for j in range(len(Cs))]
  seal=json.loads((d/'construction-seal.json').read_text())
  for file,h in seal['files'].items():assert sha(d/file)==h;seals+=1
  pickseal=json.loads((d/'selection-seal.json').read_text());assert sha(d/'selection.json')==pickseal['selection_sha256'];assert sha(d/'construction-seal.json')==pickseal['construction_seal_sha256'];seals+=2
  for file in d.glob('checks-*.npz'):
   trace=load(file)
   for t in range(len(trace['time'])):
    v=trace['coordinate_v'][t];a=p['anchor']+v@p['B'].T;Y=native_full(p,a);ia=trace['first'][t];ib=trace['second'][t];nr=np.arange(len(v))[:,None];va=Y[nr,ia];live=a+np.einsum('nkh,nk->nh',p['bank_D'][ia],va);Z=native_full(p,live);vb=Z[nr,ib];full=np.einsum('nkh,nk->nh',p['bank_D'][ia],va)+np.einsum('nkh,nk->nh',p['bank_D'][ib],vb);y=full@p['output_basis']
    assert np.allclose(y,trace['output'][t],rtol=2e-5,atol=2e-5)
    native_relative_error=max(native_relative_error,float(np.linalg.norm(y-trace['output'][t])/max(np.linalg.norm(y),1e-20)))
    native_err=max(native_err,float(np.max(abs(y-trace['output'][t]))));fragment_amplitude_err=max(fragment_amplitude_err,float(np.max(abs(va-trace['first_amplitudes'][t]))),float(np.max(abs(vb-trace['second_amplitudes'][t]))));checked_fields+=len(v)
    # Independently recompute conditional ranking from frozen profile arrays.
    pred=profile_value(p,v);norm=np.linalg.norm(p['bank_D'],axis=1);first=np.argsort(-abs(pred)*norm,axis=1,kind='stable')[:,:c['active_fragments_per_stage']];assert np.array_equal(first,ia)
    delta=np.einsum('nkd,nk->nd',p['W'].T[ia],va)
    grad=p['profile_L'].T[None]+np.einsum('oij,nj->nio',p['profile_H'],v)
    first_order=np.einsum('ndo,nd->no',grad,delta);curvature=.5*np.einsum('ni,oij,nj->no',delta,p['profile_H'],delta)
    integrated=profile_value(p,v+delta)-pred;integral_error=max(integral_error,float(np.max(abs(first_order+curvature-integrated))));integral_relative_error=max(integral_relative_error,float(np.max(abs(first_order+curvature-integrated)))/max(1.,float(np.max(abs(first_order)+abs(curvature)))))
    cond=first_order if res['variant']=='first_order' else integrated[:,p['permutation']] if res['variant']=='shuffled' else integrated
    sc=abs(cond)*norm;sc[nr,ia]=-np.inf;second=np.argsort(-sc,axis=1,kind='stable')[:,:c['active_fragments_per_stage']];assert np.array_equal(second,ib);checked_routes+=len(v)*2
  # Independently refit each head and shared interface from source arrays.
  src=load(d/'source.npz');Y=targets(src['drive'],c['burn'],c['training_steps'],desc)
  for j in range(len(Cs)):
   X=src['features'][j,0,c['burn']:].astype(float);h=heads[j];A=(X-h['feature_mean'])/h['feature_scale'];B=np.linalg.solve(A.T@A+c['readout_ridge']*len(A)*np.eye(A.shape[1]),A.T@(Y-Y.mean(0)));assert np.allclose(B,h['readout'],atol=1e-10,rtol=1e-9)
   m=maps[j];base=src['features'][0,0,c['burn']:].astype(float);XX=(X-m['new_feature_mean'])/m['new_feature_scale'];YY=(base-m['old_feature_mean'])/m['old_feature_scale'];lam=c['transport_ridge']*len(X);T=np.linalg.solve(XX.T@XX+lam*np.eye(XX.shape[1]),XX.T@YY+lam*np.eye(XX.shape[1]));assert np.array_equal(T,m['transport'])
  feedback=[];fresh=[]
  for group,blocklist in [('feedback',feedback),('fresh',fresh)]:
   recorded=json.loads((d/f'{group}-scores.json').read_text())
   for block in range(2):
    z=load(d/f'{group}-{block}.npz');Y=targets(z['drive'],c['burn'],c['assessment_steps'],desc);replay=[]
    for j in range(len(Cs)):
     X=z['features'][j,0,c['burn']:].astype(float);m=maps[j];pnew=r2(predict(X,heads[j]),Y);direct=r2(predict(X,heads[0]),Y);old=r2(((X-m['new_feature_mean'])/m['new_feature_scale'])@m['transport']@heads[0]['readout']+heads[0]['target_mean'],Y)
     if not z['valid'][j,0]:pnew[:]=-1;direct[:]=-1;old[:]=-1
     for name,v in [('r2',pnew),('old_r2',old),('direct_old_r2',direct)]:
      worst=max(worst,float(np.max(abs(v-np.array(recorded[block][j][name])))));score_vectors+=1
     replay.append(dict(quality=float(np.maximum(pnew[multi],0).sum()),r2=pnew,old=old,bounded=bool(z['valid'][j,0])))
    blocklist.append(replay)
  old=(feedback[0][0]['r2']>=c['reliability_threshold'])&(feedback[1][0]['r2']>=c['reliability_threshold']);eligible=[]
  for j in range(1,len(Cs)):
   gains=[(feedback[b][j]['quality']-feedback[b][0]['quality'])/max(feedback[b][0]['quality'],.25) for b in range(2)];ok=old.any() and min(gains)>=c['quality_acceptance'] and all(feedback[b][j]['bounded'] and feedback[b][0]['bounded'] and np.all(feedback[b][j]['old'][old]>=c['reliability_threshold']) for b in range(2))
   if ok:eligible.append((min(gains),-j,j))
  j=max(eligible)[2] if eligible else 0;choice=json.loads((d/'selection.json').read_text());assert choice['selected_slot']==j;assert choice['old_program_ids']==np.flatnonzero(old).tolist();assert j==res['selected_slot']
  gains=[(fresh[b][j]['quality']-fresh[b][0]['quality'])/max(fresh[b][0]['quality'],.25) for b in range(2)];ret=min(int(np.sum(fresh[b][j]['old'][old]>=c['reliability_threshold'])) for b in range(2));assert np.allclose(gains,res['relative_gains'],atol=1e-9);assert ret==res['old_retained_both']
  rows.append(dict(context=res['context'],selected_slot=j,old=int(old.sum()),retained=ret,strict_joint=res['strict_joint']))
 print(json.dumps(dict(max_score_error=worst,max_full_space_error=native_err,max_integral_error=integral_error)),flush=True)
 assert worst<1e-8;assert native_relative_error<2e-5;assert integral_relative_error<1e-10
 record=dict(passed=True,score_vectors=score_vectors,max_score_error=worst,full_space_native_fields=checked_fields,individual_binding_choices=checked_routes,max_native_output_error=native_err,max_native_output_relative_error=native_relative_error,native_comparison_tolerance=dict(rtol=2e-5,atol=2e-5),max_native_amplitude_error=fragment_amplitude_err,max_quadratic_integral_identity_error=integral_error,max_quadratic_integral_scaled_error=integral_relative_error,seals_checked=seals,all_selections_replayed=True,all_heads_refitted=True,all_shared_interfaces_refitted=True,wall_seconds=time.perf_counter()-start,original_transformer_calls=0,contexts=rows)
 dump(args.output,record);print(json.dumps(record))
if __name__=='__main__':main()
