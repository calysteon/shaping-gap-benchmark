"""Finite acquisition, compatibility, experiential selection, locked fresh reuse."""
import numpy as np
from component import rms,orthogonal
from capacity_engine import capacity_targets
from policy import fit_interface,apply_interface

def rr(pred,y):return 1-np.sum((pred-y)**2,axis=0)/np.maximum(np.sum((y-y.mean(0))**2,axis=0),1e-30)
def fit_head(X,Y,ridge):
 mu=X.mean(0);sd=np.maximum(X.std(0),1e-8);A=(X-mu)/sd;ym=Y.mean(0);B=np.linalg.solve(A.T@A+ridge*len(X)*np.eye(X.shape[1]),A.T@(Y-ym))
 return dict(feature_mean=mu,feature_scale=sd,target_mean=ym,readout=B)
def head(X,h):return ((X-h['feature_mean'])/h['feature_scale'])@h['readout']+h['target_mean']
def metrics(r,desc):
 multi=np.array([d['degree']>=2 and d['distinct_lags']>=2 for d in desc]);return dict(quality=float(np.maximum(r[multi],0).sum()),reliable=int(sum(r>=.5)),r2=r.tolist())

def execute(rt,g,Cs,drive,c,phase,replicas=1,checks=0):
 S=len(Cs);m=8;T=len(drive);rt.stage=phase;Z=np.zeros((S,replicas,m));valid=np.ones((S,replicas),bool);failure=np.full((S,replicas),-1,int)
 if replicas==2:Z[:,1]=np.random.default_rng(rt.seed+830000).normal(size=(S,m))*.1
 features=np.zeros((S,replicas,T,m+c['observable_dimension']),np.float32);states=np.zeros((S,replicas,T,m),np.float32);records=[]
 times=set(np.linspace(0,T-1,checks,dtype=int).tolist()) if checks else set()
 for t,u in enumerate(drive):
  stimulus=np.concatenate([Z.reshape(-1,8),np.full((S*replicas,1),c['input_gain']*u)],axis=1)
  if t in times:Y,rec=rt.response(stimulus,True);rec.update(time=t,prior=Z.copy());records.append(rec)
  else:Y=rt.response(stimulus)
  delta=Y.reshape(S,replicas,-1)-g['base'];Z=np.einsum('smn,srn->srm',Cs,delta,optimize=True)
  good=np.isfinite(Z).all(2)&(np.max(abs(Z),axis=2)<=c['state_limit']);failure[(~good)&valid]=t;valid&=good;Z[~valid]=0
  features[:,:,t,:m]=Z;features[:,:,t,m:]=np.einsum('on,srn->sro',g['observable'],delta,optimize=True);features[~valid,t]=0;states[:,:,t]=Z
 return features,states,valid,failure,records

def baseline(rt,g,c):
 rng=np.random.default_rng(rt.seed+600000);Jz=g['J'][:,:8];mats=[];meta=[]
 for j in range(c['baseline_orientations']):
  R=orthogonal(rng,8,rt.b);M=R@Jz;radius=max(float(max(abs(np.linalg.eigvals(M)))),1e-12)
  for rho in c['spectral_radii']:mats.append(rho*R/radius);meta.append(dict(orientation=j,rho=rho))
 drive=rng.uniform(-1,1,c['calibration_steps']);features,states,valid,fail,_=execute(rt,g,np.stack(mats),drive,c,'baseline_acquisition',2)
 rows=[]
 for i,s in enumerate(states):
  a=s[0,c['burn']:].astype(float);b=s[1,c['burn']:].astype(float);var=np.mean(np.var(a,axis=0));echo=rms(a-b)/max(np.sqrt(var),1e-30);e=np.linalg.eigvalsh(np.cov(a,rowvar=False));rank=float(e.sum()**2/max(e@e,1e-30));ok=bool(valid[i].all() and var>1e-10 and echo<c['echo_tolerance'])
  rows.append(dict(slot=i,**meta[i],bounded=bool(valid[i].all()),variance=float(var),echo_ratio=echo,effective_rank=rank,admissible=ok))
 good=[r for r in rows if r['admissible']];selected=max(good,key=lambda r:(r['effective_rank'],-r['slot']))['slot'] if good else 0
 return mats[selected],dict(candidates=rows,selected=selected,source_admissible=bool(good)),dict(drive=drive,states=states,valid=valid,failure=fail)

def proposals(rt,g,C,c):
 b=C@g['J'][:,8];v=g['v'].copy();q=g['q'].copy()
 if v@b<0:v=-v
 probe=np.r_[.1*v,.0][None];even=rt.response(np.r_[probe,-probe])-g['base'];orientation=float(even.sum(0)@q)
 if orientation<0:q=-q
 drive=np.random.default_rng(rt.seed+700000).uniform(-1,1,c['calibration_steps']);f,z,valid,fail,records=execute(rt,g,C[None],drive,c,'return_calibration',checks=8)
 # The response signal is reconstructed from the source replay, not task labels.
 old=rt.stage;rt.stage='return_signal_calibration';signal=[]
 for t,u in enumerate(drive):
  prior=np.zeros(8) if t==0 else z[0,0,t-1];s=np.r_[prior,c['input_gain']*u][None];signal.append(float((rt.response(s)[0]-g['base'])@q))
 rt.stage=old;signal=np.array(signal);parent=rms(z[0,0,c['burn']:]);sig=rms(signal[c['burn']:]);ready=bool(valid.all() and parent>1e-10 and sig>1e-12);mats=[C.copy()];meta=[dict(slot=0,fraction=0.,alpha=0.,first_order_leak=0.)]
 for fraction in c['signed_action_fractions']:
  alpha=fraction*parent/max(sig*rms(v),1e-30) if ready else 0.;delta=alpha*np.outer(v,q);leak=float(np.linalg.norm(delta@g['J'])/max(np.linalg.norm(C@g['J']),1e-30));mats.append(C+delta);meta.append(dict(slot=len(mats)-1,fraction=fraction,alpha=alpha,first_order_leak=leak))
 return np.stack(mats),dict(candidates=meta,parent_rms=parent,signal_rms=sig,ready=ready,v=v.tolist(),q=q.tolist()),dict(drive=drive,signal=signal,states=z)

def fit(rt,g,Cs,c):
 T=c['burn']+c['training_steps'];drive=np.random.default_rng(c['training_seed']).uniform(-1,1,T);F,Z,V,fail,checks=execute(rt,g,Cs,drive,c,'source_programs_and_compatibility',checks=8);Y,desc=capacity_targets(drive,c['burn'],c['training_steps']);X=F[:,0,c['burn']:].astype(float)
 heads=[fit_head(A,Y,c['readout_ridge']) for A in X];maps=[fit_interface(X[0],A,c['transport_ridge']) for A in X]
 return heads,maps,desc,dict(drive=drive,features=F,states=Z,valid=V,failure=fail),checks

def evaluate(rt,g,Cs,heads,maps,c,seed,phase,desc):
 drive=np.random.default_rng(seed).uniform(-1,1,c['burn']+c['assessment_steps']);F,Z,V,fail,checks=execute(rt,g,Cs,drive,c,phase,checks=8);Y,_=capacity_targets(drive,c['burn'],c['assessment_steps']);rows=[]
 for i in range(len(Cs)):
  X=F[i,0,c['burn']:].astype(float);r=rr(head(X,heads[i]),Y);old=rr(apply_interface(X,maps[i])@heads[0]['readout']+heads[0]['target_mean'],Y);direct=rr(head(X,heads[0]),Y)
  if not V[i,0]:r[:]=-1;old[:]=-1;direct[:]=-1
  rows.append(dict(slot=i,bounded=bool(V[i,0]),**metrics(r,desc),old_r2=old.tolist(),direct_old_r2=direct.tolist()))
 return rows,dict(drive=drive,features=F,states=Z,valid=V,failure=fail),checks

def select(blocks,c):
 base=[b[0] for b in blocks];old=np.min([r['r2'] for r in base],0)>=c['reliability_threshold'];candidates=[]
 for j in range(1,len(blocks[0])):
  rr0=[b[j] for b in blocks];gains=[(r['quality']-b['quality'])/max(b['quality'],.25) for r,b in zip(rr0,base)];kept=old&(np.min([r['old_r2'] for r in rr0],0)>=c['reliability_threshold']);ok=bool(old.sum()>0 and all(r['bounded'] for r in base+rr0) and min(gains)>=c['quality_acceptance'] and kept.sum()==old.sum());candidates.append(dict(slot=j,gains=gains,old=int(old.sum()),retained=int(kept.sum()),admissible=ok))
 good=[r for r in candidates if r['admissible']];best=max(good,key=lambda r:(min(r['gains']),-r['slot']))['slot'] if good else 0
 return dict(selected_slot=best,old_program_ids=np.flatnonzero(old).tolist(),feedback_candidates=candidates)

def outcome(choice,blocks,c):
 j=choice['selected_slot'];old=np.array(choice['old_program_ids'],int);gains=[];ret=[];direct=[];baseline_ret=[]
 for block in blocks:
  base=block[0];r=block[j];gains.append((r['quality']-base['quality'])/max(base['quality'],.25));ret.append(int(np.sum(np.array(r['old_r2'])[old]>=c['reliability_threshold'])));direct.append(int(np.sum(np.array(r['direct_old_r2'])[old]>=c['reliability_threshold'])));baseline_ret.append(int(np.sum(np.array(base['r2'])[old]>=c['reliability_threshold'])))
 return dict(selected_slot=j,update=bool(j),relative_gains=gains,old_programs=len(old),old_retained_both=min(ret),old_retained_per_block=ret,direct_retained_both=min(direct),baseline_retained_both=min(baseline_ret),strict_joint=bool(len(old)>0 and min(gains)>=c['quality_acceptance'] and min(ret)==len(old) and all(b[j]['bounded'] and b[0]['bounded'] for b in blocks)),positive_both_full_retention=bool(len(old)>0 and min(gains)>0 and min(ret)==len(old)),quality_changes=[b[j]['quality']-b[0]['quality'] for b in blocks],baseline_quality=[b[0]['quality'] for b in blocks],selected_quality=[b[j]['quality'] for b in blocks])
