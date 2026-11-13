"""Bounded recruitment, composed raw-affine compatibility and protected programs."""
import numpy as np
from recruit import payload,rms,affine_interface
from legacy_engine import fit_head,head,rr
from capacity_engine import capacity_targets

def targets(drive,start,n):return capacity_targets(drive,start,n)
def mask(desc):return np.array([d['degree']>=2 and d['distinct_lags']>=2 for d in desc])
def quality(r,multi):return float(np.maximum(r[multi],0).sum())
def canonical(F,state):return F@state['A']+state['b']
def library_prediction(X,lib):return head(X@lib['A']+lib['b'],lib['head'])
def execute(rt,arc,g,C,states,drive,c,phase,checks=0):
 rt.stage=phase;S=len(states);T=len(drive);Z=np.zeros((S,8));valid=np.ones(S,bool);first_failure=np.full(S,-1,int);F=np.zeros((S,T,24),np.float32);trace=np.zeros((S,T,8),np.float32);records=[];indices=set(np.linspace(0,T-1,checks,dtype=int)) if checks else set()
 for t,u in enumerate(drive):
  s=np.column_stack([Z,np.full(S,c['input_gain']*u)])
  if t in indices:y,rec=rt.response(s,True);rec['time']=t;rec['previous_state']=Z.copy();records.append(rec)
  else:y=rt.response(s)
  dy=y-g['base'];Z=dy@C.T
  for j,state in enumerate(states):Z[j]+=arc.edge_update(state['edges'],s[j])
  good=np.isfinite(Z).all(1)&(np.max(abs(Z),axis=1)<=c['state_limit']);first_failure[(~good)&valid]=t;valid&=good;Z[~valid]=0
  F[:,t,:8]=Z;F[:,t,8:]=dy@g['observable'].T;F[~valid,t]=0;trace[:,t]=Z
 return dict(drive=drive,features=F,states=trace,valid=valid,first_failure=first_failure),records

def source_gradient(src,state,current_head,c,multi):
 X=canonical(src['features'][0,c['burn']:].astype(float),state);Y,_=targets(src['drive'],c['burn'],c['training_steps']);pred=head(X,current_head);res=(Y-pred)[:,multi];scale=np.maximum(Y[:,multi].std(0),1e-5);B=(current_head['readout']/current_head['feature_scale'][:,None])[:,multi]/scale[None];G=state['A'][:8]@B;grad=(res/scale[None])@G.T
 prior=np.vstack([np.zeros((1,8)),src['states'][0,:-1].astype(float)]);s=np.column_stack([prior,c['input_gain']*src['drive']])[c['burn']:]
 return s,grad

def build_candidates(arc,state,ids,s,src,c,seed,folder):
 base=state['edges'];evict=None
 if len(base)>=c['retained_recruits']:
  utility=[rms(np.array([arc.predict_edge(e,x) for x in s])*e['alpha'])*rms(e['b']) for e in base];evict=int(np.argmin(utility))
 candidates=[dict(edges=list(base),A=state['A'].copy(),b=state['b'].copy())];meta=[dict(slot=0,kind='baseline',port=None,evicted=None,fraction=0.)];calibration=[];parent_rms=rms(src['states'][0,c['burn']:])
 for index in ids:
  e,data=arc.acquire_port(index,s,seed+index+100000);np.savez_compressed(folder/f'port-{index}-source.npz',**data);r=arc.residual(arc.get(index),s)[:,0];power=rms(r)*rms(e['b']);ready=power>c['signal_floor'];calibration.append(dict(port=index,signal_rms=rms(r),direction_rms=rms(e['b']),parent_state_rms=parent_rms,profile_relative_error=float(e['profile_relative_error'])))
  for fraction in c['signed_action_fractions']:
   new={**e,'alpha':fraction*parent_rms/max(power,1e-30) if ready else 0.};edges=[b for j,b in enumerate(base) if j!=evict]+[new];assert len(edges)<=c['retained_recruits'];candidates.append(dict(edges=edges,A=state['A'].copy(),b=state['b'].copy()));meta.append(dict(slot=len(candidates)-1,kind='recruitment',port=index,evicted=None if evict is None else int(base[evict]['id']),fraction=fraction))
 return candidates,meta,calibration

def fit_candidates(src,candidates,parent,c):
 raw=src['features'][:len(candidates),c['burn']:].astype(float);Y,desc=targets(src['drive'],c['burn'],c['training_steps']);heads=[]
 for j,state in enumerate(candidates):
  if j:
   A,b=affine_interface(raw[j],raw[0],c['transport_ridge']);state['A']=A@parent['A'];state['b']=b@parent['A']+parent['b']
  X=canonical(raw[j],state);heads.append(fit_head(X,Y,c['readout_ridge']))
 return heads,desc

def evaluate(traces,states,heads,libraries,c,multi,reference_index):
 T=len(traces['drive'])-c['burn'];Y,desc=targets(traces['drive'],c['burn'],T);results=[];Xs=[canonical(traces['features'][j,c['burn']:].astype(float),st) for j,st in enumerate(states)]
 reference_scores=[rr(library_prediction(Xs[reference_index],lib),Y) for lib in libraries]
 for j,state in enumerate(states):
  X=Xs[j];r=rr(head(X,heads[min(j,len(heads)-1)]),Y) if j<len(heads) else np.full(Y.shape[1],-1.);libs=[rr(library_prediction(X,lib),Y) for lib in libraries]
  if not traces['valid'][j]:r[:]=-1;libs=[np.full_like(v,-1) for v in libs]
  results.append(dict(slot=j,bounded=bool(traces['valid'][j]),quality=quality(r,multi),r2=r.tolist(),program_r2=[v.tolist() for v in libs],program_quality=[quality(v,multi) for v in libs],anchor_program_quality=[quality(v,multi) for v in reference_scores]))
 return results

def contract(row,base,libs,c):
 retained=0;total=0;quality_ok=True
 for k,lib in enumerate(libs):
  ids=lib['ids'];arr=np.array(row['program_r2'][k]);retained+=int(np.sum(arr[ids]>=c['reliability_threshold']));total+=len(ids)
  floor=base['program_quality'][k]-c['preservation_tolerance']*max(base['program_quality'][k],.25)
  quality_ok &= row['program_quality'][k]>=floor
 # Root-current library preserves the original accepted improvement against a fixed anchor.
 floor=row['anchor_program_quality'][1]-c['preservation_tolerance']*max(row['anchor_program_quality'][1],.25)
 quality_ok &= row['program_quality'][1]>=floor
 return dict(retained=retained,total=total,quality_ok=bool(quality_ok),programs_ok=retained==total)

def select(blocks,libs,c):
 candidates=[]
 for j in range(1,len(blocks[0])-1):
  gains=[(b[j]['quality']-b[0]['quality'])/max(b[0]['quality'],.25) for b in blocks];checks=[contract(b[j],b[0],libs,c) for b in blocks];ok=all(b[j]['bounded'] and b[0]['bounded'] for b in blocks) and min(gains)>=c['quality_acceptance'] and all(v['programs_ok'] and v['quality_ok'] for v in checks)
  candidates.append(dict(slot=j,gains=gains,contracts=checks,admissible=bool(ok)))
 good=[r for r in candidates if r['admissible']];slot=max(good,key=lambda r:(min(r['gains']),-r['slot']))['slot'] if good else 0
 return dict(selected_slot=slot,candidates=candidates)

def outcome(slot,blocks,libs,c):
 gains=[(b[slot]['quality']-b[0]['quality'])/max(b[0]['quality'],.25) for b in blocks];checks=[contract(b[slot],b[0],libs,c) for b in blocks];strict=bool(slot and min(gains)>=c['quality_acceptance'] and all(q['programs_ok'] and q['quality_ok'] for q in checks) and all(b[slot]['bounded'] and b[0]['bounded'] for b in blocks));return dict(selected_slot=slot,relative_gains=gains,quality_changes=[b[slot]['quality']-b[0]['quality'] for b in blocks],strict_fresh_gain=strict,contracts=checks)
