"""Streamed native-port recruitment and fixed-capacity local response interfaces."""
from pathlib import Path
import numpy as np
from component import ComponentRuntime,rms,digest
from interfaces import QuadraticInterface

def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}

def restore_runtime(parent,c):
 p=load(parent/'runtime.npz');rt=object.__new__(ComponentRuntime);rt.c=c;rt.seed=int(parent.name.split('-')[1]);rt.bank={k[5:]:v for k,v in p.items() if k.startswith('bank_')};rt.before=digest(rt.bank);rt.h=rt.bank['G'].shape[1];rt.b=len(rt.bank['G']);rt.d=p['B'].shape[1]
 for k in ['anchor','Ux','output_basis','Dout','B','V','W','G','Up','g0','u0','norm0','norm1','norm2','permutation']:setattr(rt,k,p[k])
 rt.profile=QuadraticInterface(p['profile_c'],p['profile_L'],p['profile_H'],(rt.b,1));rt.mode='integrated';rt.stage='setup';rt.counts={};rt.route_counts=np.zeros((2,rt.b),int);rt.route_switches=0;rt.last_route=None
 g=load(parent/'geometry.npz');slot=int(__import__('json').loads((parent/'selection.json').read_text())['selected_slot']);C=load(parent/'returns.npz')['returns'][slot]
 return rt,g,C,slot

class Archive:
 def __init__(self,folder,rt,C,c):
  self.folder=Path(folder);self.rt=rt;self.C=C;self.c=c;self.rows=0;self.queries=0;self.partial_queries=0;self.profile_count=0;self.screen_peak_bytes=0;self.bytes_read=0;self.current_chunk_bytes=0
  self.U=c['epsilon']*rt.Ux.T;self.n0=float(rt.anchor@rt.anchor/rt.h);self.n1=self.U.T@rt.anchor/rt.h;self.n2=self.U.T@self.U/rt.h
 def compile(self,raw,ids):
  self.rows+=len(ids);self.current_chunk_bytes=sum(v.nbytes for v in raw.values());self.bytes_read+=self.current_chunk_bytes;G=raw['G']@self.U;Up=raw['U']@self.U;g0=raw['G']@self.rt.anchor;u0=raw['U']@self.rt.anchor;b=raw['D']@self.rt.output_basis@self.C.T
  pack=dict(G=G,Up=Up,g0=g0,u0=u0,b=b,ids=np.array(ids),eps=raw['eps']);zero,grad=self.native(pack,np.zeros((1,9)),True);pack['a0']=zero[0];pack['grad0']=grad[0];return pack
 def native(self,p,s,grad=False):
  s=np.atleast_2d(s);r=np.sqrt(np.maximum(self.n0+2*s@self.n1+np.einsum('ni,ij,nj->n',s,self.n2,s)+p['eps'][0],1e-20));g=(p['g0']+s@p['G'].T)/r[:,None];u=(p['u0']+s@p['Up'].T)/r[:,None];sig=1/(1+np.exp(np.clip(-g,-60,60)));a=g*sig*u;self.queries+=a.size
  if not grad:return a
  dr=(self.n1[None]+s@self.n2)/r[:,None];dg=(p['G'][None]-g[:,:,None]*dr[:,None])/r[:,None,None];du=(p['Up'][None]-u[:,:,None]*dr[:,None])/r[:,None,None];der=(sig+g*sig*(1-sig))[:,:,None]*u[:,:,None]*dg+(g*sig)[:,:,None]*du;self.partial_queries+=der.size;return a,der.transpose(0,2,1)
 def residual(self,p,s):return self.native(p,s)-p['a0']-s@p['grad0']
 def chunks(self,pool):
  if pool=='resident':
   for a in range(0,self.rt.b,self.c['screen_chunk']):
    idx=np.arange(a,min(a+self.c['screen_chunk'],self.rt.b));raw={k:v[idx] for k,v in self.rt.bank.items()};yield self.compile(raw,-idx-1)
  else:
   for a in range(0,int(pool),self.c['screen_chunk']):
    raw=load(self.folder/f'chunk-{a//self.c["screen_chunk"]:03d}.npz');n=min(len(raw['G']),int(pool)-a);raw={k:v[:n] for k,v in raw.items()};yield self.compile(raw,np.arange(a,a+n))
 def get(self,index):
  if index<0:idx=-index-1;raw={k:v[idx:idx+1] for k,v in self.rt.bank.items()}
  else:
   q,r=divmod(index,self.c['screen_chunk']);raw=load(self.folder/f'chunk-{q:03d}.npz');raw={k:v[r:r+1] for k,v in raw.items()}
  return self.compile(raw,[index])
 def select(self,s,gradient,edges,pool,random,seed):
  active={e['id'] for e in edges};records=[];best=[]
  if random:
   ids=np.setdiff1d(np.arange(int(pool)),np.array(list(active),int));chosen=np.random.default_rng(seed).choice(ids,self.c['proposed_ports'],replace=False)
   for index in chosen:
    p=self.get(int(index));r=self.residual(p,s);den=rms(r)*rms(p['b']);records.append(dict(index=int(index),score=None,signal_rms=rms(r),direction_rms=rms(p['b'])));best.append(int(index))
   return best,records
  for p in self.chunks(pool):
   r=self.residual(p,s);descent=gradient@p['b'].T;num=np.mean(r*descent,axis=0);den=np.mean(r*r,axis=0)*np.mean(p['b']*p['b'],axis=1);scores=num*num/np.maximum(den,1e-30);memory=self.current_chunk_bytes+r.nbytes+descent.nbytes+sum(v.nbytes for v in p.values());self.screen_peak_bytes=max(self.screen_peak_bytes,memory)
   for j,index in enumerate(p['ids']):
    valid=bool(index not in active and np.isfinite(scores[j]) and den[j]>self.c['signal_floor']);records.append(dict(index=int(index),score=float(scores[j]),signal_rms=rms(r[:,j]),direction_rms=rms(p['b'][j]),admissible=valid))
  good=sorted([r for r in records if r.get('admissible')],key=lambda r:(-r['score'],r['index']));return [r['index'] for r in good[:self.c['proposed_ports']]],records
 def acquire_port(self,index,s,seed):
  p=self.get(index);rng=np.random.default_rng(seed);mu=s.mean(0);sd=np.maximum(s.std(0),self.c['local_scale_floor']);idx=rng.choice(len(s),self.c['local_profile_samples'],replace=False);samples=s[idx]+rng.normal(size=(len(idx),9))*sd*self.c['local_jitter'];values,g=self.native(p,samples,True);w=(samples-mu)/sd;data=dict(z=w,y=values[:,:,None],g=(g*sd[None,:,None])[:,:,:,None]);model=QuadraticInterface.fit(data,1.,True,self.c['local_profile_ridge']);self.profile_count+=1
  edge={k:v[0].copy() if isinstance(v[0],np.ndarray) else np.array(v[0]) for k,v in p.items() if k not in ['ids','grad0']};edge['grad0']=p['grad0'][:,0].copy();edge.update(id=int(index),mu=mu,sd=sd,profile_c=model.c.copy(),profile_L=model.L[0].copy(),profile_H=model.H[0].copy(),alpha=0.)
  # Pack all primitive coordinates as9-vectors, with scalar constants.
  edge['G']=p['G'][0];edge['Up']=p['Up'][0];edge['g0']=p['g0'][0];edge['u0']=p['u0'][0];edge['a0']=p['a0'][0];edge['eps']=p['eps'][0];edge['b']=p['b'][0]
  pred=self.predict_edge(edge,samples);truth=values[:,0]-p['a0'][0]-samples@p['grad0'][:,0];edge['profile_relative_error']=rms(pred-truth)/max(rms(truth),1e-20)
  return edge,dict(samples=samples,values=values,partials=g,whitened=w)
 @staticmethod
 def predict_edge(e,s):
  w=(np.asarray(s)-e['mu'])/e['sd'];return e['profile_c'].item()+w@e['profile_L']+.5*np.einsum('...i,ij,...j->...',w,e['profile_H'],w)-e['a0']-np.asarray(s)@e['grad0']
 def edge_value(self,e,s):
  s=np.asarray(s);r=np.sqrt(max(self.n0+2*s@self.n1+s@self.n2@s+float(e['eps']),1e-20));g=(e['g0']+s@e['G'])/r;u=(e['u0']+s@e['Up'])/r;self.queries+=1;return g/(1+np.exp(np.clip(-g,-60,60)))*u-e['a0']-s@e['grad0']
 def edge_update(self,edges,s):
  if not edges:return np.zeros(8)
  rank=np.array([abs(self.predict_edge(e,s))*abs(e['alpha'])*rms(e['b']) for e in edges]);keep=np.argsort(-rank,kind='stable')[:self.c['executed_recruits']];return sum((edges[i]['alpha']*edges[i]['b']*self.edge_value(edges[i],s) for i in keep),start=np.zeros(8))
 def counters(self):return dict(compiled_port_rows=self.rows,native_scalar_responses=self.queries,native_scalar_partials=self.partial_queries,local_interfaces_fitted=self.profile_count,maximum_screen_workspace_bytes=self.screen_peak_bytes,source_array_bytes_read=self.bytes_read)

def affine_interface(X,Y,ridge):
 from policy import fit_interface
 m=fit_interface(Y,X,ridge);A=m['transport']*m['old_feature_scale'][None,:]/m['new_feature_scale'][:,None];b=m['old_feature_mean']-m['new_feature_mean']@A;return A,b

def save_state(folder,state):
 import json
 folder.mkdir(exist_ok=True,parents=True);np.savez_compressed(folder/'map.npz',A=state['A'],b=state['b']);meta=[]
 for i,e in enumerate(state['edges']):
  np.savez_compressed(folder/f'edge-{i}.npz',**{k:np.asarray(v) for k,v in e.items()});meta.append(dict(id=int(e['id']),alpha=float(e['alpha']),profile_relative_error=float(e['profile_relative_error'])))
 (folder/'edges.json').write_text(json.dumps(meta,indent=2)+'\n')

def payload(state):
 return state['A'].nbytes+state['b'].nbytes+sum(sum(np.asarray(v).nbytes for v in e.values()) for e in state['edges'])
