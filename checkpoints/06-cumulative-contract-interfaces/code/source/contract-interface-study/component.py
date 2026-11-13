"""A transformer-free runtime over extracted, immutable pretrained fragments.

Only array arithmetic is imported here. Component responses preserve the original
SwiGLU/norm formula with float32-materialized pretrained rows; the original layer
schedule, attention, tokens and generation head have no execution path.
"""
import hashlib
import numpy as np
from interfaces import QuadraticInterface

def digest(arrays):
 h=hashlib.sha256()
 for key,a in sorted(arrays.items()):
  a=np.ascontiguousarray(a);h.update(key.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
 return h.hexdigest()
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
def orthogonal(rng,rows,width):return np.linalg.qr(rng.normal(size=(width,rows)))[0].T

def scrambled(bank,rng):
 out={k:v.copy() for k,v in bank.items()}
 for key in ['G','U','D']:
  x=rng.normal(size=out[key].shape);x*=np.linalg.norm(out[key],axis=1)[:,None]/np.maximum(np.linalg.norm(x,axis=1)[:,None],1e-30);out[key]=x
 return out

class ComponentRuntime:
 def __init__(self,bank,seed,c,random_weights=False):
  self.c=c;self.seed=seed;self.bank={k:v.copy() for k,v in bank.items()};self.before=digest(self.bank)
  rng=np.random.default_rng(seed+750000)
  if random_weights:self.bank=scrambled(self.bank,np.random.default_rng(seed+970000));self.before=digest(self.bank)
  h=self.bank['G'].shape[1];b=len(self.bank['G']);self.h=h;self.b=b
  self.anchor=rng.normal(size=h);self.anchor*=c['anchor_rms']/(rms(self.anchor))
  self.Ux=orthogonal(rng,9,h)
  self.output_basis=np.linalg.qr(self.bank['D'].T)[0]
  self.Dout=self.output_basis.T@self.bank['D'].T
  self.B=np.linalg.qr(np.column_stack([self.Ux.T,self.bank['D'].T]))[0]
  self.V=self.B.T@(c['epsilon']*self.Ux.T);self.W=self.B.T@self.bank['D'].T;self.d=self.B.shape[1]
  self.G=self.bank['G']@self.B;self.Up=self.bank['U']@self.B
  self.g0=self.bank['G']@self.anchor;self.u0=self.bank['U']@self.anchor
  self.norm0=float(self.anchor@self.anchor/h);self.norm1=self.B.T@self.anchor/h;self.norm2=self.B.T@self.B/h
  self.mode='integrated';self.stage='setup';self.counts={};self.route_counts=np.zeros((2,b),int);self.route_switches=0;self.last_route=None
  self.permutation=np.random.default_rng(seed+860000).permutation(b)
 def meter(self,kind,count):
  key=self.stage+':'+kind;self.counts[key]=self.counts.get(key,0)+int(count)
 def native(self,v,gradient=False):
  """Exact fragment formula on the retained affine coordinate subspace."""
  v=np.atleast_2d(np.asarray(v,float));r=np.sqrt(np.maximum(self.norm0+2*v@self.norm1+np.einsum('ni,ij,nj->n',v,self.norm2,v)+self.bank['eps'][0],1e-20))
  g=(self.g0+v@self.G.T)/r[:,None];up=(self.u0+v@self.Up.T)/r[:,None];sig=1/(1+np.exp(np.clip(-g,-60,60)));si=g*sig;a=si*up
  self.meter('native_response_candidates',len(v)*self.b)
  if not gradient:return a
  dr=(self.norm1[None]+v@self.norm2)/r[:,None]
  dg=(self.G[None]-g[:,:,None]*dr[:,None])/r[:,None,None]
  du=(self.Up[None]-up[:,:,None]*dr[:,None])/r[:,None,None]
  grad=(sig+g*sig*(1-sig))[:,:,None]*up[:,:,None]*dg+si[:,:,None]*du
  self.meter('native_response_partials',len(v)*self.b*self.d)
  return a,grad.transpose(0,2,1)
 def native_selected(self,v,indices):
  v=np.atleast_2d(v);r=np.sqrt(np.maximum(self.norm0+2*v@self.norm1+np.einsum('ni,ij,nj->n',v,self.norm2,v)+self.bank['eps'][0],1e-20))
  g=(self.g0[indices]+np.einsum('nkd,nd->nk',self.G[indices],v))/r[:,None]
  up=(self.u0[indices]+np.einsum('nkd,nd->nk',self.Up[indices],v))/r[:,None]
  self.meter('native_selected_responses',indices.size)
  return g/(1+np.exp(np.clip(-g,-60,60)))*up
 def acquire(self):
  self.stage='interface_acquisition';rng=np.random.default_rng(self.seed+400000);v=rng.uniform(-self.c['profile_radius'],self.c['profile_radius'],(self.c['profile_samples'],self.d))
  a,g=self.native(v,True);data=dict(z=v,y=a[:,:,None],g=g[:,:,:,None])
  self.profile=QuadraticInterface.fit(data,self.c['profile_radius'],True,self.c['profile_ridge'])
  self.stage='interface_diagnostic';vv=rng.uniform(-self.c['profile_radius'],self.c['profile_radius'],(self.c['profile_validation'],self.d));truth,tg=self.native(vv,True)
  pred,j=self.predict(vv,True)
  diagnostics=dict(value_relative_rmse=rms(pred-truth)/max(rms(truth),1e-30),partial_relative_rmse=rms(j-tg)/max(rms(tg),1e-30),profile_bytes=self.profile.payload(),source_points=len(v),validation_points=len(vv))
  return data,dict(z=vv,y=truth,g=tg,prediction=pred,partial_prediction=j),diagnostics
 def predict(self,v,gradient=False):
  v=np.atleast_2d(v);p=self.profile;c=p.c;L=p.L;H=p.H
  # Same integrable retained quadratic used for both effects and partials.
  a=c[None]+v@L.T+.5*np.einsum('ni,oij,nj->no',v,H,v,optimize=True)
  self.meter('interface_predictions',len(v)*self.b)
  if gradient:return a,L.T[None]+np.einsum('oij,nj->nio',H,v,optimize=True)
  return a
 def response(self,s,record=False):
  """Two intentional fragment stages; selected effects feed live successor input."""
  s=np.atleast_2d(s);v=s@self.V.T;pred,grad=self.predict(v,True);N=len(v);k=self.c['active_fragments_per_stage'];norm=np.linalg.norm(self.bank['D'],axis=1)
  first=np.argsort(-np.abs(pred)*norm[None],axis=1,kind='stable')[:,:k]
  rows=np.arange(N)[:,None];aa=self.native_selected(v,first)
  write=np.einsum('nkj,nk->nj',self.W.T[first],aa);live=v+write
  if self.mode=='integrated':after=self.predict(live);conditioning=after-pred
  elif self.mode=='first_order':conditioning=np.einsum('ndo,nd->no',grad,write)
  elif self.mode=='shuffled':
   after=self.predict(live);conditioning=(after-pred)[:,self.permutation]
  else:raise ValueError(self.mode)
  scores=np.abs(conditioning)*norm[None];scores[rows,first]=-np.inf
  second=np.argsort(-scores,axis=1,kind='stable')[:,:k]
  ab=self.native_selected(live,second)
  coefficients=np.zeros((N,self.b));coefficients[rows,first]=aa;coefficients[rows,second]=ab
  self.meter('committed_fragments',2*N*k);self.meter('routing_comparisons',N*self.b*2)
  np.add.at(self.route_counts[0],first.ravel(),1);np.add.at(self.route_counts[1],second.ravel(),1)
  if self.last_route is not None and self.last_route.shape==second.shape:self.route_switches+=int(np.sum(np.any(second!=self.last_route,axis=1)))
  self.last_route=second.copy();y=coefficients@self.Dout.T
  if record:return y,dict(input_s=s,coordinate_v=v,first=first,second=second,first_amplitudes=aa,second_amplitudes=ab,output=y)
  return y
 def geometry(self):
  self.stage='return_geometry';c=self.c;rng=np.random.default_rng(self.seed+500000);d=c['differential_step'];eye=np.eye(9)
  base=self.response(np.zeros((1,9)))[0];Y=self.response(np.r_[d*eye,-d*eye]);J=((Y[:9]-Y[9:])/(2*d)).T
  S=rng.normal(size=(c['curvature_samples'],9))*c['curvature_probe_scale']/3;Y=self.response(S);residual=Y-base-S@J.T
  pairs=[(i,j) for i in range(9) for j in range(i,9)];X=np.column_stack([S[:,i]*S[:,j]*(1 if i==j else 2) for i,j in pairs]);Q=np.linalg.lstsq(X,residual,rcond=1e-9)[0]
  u,s,_=np.linalg.svd(J,full_matrices=False);rank=int(sum(s>s[0]*1e-5));space=u[:,:rank];curv=Q-(Q@space)@space.T;_,sv,V=np.linalg.svd(curv,full_matrices=False);q=V[0];q-=space@(space.T@q);q/=max(np.linalg.norm(q),1e-30)
  H=np.zeros((9,9))
  for row,(i,j) in enumerate(pairs):H[i,j]=H[j,i]=Q[row]@q
  e,w=np.linalg.eigh(H[:8,:8]);direction=w[:,np.argmax(abs(e))]
  R=V[:c['observable_dimension']]
  return dict(base=base,J=J,q=q,v=direction,observable=R,rank=rank,curvature_singular=sv,curvature_relative_error=rms(X@Q-residual)/max(rms(residual),1e-30))
 def save(self,path):
  np.savez_compressed(path,**{'bank_'+k:v for k,v in self.bank.items()},anchor=self.anchor,Ux=self.Ux,output_basis=self.output_basis,Dout=self.Dout,B=self.B,V=self.V,W=self.W,G=self.G,Up=self.Up,g0=self.g0,u0=self.u0,norm0=np.array(self.norm0),norm1=self.norm1,norm2=self.norm2,profile_c=self.profile.c,profile_L=self.profile.L,profile_H=self.profile.H,permutation=self.permutation)
