"""One shared compatibility map fitted against retained program contracts.

This explicitly uses self-supervised functional target values on source data.
Old programs and pretrained neural components remain unchanged. Linear algebra,
program-aware loss and regularization selection are supplied construction tools.
"""
import numpy as np
from cycle import execute as physical_execute,canonical
from legacy_engine import fit_head
from capacity_engine import capacity_targets
from policy import fit_interface

def kernel_key(st):return tuple((e['id'],float(e['alpha'])) for e in st['edges'])
def execute(rt,arc,g,C,states,drive,c,phase,checks=0):
 keys={};unique=[];index=[]
 for st in states:
  key=kernel_key(st)
  if key not in keys:keys[key]=len(unique);unique.append(st)
  index.append(keys[key])
 tr,rec=physical_execute(rt,arc,g,C,unique,drive,c,phase,checks)
 index=np.array(index,int);tr={**tr,'features':tr['features'][index],'states':tr['states'][index],'valid':tr['valid'][index],'first_failure':tr['first_failure'][index],'physical_view_index':index}
 for row in rec:row['physical_view_index']=index
 return tr,rec

def contract_map(raw,canonical_parent,Y,libraries,feature_weight,ridge):
 nm=raw.mean(0);ns=np.maximum(raw.std(0),1e-8);bm=canonical_parent.mean(0);bs=np.maximum(canonical_parent.std(0),1e-8);X=np.column_stack([(raw-nm)/ns,np.ones(len(raw))]);Z=(canonical_parent-bm)/bs;M=np.eye(24)*feature_weight;R=feature_weight*(X.T@Z);target_scale=np.maximum(Y.std(0),1e-5)
 for lib in libraries:
  h=lib['head'];W=bs[:,None]*(lib['A']@(h['readout']/h['feature_scale'][:,None]));bias=((bm@lib['A']+lib['b']-h['feature_mean'])/h['feature_scale'])@h['readout']+h['target_mean'];weights=np.ones(Y.shape[1]);weights[lib['ids']]=4.;weights=weights/target_scale**2/len(libraries);weighted=W*weights[None];M+=weighted@W.T;R+=X.T@(Y-bias)@weighted.T
 gram=X.T@X;prior=np.vstack([np.eye(24),np.zeros((1,24))]);lam=ridge*len(X);R+=lam*prior;ev,V=np.linalg.eigh(M);ev=np.maximum(ev,0);RV=R@V;P=np.column_stack([np.linalg.solve(ev[j]*gram+lam*np.eye(25),RV[:,j]) for j in range(24)])@V.T
 A=P[:24]*bs[None]/ns[:,None];b=bm+P[24]*bs-nm@A
 return A,b,dict(feature_weight=feature_weight,ridge=ridge,minimum_output_weight_eigenvalue=float(ev.min()),maximum_output_weight_eigenvalue=float(ev.max()))

def views(physical,raw_source,parent,libraries,c,arm):
 Y,desc=capacity_targets(raw_source['drive'],c['burn'],c['training_steps']);raw=raw_source['features'][:len(physical),c['burn']:].astype(float);target=canonical(raw[0],parent);states=[physical[0]];heads=[fit_head(target,Y,c['readout_ridge'])];meta=[dict(physical_slot=0,interface='unchanged')]
 phys_indices=[0] if arm=='interface_only' else list(range(len(physical)))
 for slot in phys_indices:
  if slot:
   m=fit_interface(raw[0],raw[slot],c['transport_ridge']);A=m['transport']*m['old_feature_scale'][None]/m['new_feature_scale'][:,None];b=m['old_feature_mean']-m['new_feature_mean']@A;st=dict(edges=physical[slot]['edges'],A=A@parent['A'],b=b@parent['A']+parent['b']);states.append(st);heads.append(fit_head(canonical(raw[slot],st),Y,c['readout_ridge']));meta.append(dict(physical_slot=slot,interface='feature',ridge=c['transport_ridge']))
  if arm!='feature':
   for fw in c['contract_feature_weights']:
    for ridge in c['contract_ridges']:
     A,b,diagnostic=contract_map(raw[slot],target,Y,libraries,fw,ridge);st=dict(edges=physical[slot]['edges'],A=A,b=b);states.append(st);heads.append(fit_head(canonical(raw[slot],st),Y,c['readout_ridge']));meta.append(dict(physical_slot=slot,interface='program_contract',**diagnostic))
 return states,heads,meta,desc
