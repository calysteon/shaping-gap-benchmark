"""Independent programs/contract/selection replay, including rollback and terminal."""
from pathlib import Path
import json,hashlib,time,sys
import numpy as np
from independent_targets import targets
HERE=Path(__file__).resolve().parent;LAB=HERE.parent
CHECKPOINT=Path(__file__).resolve().parents[2];DATA=CHECKPOINT/'data/experiments';PREVIOUS=CHECKPOINT.parent/'05-component-learning-cycle';OUTPUT=CHECKPOINT/'runs/replay';OUTPUT.mkdir(parents=True,exist_ok=True)


def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pred(X,h):return ((X-h['feature_mean'])/h['feature_scale'])@h['readout']+h['target_mean']
def rr(P,Y):return 1-np.sum((P-Y)**2,0)/np.maximum(np.sum((Y-Y.mean(0))**2,0),1e-30)
def quality(r,multi):return float(np.maximum(r[multi],0).sum())
def main():
 root=Path(sys.argv[1]) if len(sys.argv)>1 else DATA/'p65-program-contract-interfaces-001';summary=json.loads((root/'summary.json').read_text());c=summary['config'];worst=0.;vectors=0;seals=0;contexts=0;rounds=0;started=time.perf_counter()
 for ctx in summary['results']:
  d=root/'contexts'/ctx['context'];parent=PREVIOUS/'data/experiment/contexts'/f"{ctx['model']}-{ctx['seed']}-integrated";desc=json.loads((parent/'targets.json').read_text());multi=np.array([r['degree']>=2 and r['distinct_lags']>=2 for r in desc]);libraries=[]
  for folder in sorted(d.glob('library-*'),key=lambda p:int(p.name.split('-')[-1])):
   libraries.append(dict(head=load(folder/'head.npz'),map=load(folder/'input-map.npz'),ids=json.loads((folder/'program-ids.json').read_text())))
  active_libraries=2;accepted_maps=[];committed=None
  for step,reported in enumerate(ctx['rounds']):
   rd=d/f'round-{step}';seal=json.loads((rd/'construction-seal.json').read_text());N=len(seal['heads']);maps=[load(rd/f'candidate-{j}/map.npz') for j in range(N)]+[dict(A=np.eye(24),b=np.zeros(24))];heads=[load(rd/f'head-{j}.npz') for j in range(N)]
   for j,h in seal['heads'].items():assert sha(rd/f'head-{j}.npz')==h;seals+=1
   for j,h in seal['prior_library_hashes'].items():assert sha(d/f'library-{j}/head.npz')==h;seals+=1
   pickseal=json.loads((rd/'selection-seal.json').read_text());assert sha(rd/'selection.json')==pickseal['sha256'];assert sha(rd/'construction-seal.json')==pickseal['construction_sha256'];seals+=2
   # Independently refit each assessment head from immutable source features.
   tr=load(rd/'source.npz');Y=targets(tr['drive'],c['burn'],c['training_steps'],desc)
   for j,h in enumerate(heads):
    X=tr['features'][j,c['burn']:].astype(float)@maps[j]['A']+maps[j]['b'];A=(X-h['feature_mean'])/h['feature_scale'];coef=np.linalg.solve(A.T@A+c['readout_ridge']*len(A)*np.eye(24),A.T@(Y-h['target_mean']));assert np.allclose(coef,h['readout'],rtol=1e-8,atol=1e-9)
   feedback=[];fresh=[]
   for group,dest in [('feedback',feedback),('fresh',fresh)]:
    saved=json.loads((rd/f'{group}-scores.json').read_text())
    for b in range(2):
     tr=load(rd/f'{group}-{b}.npz');Y=targets(tr['drive'],c['burn'],c['assessment_steps'],desc);scores=[]
     for j,mp in enumerate(maps):
      X=tr['features'][j,c['burn']:].astype(float)@mp['A']+mp['b'];r=rr(pred(X,heads[j]),Y) if j<N else np.full(143,-1.);old=[]
      for lib in libraries[:active_libraries]:old.append(rr(pred(X@lib['map']['A']+lib['map']['b'],lib['head']),Y))
      if not tr['valid'][j]:r[:]=-1;old=[np.full(143,-1.) for _ in old]
      for v,stored in [(r,saved[b][j]['r2'])]+list(zip(old,saved[b][j]['program_r2'])):worst=max(worst,float(np.max(abs(v-stored))));vectors+=1
      scores.append(dict(quality=quality(r,multi),r=r,old=old,bounded=bool(tr['valid'][j])))
     dest.append(scores)
   def contract(row,base,ref):
    okay=all(np.all(row['old'][k][lib['ids']]>=.5) and quality(row['old'][k],multi)>=quality(base['old'][k],multi)-c['preservation_tolerance']*max(quality(base['old'][k],multi),.25) for k,lib in enumerate(libraries[:active_libraries]));okay &= quality(row['old'][1],multi)>=quality(ref['old'][1],multi)-c['preservation_tolerance']*max(quality(ref['old'][1],multi),.25);return bool(okay)
   eligible=[]
   for j in range(1,N):
    gains=[(b[j]['quality']-b[0]['quality'])/max(b[0]['quality'],.25) for b in feedback];valid=min(gains)>=c['quality_acceptance'] and all(b[j]['bounded'] and b[0]['bounded'] and contract(b[j],b[0],b[-1]) for b in feedback)
    if valid:eligible.append((min(gains),-j,j))
   j=max(eligible)[2] if eligible else 0;assert j==json.loads((rd/'selection.json').read_text())['selected_slot']==reported['selected_slot']
   gains=[(b[j]['quality']-b[0]['quality'])/max(b[0]['quality'],.25) for b in fresh];accepted=bool(j and min(gains)>=c['quality_acceptance'] and all(b[j]['bounded'] and b[0]['bounded'] and contract(b[j],b[0],b[-1]) for b in fresh));assert accepted==reported['accepted'];assert np.allclose(gains,reported['relative_gains'],atol=1e-8)
   if accepted:active_libraries+=1;accepted_maps.append(maps[j]);committed=maps[j]
   assert reported['shared_map_values']==24*24+24;assert reported['retained_payload_bytes']<=c['retained_payload_cap'];assert len(reported['retained_port_ids'])<=c['retained_recruits'];rounds+=1
  assert len(accepted_maps)==ctx['accepted_acquisitions'];td=d/'terminal';final=load(td/'final-state/map.npz');maps=[dict(A=np.eye(24),b=np.zeros(24)),final]+accepted_maps;heads=[load(td/f'head-{j}.npz') for j in range(len(maps))];saved=json.loads((td/'scores.json').read_text());terminal=[]
  for b in range(2):
   tr=load(td/f'fresh-{b}.npz');Y=targets(tr['drive'],c['burn'],c['terminal_steps'],desc);rows=[]
   for j,mp in enumerate(maps):
    X=tr['features'][j,c['burn']:].astype(float)@mp['A']+mp['b'];r=rr(pred(X,heads[j]),Y);old=[rr(pred(X@lib['map']['A']+lib['map']['b'],lib['head']),Y) for lib in libraries]
    if not tr['valid'][j]:r[:]=-1;old=[np.full(143,-1.) for _ in old]
    for v,v0 in [(r,saved[b][j]['r2'])]+list(zip(old,saved[b][j]['program_r2'])):worst=max(worst,float(np.max(abs(v-v0))));vectors+=1
    rows.append(dict(quality=quality(r,multi),old=old,r=r,bounded=bool(tr['valid'][j])))
   terminal.append(rows)
  gains=[(b[1]['quality']-b[0]['quality'])/max(b[0]['quality'],.25) for b in terminal];assert np.allclose(gains,ctx['terminal_relative_gains'],atol=1e-8);contexts+=1
 assert worst<1e-7
 result=dict(passed=True,contexts=contexts,rounds=rounds,score_vectors=vectors,maximum_score_error=worst,seals_checked=seals,all_source_heads_refitted=True,all_feedback_choices_replayed=True,all_acceptance_and_rollback_decisions_replayed=True,terminal_gains_replayed=True,retained_slots_and_shared_map_shape_verified=True,wall_seconds=time.perf_counter()-started)
 (OUTPUT/'p65-independent.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
