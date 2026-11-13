"""Prospective chronological application transfer with unchanged controllers."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib,time,resource,sys
import numpy as np
HERE=Path(__file__).resolve().parent;LAB=HERE.parent
sys.path.insert(0,str(LAB/'contract-interface-study'))
from recruit import restore_runtime,Archive,load
from legacy_engine import fit_head,head

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_load(p):
 m=load(p/'map.npz');edges=[]
 for f in sorted(p.glob('edge-*.npz')):
  e=load(f);e['id']=int(e['id']);e['alpha']=float(e['alpha']);edges.append(e)
 return dict(A=m['A'],b=m['b'],edges=edges)

def execute(rt,arc,g,systems,drive,Z,c,phase):
 rt.stage=phase;N=len(systems);F=np.zeros((N,len(drive),24),np.float32);valid=np.ones(N,bool);failure=np.full(N,-1,int)
 for t,u in enumerate(drive):
  s=np.column_stack([Z,np.full(N,c['input_gain']*u)]);Y=rt.response(s)-g['base'];Z=np.stack([systems[j]['C']@Y[j]+arc.edge_update(st['edges'],s[j]) for j,st in enumerate(systems)])
  good=np.isfinite(Z).all(1)&(np.max(abs(Z),axis=1)<=c['state_limit']);failure[(~good)&valid]=t;valid&=good;Z[~valid]=0;F[:,t,:8]=Z;F[:,t,8:]=Y@g['observable'].T;F[~valid,t]=0
 return F,Z,valid,failure

def error_record(pred,truth):
 e=pred-truth;return dict(mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e*e))),samples=len(e))
def bootstrap_delta(base,new,block,repeats,seed):
 # Paired moving blocks preserve local temporal dependence in error differences.
 n=len(base);starts=np.arange(0,n,block);blocks=[np.arange(i,min(i+block,n)) for i in starts];rng=np.random.default_rng(seed);vals=[]
 for _ in range(repeats):
  chosen=rng.integers(0,len(blocks),len(blocks));idx=np.concatenate([blocks[i] for i in chosen]);b=float(np.sqrt(np.mean(base[idx]**2)));v=float(np.sqrt(np.mean(new[idx]**2)));vals.append((b-v)/max(b,1e-20))
 return np.quantile(vals,[.025,.975]).tolist()

def main():
 plan=json.loads((HERE/'protocol.json').read_text());c=plan['config'];out=LAB/'results'/plan['run_id'];out.mkdir(exist_ok=False);started=time.perf_counter();phase='load';dump(out/'protocol.json',plan);__import__('shutil').copy2(__file__,out/'runner_snapshot.py');dump(out/'manifest.json',dict(plan_id=plan['id'],created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),config=c,data_sha256=sha(HERE/'data/energydata_complete.csv')))
 try:
  with (HERE/'data/energydata_complete.csv').open() as f:raw=list(csv.DictReader(f))
  y=np.array([float(r['Appliances']) for r in raw]);n=len(y);assert n==c['expected_rows'];train_end=int(c['train_fraction']*n);valid_end=int(c['validation_end_fraction']*n);log=np.log1p(y);median=float(np.median(log[:train_end]));sd=max(float(np.std(log[:train_end])),1e-8);drive=np.tanh((log-median)/(3*sd));dump(out/'scaler.json',dict(log1p=True,median=median,standard_deviation=sd,fitted_rows=[0,train_end],train_end=train_end,validation_end=valid_end,total_rows=n));np.savez_compressed(out/'input.npz',energy_wh=y,drive=drive)
  pc=json.loads((LAB/'component-cycle-study/protocol.json').read_text())['config'];parent=LAB/'results/p63-component-acquisition-reuse-001/contexts/0.6B-151003-integrated';pc={**pc,'anchor_rms':pc['anchor_rms_by_seed']['151003']};rt,g,C,slot=restore_runtime(parent,pc);arc=Archive(LAB/'cumulative-recruitment-study/pools/0.6B-151003',rt,C,c);p65=LAB/'results/p65-program-contract-interfaces-001/contexts';systems=[]
  C0=load(parent/'returns.npz')['returns'][0];empty=dict(A=np.eye(24),b=np.zeros(24),edges=[])
  systems.append(dict(name='checkpoint05_before_first_gain',C=C0,**empty));systems.append(dict(name='checkpoint05_accepted',C=C,**empty))
  for arm in ['feature','contract','interface_only']:
   folder=p65/f'0.6B-151003-{arm}/terminal/final-state';st=state_load(folder);systems.append(dict(name=f'p65_{arm}',C=C,**st));dump(out/f'controller-{arm}.json',dict(source=str(folder.relative_to(LAB)),map_sha256=sha(folder/'map.npz'),edges=[dict(id=e['id'],alpha=e['alpha']) for e in st['edges']]))
  names=[s['name'] for s in systems];Z=np.zeros((len(systems),8));a=time.perf_counter();F,Z,V,failure=execute(rt,arc,g,systems,drive[:train_end],Z,c,'application_training');train_seconds=time.perf_counter()-a;np.savez_compressed(out/'training-features.npz',features=F,valid=V,final_state=Z)
  maxh=max(c['horizons']);idx=np.arange(c['washout'],train_end-maxh);Y=np.column_stack([y[idx+h] for h in c['horizons']]);heads=[]
  for j,st in enumerate(systems):
   X=F[j,idx].astype(float)@st['A']+st['b'];h=fit_head(X,Y,c['readout_ridge']);heads.append(h);np.savez_compressed(out/f'head-{j}.npz',**h)
  # Strong classical reference:64 causal raw-energy lags, same source targets.
  AR=np.column_stack([y[idx-lag] for lag in range(c['ar_lags'])]);arhead=fit_head(AR,Y,c['readout_ridge']);np.savez_compressed(out/'ar-head.npz',**arhead)
  dump(out/'program-seal.json',dict(utc=datetime.now(timezone.utc).isoformat(),source_row_indices=[int(idx[0]),int(idx[-1])],heads={str(j):sha(out/f'head-{j}.npz') for j in range(len(heads))},ar_head_sha256=sha(out/'ar-head.npz'),test_outcomes_exposed=False,controller_selection='Contract arm fixed prospectively; no application-based controller choice.'))
  results=[];errors={};phase='validation_and_test';a=time.perf_counter();tail,Z,V2,failure2=execute(rt,arc,g,systems,drive[train_end:],Z,c,'application_validation_and_test');test_seconds=time.perf_counter()-a;np.savez_compressed(out/'later-features.npz',features=tail,valid=V2,final_state=Z)
  for split,lo,hi in [('validation',train_end,valid_end),('test',valid_end,n)]:
   t=np.arange(lo,hi-maxh);truth=np.column_stack([y[t+h] for h in c['horizons']]);predictions={}
   for j,st in enumerate(systems):
    X=tail[j,t-train_end].astype(float)@st['A']+st['b'];predictions[st['name']]=np.maximum(head(X,heads[j]),0.)
   X=np.column_stack([y[t-lag] for lag in range(c['ar_lags'])]);predictions['ar64']=np.maximum(head(X,arhead),0.);predictions['persistence']=np.repeat(y[t,None],len(c['horizons']),axis=1);predictions['daily_seasonal']=np.column_stack([y[t+h-c['day_steps']] for h in c['horizons']])
   np.savez_compressed(out/f'{split}-predictions.npz',time_index=t,truth=truth,**predictions)
   for name,P in predictions.items():
    for j,horizon in enumerate(c['horizons']):results.append(dict(split=split,model=name,horizon=horizon,forecast_minutes=10*horizon,units='Wh',**error_record(P[:,j],truth[:,j])))
   if split=='test':
    base=predictions['checkpoint05_accepted']-truth;new=predictions['p65_contract']-truth;errors={str(h):dict(relative_rmse_reduction=float((np.sqrt(np.mean(base[:,j]**2))-np.sqrt(np.mean(new[:,j]**2)))/max(np.sqrt(np.mean(base[:,j]**2)),1e-20)),paired_block_interval=bootstrap_delta(base[:,j],new[:,j],c['bootstrap_block'],c['bootstrap_repeats'],c['bootstrap_seed']+j)) for j,h in enumerate(c['horizons'])}
  primary=errors[str(c['primary_horizon'])];positive=primary['relative_rmse_reduction']>=c['quality_acceptance'] and primary['paired_block_interval'][0]>0 and min(v['relative_rmse_reduction'] for v in errors.values())>=-c['regression_allowance'];result=dict(results=results,primary_fresh_transfer_criterion_met=bool(positive),test_paired_changes=errors,train_seconds=train_seconds,later_execution_seconds=test_seconds,wall_seconds=time.perf_counter()-started,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,all_native_models_bounded=bool(V.all() and V2.all()),base_native_work=rt.counts,recruited_port_work=arc.counters(),controller_names=names,data_reference='UCI Appliances Energy Prediction; Luis Candanedo2017; CC BY4.0; DOI10.24432/C5VC8G',forecast_scope='Univariate chronological forecasting from prior/current appliance consumption; no weather/room covariates. Mechanism transfer test, not a full-dataset state-of-the-art claim.')
  dump(out/'summary.json',result);print(json.dumps(result))
 except BaseException as e:dump(out/'failure.json',dict(phase=phase,type=type(e).__name__,message=str(e)));raise
if __name__=='__main__':main()
