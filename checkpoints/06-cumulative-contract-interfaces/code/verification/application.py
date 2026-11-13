"""Independent preprocessing, frozen forecast head and held-out metric replay."""
from pathlib import Path
import json,csv,hashlib,time
import numpy as np
HERE=Path(__file__).resolve().parent;LAB=HERE.parent
CHECKPOINT=Path(__file__).resolve().parents[2];DATA=CHECKPOINT/'data/experiments';PREVIOUS=CHECKPOINT.parent/'05-component-learning-cycle';OUTPUT=CHECKPOINT/'runs/replay';OUTPUT.mkdir(parents=True,exist_ok=True)


def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def predict(X,h):return ((X-h['feature_mean'])/h['feature_scale'])@h['readout']+h['target_mean']
def main():
 start=time.perf_counter();root=DATA/'p66-energy-forecast-transfer-001';p=json.loads((root/'protocol.json').read_text());c=p['config'];summary=json.loads((root/'summary.json').read_text());seal=json.loads((root/'program-seal.json').read_text());x=load(root/'input.npz');y=x['energy_wh'];n=len(y);train=int(.6*n);valid=int(.8*n);scaler=json.loads((root/'scaler.json').read_text());log=np.log1p(y);assert np.median(log[:train])==scaler['median'];assert np.std(log[:train])==scaler['standard_deviation'];assert np.array_equal(np.tanh((log-scaler['median'])/(3*scaler['standard_deviation'])),x['drive']);features=load(root/'training-features.npz')['features'];later=load(root/'later-features.npz')['features'];maps=[dict(A=np.eye(24),b=np.zeros(24)) for _ in range(2)]
 parent=DATA/'p65-program-contract-interfaces-001/contexts'
 for arm in ['feature','contract','interface_only']:maps.append(load(parent/f'0.6B-151003-{arm}/terminal/final-state/map.npz'))
 ids=np.arange(c['washout'],train-max(c['horizons']));Y=np.column_stack([y[ids+h] for h in c['horizons']]);maxerr=0.;refiterror=0.;heads=[]
 for j in range(5):
  path=root/f'head-{j}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['heads'][str(j)];h=load(path);X=features[j,ids].astype(float)@maps[j]['A']+maps[j]['b'];A=(X-h['feature_mean'])/h['feature_scale'];B=np.linalg.solve(A.T@A+c['readout_ridge']*len(A)*np.eye(24),A.T@(Y-h['target_mean']));refiterror=max(refiterror,float(np.max(abs(B-h['readout']))));heads.append(h)
 for split,lo,hi in [('validation',train,valid),('test',valid,n)]:
  z=load(root/f'{split}-predictions.npz');idx=np.arange(lo,hi-max(c['horizons']));truth=np.column_stack([y[idx+h] for h in c['horizons']]);assert np.array_equal(idx,z['time_index']);assert np.array_equal(truth,z['truth']);preds={}
  for j,name in enumerate(summary['controller_names']):preds[name]=np.maximum(predict(later[j,idx-train].astype(float)@maps[j]['A']+maps[j]['b'],heads[j]),0)
  ar=load(root/'ar-head.npz');XX=np.column_stack([y[idx-lag] for lag in range(64)]);preds['ar64']=np.maximum(predict(XX,ar),0);preds['persistence']=np.repeat(y[idx,None],3,axis=1);preds['daily_seasonal']=np.column_stack([y[idx+h-144] for h in c['horizons']])
  for name,P in preds.items():
   maxerr=max(maxerr,float(np.max(abs(P-z[name]))))
   for j,h in enumerate(c['horizons']):
    e=P[:,j]-truth[:,j];row=next(r for r in summary['results'] if r['split']==split and r['model']==name and r['horizon']==h);assert abs(np.mean(abs(e))-row['mae'])<1e-10;assert abs(np.sqrt(np.mean(e*e))-row['rmse'])<1e-10
 assert refiterror<1e-8 and maxerr<1e-8
 result=dict(passed=True,forecast_metric_rows=len(summary['results']),source_heads_refit=True,training_only_scaler_verified=True,chronological_no_boundary_crossing_targets=True,head_seals_verified=True,maximum_coefficient_refit_error=refiterror,maximum_prediction_replay_error=maxerr,all_classical_baselines_recomputed=True,wall_seconds=time.perf_counter()-start);(OUTPUT/'application.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
