"""Independent polynomial construction and frozen-readout/interface replay."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from numpy.polynomial.legendre import legval

LAB=Path(__file__).resolve().parents[1];OUT=LAB.parent.parent/'outputs/neural-stimulus/preserved-augmentation'
def targets(u,desc,start,N):
    ix=np.arange(start,start+N);Y=[]
    for d in desc:
        y=np.ones(N)
        for lag,degree in d['parts']:
            coeff=np.zeros(degree+1);coeff[-1]=1;y*=np.sqrt(2*degree+1)*legval(u[ix-lag],coeff)
        Y.append(y)
    return np.stack(Y,axis=1)
def rr(pred,y):return 1-np.sum((pred-y)**2,0)/np.sum((y-y.mean(0))**2,0)
def headpredict(X,h):return ((X-h['feature_mean'])/h['feature_scale'])@h['readout']+h['target_mean']
def load(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def main():
    began=time.perf_counter();p59=LAB/'results/p59-preserved-return-augmentation-001';p60=LAB/'results/p60-shared-observer-transport-001'
    s=json.loads((p59/'summary.json').read_text());c=s['config'];score_max=0.;source_max=0.;rank_counts=[];checks=0;seals=0;transport_scores=0;transport_error=0.
    for model in s['models']:
        label=model['model'];d=p59/'models'/label;data=load(d/'assessment.npz');C=load(d/'returns.npz')['returns'];meta=json.loads((d/'systems.json').read_text());desc=json.loads((d/'targets.json').read_text());source=load(d/'source-calibration.npz')
        for name,h in json.loads((d/'sealed.json').read_text())['files'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h;seals+=1
        rows=[r for r in s['results'] if r['model']==label]
        for i in range(6):
            case=load(d/f'case-{i}.npz');assert np.array_equal(C[i,0],case['C']);assert np.array_equal(data['features'][i,0],data['features'][i,13])
            origin=LAB/'results'/meta[i][0]['parent']/'models'/label;oldmeta=json.loads((origin/'systems.json').read_text());oldindex=next(j for j,m in enumerate(oldmeta) if m['kind']=='unaligned' and m['seed']==meta[i][0]['seed']);old=load(origin/f'system-{oldindex}.npz')
            for k in ['C','frame','U','base','observable','J']:assert np.array_equal(case[k],old[k])
            parent_rms=float(np.sqrt(np.mean(source['states'][i,0,c['source_burn']:].astype(float)**2)))
            for m in meta[i]:
                j=m['slot'];sv=np.linalg.svd(C[i,j]@case['observable'].T,compute_uv=False);rank_counts.append(int(sum(sv>max(sv[0]*1e-8,1e-12))))
                if m['kind'] in ['acquired','random_same_action']:
                    q=case['q'] if m['kind']=='acquired' else case['random_q'];qidx=0 if m['kind']=='acquired' else 1
                    delta=C[i,j]-case['C'];action=source['signals'][i,c['source_burn']:,qidx,None]*(delta@q)
                    actual=float(np.sqrt(np.mean(action**2))/parent_rms);source_max=max(source_max,abs(actual-abs(m['fraction'])));assert abs(actual-abs(m['fraction']))<1e-8
                row=next(r for r in rows if r['case']==i and r['slot']==j);h=load(d/f'readout-{i}-{j}.npz')
                for b in range(2):
                    start=c['evaluation_washout']+c['evaluation_training']+b*c['evaluation_test'];y=targets(data['drive'],desc,start,c['evaluation_test']);X=data['features'][i,j,start:start+c['evaluation_test']].astype(float)
                    r=rr(headpredict(X,h),y)
                    if not row['bounded']:r[:]=-1
                    e=float(np.max(abs(r-row['blocks'][b]['r2'])));score_max=max(score_max,e);assert e<1e-8;checks+=1
    if (p60/'summary.json').exists():
        z=json.loads((p60/'summary.json').read_text());cfg=z['config']
        for model in z['models']:
            label=model['model'];d=p60/'models'/label;pd=p59/'models'/label;data=load(d/'validation.npz');olddata=load(pd/'assessment.npz');meta=json.loads((d/'systems.json').read_text());desc=json.loads((pd/'targets.json').read_text());y=targets(data['drive'],desc,cfg['washout'],cfg['test_steps'])
            for name,h in json.loads((d/'sealed.json').read_text())['files'].items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h;seals+=1
            for i,rows in enumerate(meta):
                bh=load(pd/f'readout-{i}-0.npz');Y=(olddata['features'][i,0,c['evaluation_washout']:c['evaluation_washout']+c['evaluation_training']].astype(float)-bh['feature_mean'])/bh['feature_scale']
                for j,m in enumerate(rows):
                    h=load(pd/f"readout-{i}-{m['slot']}.npz");face=load(d/f'interface-{i}-{j}.npz');src=olddata['features'][i,m['slot'],c['evaluation_washout']:c['evaluation_washout']+c['evaluation_training']].astype(float);src=(src-h['feature_mean'])/h['feature_scale'];ridge=cfg['transport_ridge']*len(src)
                    T=np.linalg.solve(src.T@src+ridge*np.eye(40),src.T@Y+ridge*np.eye(40));e=float(np.max(abs(T-face['transport'])));transport_error=max(transport_error,e);assert e<1e-8
                    X=data['features'][i,j,cfg['washout']:].astype(float);Xn=(X-h['feature_mean'])/h['feature_scale'];predictions=dict(raw_old=headpredict(X,bh),transported_old=Xn@T@bh['readout']+bh['target_mean'],capability=headpredict(X,h))
                    row=next(r for r in z['results'] if r['model']==label and r['case']==i and r['kind']==m['kind'])
                    for k,pred in predictions.items():
                        r=rr(pred,y)
                        if not row['bounded']:r[:]=-1
                        e=float(np.max(abs(r-row['score_vectors'][k])));score_max=max(score_max,e);assert e<1e-8;transport_scores+=1
    report=dict(p59_test_block_score_vectors_reproduced=checks,p60_frozen_program_score_vectors_reproduced=transport_scores,max_r2_error=score_max,
        source_calibration_max_error=source_max,transport_refit_max_error=transport_error,sealed_files_verified=seals,
        preserved_residual_feedback_rank_range=[min(rank_counts),max(rank_counts)],verification_seconds=time.perf_counter()-began,
        scope='Independent Legendre target construction, saved frozen heads, source action calibration, original baseline arrays and unlabeled transport refits. Native re-execution is separate.')
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'offline-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
