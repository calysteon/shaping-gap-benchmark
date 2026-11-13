"""P59: source-calibrated nonlinear augmentation of preserved native runtimes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,time,gc,resource
import numpy as np
from native import NativeField
from capacity_engine import assess,capacity_targets

HERE=Path(__file__).resolve().parent;LAB=HERE.parent
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def padded(native,X):
    X=np.asarray(X);pad=(-len(X))%native.batch
    return native.run(np.concatenate([X,np.repeat(X[-1:],pad,axis=0)]) if pad else X)[:len(X)]

def acquire(native,cases,c):
    K=len(cases);n=native.n
    frames=np.stack([s['frame'] for s in cases]);U=np.stack([s['U'] for s in cases]);base=np.stack([s['base'] for s in cases]);C=np.stack([s['C'] for s in cases]);eps=np.array([s['epsilon'] for s in cases])
    for i,s in enumerate(cases):
        v=s['v'];b=s['C']@s['J'][:,8]
        if v@b<0 or (v@b==0 and v[np.argmax(abs(v))]<0):v=-v
        s['v']=v
        J=s['J'];N=np.linalg.svd(J,full_matrices=False)[0]
        random=np.random.default_rng(c['random_seed']+s['seed']).normal(size=n);random-=N@(N.T@random);random/=np.linalg.norm(random);s['random_q']=random
    native.stage='source-even-response-orientation';ref=padded(native,frames).astype(float);curv=np.zeros((K,2))
    for h in c['sign_probe_steps']:
        direction=np.stack([s['v']@s['U'][:8] for s in cases]);Y=padded(native,np.concatenate([frames+eps[:,None]*h*direction,frames-eps[:,None]*h*direction])).astype(float)
        even=(Y[:K]+Y[K:]-2*ref)/(2*h*h)
        for i,s in enumerate(cases):curv[i,0]+=even[i]@s['q'];curv[i,1]+=even[i]@s['random_q']
    for i,s in enumerate(cases):
        if curv[i,0]<0:s['q']=-s['q']
        if curv[i,1]<0:s['random_q']=-s['random_q']
    rng=np.random.default_rng(c['source_seed']);drive=rng.uniform(-1,1,c['source_steps']);Z=np.zeros((K,2,8));Z[:,1]=rng.normal(size=(K,8))*.1
    states=np.empty((K,2,len(drive),8),np.float32);signals=np.empty((K,len(drive),2),float);valid=np.ones(K,bool)
    native.stage='source-baseline-action-calibration'
    for t,u in enumerate(drive):
        X=frames[:,None]+eps[:,None,None]*(np.einsum('csm,cmn->csn',Z,U[:,:8])+c['input_gain']*u*U[:,8,None,:])
        Y=padded(native,X.reshape(-1,n)).reshape(K,2,n).astype(float);dy=Y-base[:,None]
        Z=np.einsum('cmn,csn->csm',C,dy)
        good=np.isfinite(Z).all((1,2))&(np.max(abs(Z),(1,2))<=c['state_limit']);valid&=good;Z[~valid]=0;states[:,:,t]=Z
        for i,s in enumerate(cases):signals[i,t]=[dy[i,0]@s['q'],dy[i,0]@s['random_q']]
    rows=[];burn=c['source_burn']
    for i,s in enumerate(cases):
        parent_rms=rms(states[i,0,burn:]);qr=rms(signals[i,burn:,0]);rr=rms(signals[i,burn:,1]);v_rms=rms(s['v'])
        ready=bool(valid[i] and qr>1e-10 and rr>1e-12 and parent_rms>1e-10)
        rows.append(dict(case=i,seed=s['seed'],source_bounded=bool(valid[i]),source_ready=ready,parent_action_rms=parent_rms,
            acquired_signal_rms=qr,random_signal_rms=rr,v_rms=v_rms,even_orientation_measurements=curv[i].tolist(),
            replica_disagreement_rms=rms(states[i,0,burn:]-states[i,1,burn:])))
    return rows,dict(drive=drive,states=states,signals=signals)

def construct(cases,cal,c):
    Cs=[];meta=[]
    for i,(s,r) in enumerate(zip(cases,cal)):
        mats=[s['C'].copy()];rows=[dict(kind='baseline',fraction=0.,alpha=0.)]
        for family in ['acquired','random_same_norm','random_same_action']:
            for fraction in c['signed_action_fractions']:
                alpha=fraction*r['parent_action_rms']/max(r['acquired_signal_rms']*r['v_rms'],1e-30) if r['source_ready'] else 0.
                if family=='random_same_action':alpha*=r['acquired_signal_rms']/max(r['random_signal_rms'],1e-30)
                q=s['q'] if family=='acquired' else s['random_q'];delta=alpha*np.outer(s['v'],q);new=s['C']+delta
                leakage=float(np.linalg.norm(delta@s['J'])/max(np.linalg.norm(s['C']@s['J']),1e-30))
                assert leakage<1e-8
                mats.append(new);rows.append(dict(kind=family,fraction=fraction,alpha=alpha,addition_norm=float(np.linalg.norm(delta)),identified_transport_change=leakage,
                    source_ready=r['source_ready']))
        mats.append(s['C'].copy());rows.append(dict(kind='baseline_clone',fraction=0.,alpha=0.))
        assert len(rows)==c['conditions_per_case']
        while len(mats)<c['padded_conditions_per_case']:mats.append(s['C'].copy())
        Cs.append(np.stack(mats));meta.append([dict(case=i,parent=s['parent'],seed=s['seed'],slot=j,epsilon=s['epsilon'],**row) for j,row in enumerate(rows)])
    return np.stack(Cs),meta

def execute(native,cases,Cs,drive,c,emit):
    K,S,m,n=Cs.shape;D=40;T=len(drive);N=c['conditions_per_case']
    frames=np.stack([s['frame'] for s in cases]);U=np.stack([s['U'] for s in cases]);base=np.stack([s['base'] for s in cases]);R=np.stack([s['observable'] for s in cases]);eps=np.array([s['epsilon'] for s in cases])
    Z=np.zeros((K,S,m));valid=np.ones((K,S),bool);first_failure=np.full((K,S),-1,int)
    features=np.empty((K,N,T,D),np.float32);states=np.empty((K,N,T,m),np.float32)
    check_times=set(np.random.default_rng(c['verification_seed']).choice(T,c['verification_steps'],replace=False));checks=[]
    native.stage='preserved-return-assessment'
    for t,u in enumerate(drive):
        prior=Z.copy();X=frames[:,None]+eps[:,None,None]*(np.einsum('csm,cmn->csn',Z,U[:,:8])+c['input_gain']*u*U[:,8,None,:])
        Y=native.run(X.reshape(K*S,n)).reshape(K,S,n).astype(float);dy=Y-base[:,None]
        Z=np.einsum('csmn,csn->csm',Cs,dy);good=np.isfinite(Z).all(2)&(np.max(abs(Z),2)<=c['state_limit']);first_failure[(~good)&valid]=t;valid&=good;Z[~valid]=0
        features[:,:,t,:m]=Z[:,:N];features[:,:,t,m:]=np.einsum('crn,csn->csr',R,dy[:,:N]);features[~valid[:,:N],t]=0;states[:,:,t]=Z[:,:N]
        if t in check_times:checks.append(dict(time=t,prior=prior,input=X.copy(),native_output=Y.astype(np.float32),state=Z.copy(),valid=valid.copy()))
        if (t+1)%512==0:
            event=dict(event='assessment_progress',completed=t+1,total=T,bounded=int(valid[:,:N].sum()),conditions=K*N);emit(event);print(json.dumps(event),flush=True)
    fidelity=[]
    for i in range(K):
        same=np.array_equal(features[i,0],features[i,13]);error=float(np.max(abs(features[i,0].astype(float)-features[i,13].astype(float))))
        fidelity.append(dict(case=i,features_bitexact=same,max_feature_error=error));assert same,('baseline_clone_mismatch',i,error)
    return states,features,valid[:,:N],first_failure[:,:N],checks,fidelity

def score_block(r2,descriptors):
    positive=np.maximum(r2,0);multi=np.array([d['degree']>=2 and d['distinct_lags']>=2 for d in descriptors])
    return dict(total_capacity=float(positive.sum()),nonlinear_temporal_capacity=float(positive[multi].sum()),
        reliable_functions=int(sum(r2>=.5)),reliable_nonlinear_temporal_functions=int(sum((r2>=.5)&multi)),r2=r2.tolist())

def main():
    plan=json.loads((HERE/'protocol.json').read_text());c=plan['planned_config']['config'];out=LAB/'results/p59-preserved-return-augmentation-001'
    out.mkdir(exist_ok=False);(out/'helpers').mkdir();(out/'models').mkdir();shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(HERE/'protocol.json',out/'protocol.json')
    for n in ['native.py','capacity_engine.py','stimulus.py']:shutil.copy2(HERE/n,out/'helpers'/n)
    dump(out/'manifest.json',dict(plan_id=plan['id'],config=c,created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),helper_sha256={n:sha(HERE/n) for n in ['native.py','capacity_engine.py','stimulus.py']}))
    results=[];models=[];started=time.perf_counter()
    with (out/'runs.jsonl').open('w') as log:
      def emit(v):log.write(json.dumps(v,allow_nan=False)+'\n');log.flush()
      try:
       for label in ['0.6B','1.7B']:
        ts=time.perf_counter();directory=out/'models'/label;directory.mkdir();cases=[];inherited=[]
        for name in c['parents']:
            parent=LAB/'results'/name;summary=json.loads((parent/'summary.json').read_text());d=parent/'models'/label;metas=json.loads((d/'systems.json').read_text())
            cfg=summary['config'];inherited.append(dict(parent=name,source_records=next(m for m in summary['models'] if m['model']==label)['source_records']))
            for i,meta in enumerate(metas):
                if meta['kind']!=c['parent_kind']:continue
                p=d/f'system-{i}.npz';assert sha(p)==json.loads((d/'sealed.json').read_text())['files'][p.name]
                s=load(p);g=load(d/f"geometry-{meta['seed']}-{cfg['amplitudes'].index(meta['epsilon'])}.npz")
                s.update(q=g['q'].copy(),v=g['v'].copy(),parent=name,seed=meta['seed'],epsilon=meta['epsilon'],source_interface_sha256=sha(p));cases.append(s)
        assert len(cases)==6
        receipt=json.loads((LAB/'results'/c['parents'][0]/'models'/label/'checkpoint.json').read_text());dump(directory/'checkpoint.json',receipt)
        native=NativeField(receipt['local_path'],emit,2,c['native_batch_size']);expected=json.loads((LAB/'results'/c['parents'][0]/'models'/label/'model-summary.json').read_text())['parameter_sha256'];assert native.before[0]==expected
        before=dict(native.counts);a=time.perf_counter();cal,source=acquire(native,cases,c);Cs,meta=construct(cases,cal,c)
        source_work={k:native.counts[k]-before[k] for k in before};source_seconds=time.perf_counter()-a
        dump(directory/'source-calibration.json',cal);np.savez_compressed(directory/'source-calibration.npz',**source)
        for i,s in enumerate(cases):np.savez_compressed(directory/f'case-{i}.npz',**{k:v for k,v in s.items() if isinstance(v,np.ndarray)})
        np.savez_compressed(directory/'returns.npz',returns=Cs)
        dump(directory/'systems.json',meta);dump(directory/'sealed.json',dict(files={p.name:sha(p) for p in directory.glob('*.npz')},assessment_labels_exposed=False))
        emit(dict(event='couplings_acquired',model=label,cases=len(cases),source_work=source_work,source_seconds=source_seconds,calibration=cal))
        print(json.dumps(dict(model=label,stage='couplings_sealed',cases=len(cases),conditions=sum(map(len,meta)),source_seconds=source_seconds)),flush=True)
        T=c['evaluation_washout']+c['evaluation_training']+c['test_blocks']*c['evaluation_test'];drive=np.random.default_rng(c['assessment_seed']).uniform(-1,1,T)
        before=dict(native.counts);a=time.perf_counter();states,features,valid,failures,checks,fidelity=execute(native,cases,Cs,drive,c,emit)
        assessment_seconds=time.perf_counter()-a;assessment_work={k:native.counts[k]-before[k] for k in before}
        np.savez_compressed(directory/'assessment.npz',drive=drive,states=states,features=features,valid=valid,first_failure=failures)
        np.savez_compressed(directory/'native-checks.npz',times=np.array([r['time'] for r in checks]),prior=np.array([r['prior'] for r in checks]),inputs=np.array([r['input'] for r in checks]),outputs=np.array([r['native_output'] for r in checks]),states=np.array([r['state'] for r in checks]),valid=np.array([r['valid'] for r in checks]))
        dump(directory/'baseline-fidelity.json',fidelity)
        for i,rows in enumerate(meta):
            for j,m in enumerate(rows):
                first,head,descriptors=assess(features[i,j],drive,c);r2a=np.array(first['r2'])
                start=c['evaluation_washout']+c['evaluation_training']+c['evaluation_test'];Y,ds=capacity_targets(drive,start,c['evaluation_test']);XT=features[i,j,start:start+c['evaluation_test']].astype(float)
                pred=((XT-head['feature_mean'])/head['feature_scale'])@head['readout']+head['target_mean'];r2b=1-np.sum((pred-Y)**2,0)/np.sum((Y-Y.mean(0))**2,0)
                if not valid[i,j]:r2a=np.full(len(descriptors),-1.);r2b=r2a.copy()
                blocks=[score_block(r2a,descriptors),score_block(r2b,descriptors)]
                result=dict(model=label,**m,bounded=bool(valid[i,j]),first_failure=int(failures[i,j]),blocks=blocks)
                results.append(result);emit(dict(event='assessment_result',**result));np.savez_compressed(directory/f'readout-{i}-{j}.npz',**head)
            print(json.dumps(dict(model=label,case=i,seed=rows[0]['seed'],bounded=int(valid[i].sum()),baseline_capacity=[results[-len(rows)]['blocks'][b]['nonlinear_temporal_capacity'] for b in range(2)])),flush=True)
        dump(directory/'targets.json',descriptors);assert native.digest()==native.before
        rec=dict(model=label,parameter_sha256=native.before[0],stored_parameter_bytes=native.before[1],weights_unchanged=True,
            inherited_source=inherited,source_native_work=source_work,source_seconds=source_seconds,assessment_native_work=assessment_work,assessment_seconds=assessment_seconds,
            native_work=native.counts,native_seconds=native.seconds,wall_seconds=time.perf_counter()-ts,baseline_clones_bitexact=all(r['features_bitexact'] for r in fidelity),
            observed_array_bytes=dict(returns=Cs.nbytes,states=states.nbytes,features=features.nbytes),mlx_peak_bytes=native.mx.get_peak_memory(),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        models.append(rec);dump(directory/'model-summary.json',rec);dump(out/'progress.json',dict(models_completed=len(models),models=2))
        del native,Cs,cases,checks,states,features;gc.collect()
        import mlx.core as mx;mx.clear_cache()
       dump(out/'summary.json',dict(config=c,models=models,results=results,wall_seconds=time.perf_counter()-started));print(json.dumps(dict(complete=True,conditions=len(results))),flush=True)
      except Exception as e:dump(out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise

if __name__=='__main__':main()
