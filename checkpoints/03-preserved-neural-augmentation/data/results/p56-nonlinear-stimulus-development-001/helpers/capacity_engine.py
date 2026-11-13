"""Unlabeled transport alignment and curvature-born neural return connections."""
import numpy as np
from stimulus import orthogonal_rows


def geometry(native,frame,U,epsilon,c,rng):
    k=len(U);m=c['state_dimension'];d=c['differential_step']
    base=native.run(frame[None])[0].astype(float)
    Y=native.run(np.concatenate([frame[None]+epsilon*d*U,frame[None]-epsilon*d*U]))
    J=((Y[:k].astype(float)-Y[k:].astype(float))/(2*d)).T
    S=rng.normal(size=(c['curvature_samples'],k))*c['curvature_probe_scale']/np.sqrt(k)
    obs=native.run(frame[None]+epsilon*(S@U)).astype(float)
    pairs=[(i,j) for i in range(k) for j in range(i,k)]
    design=np.column_stack([S[:,i]*S[:,j]*(1 if i==j else 2) for i,j in pairs])
    residual=obs-base-S@J.T
    Qflat=np.linalg.lstsq(design,residual,rcond=1e-9)[0]
    u,s,vt=np.linalg.svd(J,full_matrices=False);rank=int(np.sum(s>s[0]*1e-5));N=u[:,:rank]
    curv=Qflat-(Qflat@N)@N.T
    _,singular,V=np.linalg.svd(curv,full_matrices=False)
    q=V[0];q-=N@(N.T@q);q/=max(np.linalg.norm(q),1e-30)
    H=np.zeros((k,k))
    for row,(i,j) in enumerate(pairs):H[i,j]=Qflat[row]@q;H[j,i]=H[i,j]
    ev,evec=np.linalg.eigh(H[:m,:m]);v=evec[:,np.argmax(abs(ev))]
    scale=max(float(np.linalg.norm(H[:m,:m])),1e-6)
    return dict(base=base,J=J,q=q,v=v,curvature_scale=scale,observable=V[:c['observable_dimension']],
                singular=singular,rank=rank,curvature_fit_relative_error=float(np.sum((design@Qflat-residual)**2)/max(np.sum(residual**2),1e-30)))


def build_candidates(native,seed,c):
    rng=np.random.default_rng(seed);m=c['state_dimension'];n=native.n
    ids=rng.integers(1000,native.config['vocab_size']-1000,size=c['tokens'])
    embedded=native.model.model.embed_tokens(native.mx.array(ids[None])).astype(native.mx.float32);native.mx.eval(embedded)
    frame=np.asarray(embedded)[0].reshape(-1).copy().astype(float)
    U=orthogonal_rows(rng,m+1,n);returns=orthogonal_rows(rng,c['random_return_projections']*m,n)/np.sqrt(n)
    gs=[];candidates=[]
    for eindex,epsilon in enumerate(c['amplitudes']):
        native.stage=f'geometry-{seed}-{eindex}';g=geometry(native,frame,U,epsilon,c,rng);gs.append(g)
        Jz=g['J'][:,:m];pinv=np.linalg.pinv(Jz,rcond=1e-5)
        for rho in c['spectral_radii']:
            parents=[]
            for orient in range(c['random_return_projections']):
                R=returns[orient*m:(orient+1)*m];M=R@Jz;left,_,right=np.linalg.svd(M,full_matrices=False)
                aligned=rho*(left@right)@pinv;parents.append(aligned)
                radius=max(float(np.max(abs(np.linalg.eigvals(M)))),1e-8);raw=rho*R/radius
                for kind,C in [('unaligned',raw),('aligned',aligned)]:
                    candidates.append(dict(kind=kind,epsilon=epsilon,rho=rho,orientation=orient,gamma=0.,geometry=eindex,C=C,parent=aligned))
            for gamma in c['nonlinear_gains']:
                delta=(gamma/g['curvature_scale'])*np.outer(g['v'],g['q']);C=parents[0]+delta
                leak=float(np.linalg.norm(delta@g['J'])/max(np.linalg.norm(parents[0]@g['J']),1e-30))
                candidates.append(dict(kind='curvature',epsilon=epsilon,rho=rho,orientation=0,gamma=gamma,geometry=eindex,C=C,parent=parents[0],delta=delta,first_order_leakage=leak))
    return dict(frame=frame,U=U,token_ids=ids,geometries=gs,candidates=candidates)


def source_selection(native,setup,c,seed):
    candidates=setup['candidates'];N=len(candidates);m=c['state_dimension'];U=setup['U'];frame=setup['frame']
    Cs=np.stack([p['C'] for p in candidates])
    parent_indices={(p['geometry'],p['rho'],p['orientation']):i for i,p in enumerate(candidates) if p['kind']=='aligned'}
    for i,p in enumerate(candidates):
        p['C']=Cs[i];p['parent']=Cs[parent_indices[(p['geometry'],p['rho'],p['orientation'])]];p.pop('delta',None)
    eps=np.array([p['epsilon'] for p in candidates])
    bases=np.array([setup['geometries'][p['geometry']]['base'] for p in candidates])
    rng=np.random.default_rng(seed+300000);drive=rng.uniform(-1,1,c['source_steps'])
    Z=np.zeros((N,2,m));Z[:,1]=rng.normal(size=(N,m))*.1
    trace=np.empty((N,2,c['source_steps'],m),dtype=np.float32);valid=np.ones(N,dtype=bool)
    # Arrays include candidates, learned geometry, stimulus/output buffers and a reserve.
    bound=Cs.nbytes+trace.nbytes+N*2*native.n*8*4+sum(g['J'].nbytes+g['observable'].nbytes+g['base'].nbytes for g in setup['geometries'])+8*1024**2
    assert bound<=c['controller_array_cap_bytes'],bound
    for t,u in enumerate(drive):
        X=frame[None,None]+eps[:,None,None]*(Z@U[:m]+c['input_gain']*u*U[m][None,None])
        Y=native.run(X.reshape(N*2,native.n)).reshape(N,2,native.n).astype(float)
        Z=np.einsum('cmn,crn->crm',Cs,Y-bases[:,None],optimize=True)
        good=np.isfinite(Z).all(axis=(1,2))&(np.max(abs(Z),axis=(1,2))<=c['state_limit']);valid&=good;Z[~valid]=0
        trace[:,:,t]=Z
    records=[];selected={}
    for i,p in enumerate(candidates):
        a=trace[i,0,c['source_burn']:].astype(float);b=trace[i,1,c['source_burn']:].astype(float)
        variance=float(np.mean((a-a.mean(axis=0))**2));echo=float(np.sqrt(np.mean((a-b)**2)/max(variance,1e-30)))
        cov=np.cov(a,rowvar=False);vals=np.linalg.eigvalsh(cov);score=float(vals.sum()**2/max(vals@vals,1e-30))
        admissible=bool(valid[i] and variance>1e-10 and echo<c['echo_tolerance'])
        # Learn a linear fading-memory reconstruction on one source segment;
        # measure the remaining reproducible response on a disjoint segment.
        # No assessment function, desired answer or evaluation stream is used.
        source_times=np.arange(c['source_burn'],c['source_steps'])
        lagged=np.column_stack([drive[source_times-lag] for lag in range(c['source_linear_lags'])])
        half=len(a)//2;train=a[:half];test=a[half:]
        mu=train.mean(0);cov=(train-mu).T@(train-mu)/half
        vals,vecs=np.linalg.eigh(cov);keep=vals>max(vals[-1]*c['source_whitening_floor'],1e-12)
        W=vecs[:,keep]/np.sqrt(np.maximum(vals[keep],1e-30))
        A=(train-mu)@W;AT=(test-mu)@W
        L=lagged[:half];LT=lagged[half:];lm=L.mean(0);ls=np.maximum(L.std(0),1e-8)
        L=(L-lm)/ls;LT=(LT-lm)/ls;am=A.mean(0)
        coef=np.linalg.solve(L.T@L+c['source_linear_ridge']*half*np.eye(L.shape[1]),L.T@(A-am))
        residual=AT-(LT@coef+am);den=np.sum((AT-AT.mean(0))**2,0)
        fraction=np.sum(residual**2,0)/np.maximum(den,1e-30)
        nonlinear_score=float(np.clip(fraction,0,1).sum())
        row={k:v for k,v in p.items() if k not in ['C','parent','delta']};row.update(index=i,valid=admissible,state_variance=variance,echo_ratio=echo,intrinsic_effective_rank=score,
            source_nonlinear_residual_score=nonlinear_score,source_observable_dimensions=int(keep.sum()),source_residual_fractions=fraction.tolist())
        records.append(row)
    for kind in ['unaligned','aligned','curvature']:
        rows=[r for r in records if r['kind']==kind and r['valid']]
        selected[kind]=max(rows,key=lambda r:(r['source_nonlinear_residual_score'],-r['index']))['index'] if rows else None
    raw=[r for r in records if r['kind']=='unaligned' and r['valid']]
    selected['unaligned_rank']=max(raw,key=lambda r:(r['intrinsic_effective_rank'],-r['index']))['index'] if raw else None
    return records,selected,trace,bound


def selected_systems(setup,selection,c,seed):
    systems=[];rng=np.random.default_rng(seed+400000)
    for kind,index in selection.items():
        if index is None:continue
        p=setup['candidates'][index];g=setup['geometries'][p['geometry']]
        def pack(label,C):
            return dict(kind=label,C=C.copy(),epsilon=p['epsilon'],frame=setup['frame'],U=setup['U'],base=g['base'],
                        observable=g['observable'],J=g['J'],rho=p['rho'],gamma=p['gamma'],source_index=index,seed=seed)
        systems.append(pack(kind,p['C']))
        if kind=='curvature':
            systems.append(pack('curvature_removed',p['parent']))
            q=rng.normal(size=len(g['base']))
            N=np.linalg.svd(g['J'],full_matrices=False)[0];q-=N@(N.T@q);q/=max(np.linalg.norm(q),1e-30)
            scrambled=p['parent']+(p['gamma']/g['curvature_scale'])*np.outer(g['v'],q)
            systems.append(pack('curvature_scrambled',scrambled))
    return systems


def run_systems(native,systems,c,drive,emit=None):
    N=len(systems);m=c['state_dimension'];D=m+c['observable_dimension'];T=len(drive)
    Cs=np.stack([s['C'] for s in systems]);U=np.stack([s['U'] for s in systems]);frames=np.stack([s['frame'] for s in systems])
    eps=np.array([s['epsilon'] for s in systems]);base=np.stack([s['base'] for s in systems]);R=np.stack([s['observable'] for s in systems])
    # Replace originals with views before allocating assessment traces. J was
    # sealed on disk and is not used by the native trajectory update.
    for i,s in enumerate(systems):
        s.update(C=Cs[i],U=U[i],frame=frames[i],base=base[i],observable=R[i]);s.pop('J',None)
    Z=np.zeros((N,m));states=np.zeros((N,T,m),np.float32);features=np.zeros((N,T,D),np.float32);valid=np.ones(N,dtype=bool)
    check_times=set(np.random.default_rng(93001).choice(T,32,replace=False).tolist());checks=[]
    bound=Cs.nbytes+U.nbytes+R.nbytes+frames.nbytes+base.nbytes+states.nbytes+features.nbytes+N*native.n*8*4+8*1024**2
    assert bound<=c['controller_array_cap_bytes'],bound
    for t,u in enumerate(drive):
        prior=Z.copy();X=frames+eps[:,None]*(np.einsum('im,imn->in',Z,U[:,:m])+c['input_gain']*u*U[:,m])
        Y=native.run(X).astype(float);delta=Y-base
        Z=np.einsum('imn,in->im',Cs,delta);good=np.isfinite(Z).all(axis=1)&(np.max(abs(Z),axis=1)<=c['state_limit']);valid&=good;Z[~valid]=0
        features[:,t,:m]=Z;features[:,t,m:]=np.einsum('ikn,in->ik',R,delta);features[~valid,t]=0;states[:,t]=Z
        if t in check_times:checks.append(dict(time=t,prior=prior,input=X.copy(),native_output=Y.astype(np.float32).copy(),state=Z.copy(),valid=valid.copy()))
        if emit is not None and (t+1)%512==0:emit(dict(kind='evaluation_progress',completed=t+1,total=T,valid_systems=int(valid.sum())))
    return states,features,valid,checks,bound


def capacity_targets(drive,start,count):
    idx=np.arange(start,start+count);u=np.array([drive[idx-lag] for lag in range(16)])
    P={1:np.sqrt(3)*u,2:np.sqrt(5)*(3*u*u-1)/2,3:np.sqrt(7)*(5*u**3-3*u)/2,4:3*(35*u**4-30*u*u+3)/8}
    descriptors=[];ys=[]
    def append(parts):
        y=np.ones(count)
        for lag,degree in parts:y*=P[degree][lag]
        descriptors.append(dict(degree=sum(d for _,d in parts),distinct_lags=len(parts),parts=parts));ys.append(y)
    for lag in range(16):append([(lag,1)])
    for lag in range(8):append([(lag,2)])
    from itertools import combinations
    for pair in combinations(range(8),2):append([(j,1) for j in pair])
    for lag in range(6):append([(lag,3)])
    for i in range(6):
        for j in range(6):
            if i!=j:append([(i,2),(j,1)])
    for triple in combinations(range(6),3):append([(j,1) for j in triple])
    for lag in range(4):append([(lag,4)])
    for i in range(4):
        for j in range(4):
            if i!=j:append([(i,3),(j,1)])
    for pair in combinations(range(4),2):append([(j,2) for j in pair])
    for i in range(4):
        for pair in combinations([j for j in range(4) if j!=i],2):append([(i,2),*[(j,1) for j in pair]])
    append([(j,1) for j in range(4)])
    return np.stack(ys,axis=1),descriptors


def assess(features,drive,c):
    start=c['evaluation_washout'];N=c['evaluation_training'];M=c['evaluation_test']
    Y,descriptors=capacity_targets(drive,start,N);YT,_=capacity_targets(drive,start+N,M)
    X=features[start:start+N].astype(float);XT=features[start+N:start+N+M].astype(float)
    mean=X.mean(axis=0);sd=X.std(axis=0);sd=np.maximum(sd,1e-8);X=(X-mean)/sd;XT=(XT-mean)/sd
    ym=Y.mean(axis=0);ridge=c['readout_ridge']*N
    B=np.linalg.solve(X.T@X+ridge*np.eye(X.shape[1]),X.T@(Y-ym));prediction=XT@B+ym
    ss=np.sum((YT-YT.mean(axis=0))**2,axis=0);r2=1-np.sum((prediction-YT)**2,axis=0)/np.maximum(ss,1e-30)
    positive=np.maximum(r2,0);multi=np.array([d['degree']>=2 and d['distinct_lags']>=2 for d in descriptors])
    return dict(total_capacity=float(positive.sum()),nonlinear_temporal_capacity=float(positive[multi].sum()),
        reliable_functions=int(np.sum(r2>=.5)),reliable_nonlinear_temporal_functions=int(np.sum((r2>=.5)&multi)),
        capacity_by_degree={str(d):float(positive[[r['degree']==d for r in descriptors]].sum()) for d in range(1,5)},
        tasks=len(descriptors),r2=r2.tolist()),dict(readout=B,feature_mean=mean,feature_scale=sd,target_mean=ym),descriptors
