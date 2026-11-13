"""Learn stimulus ports and a single-pass joint-response channel from observations."""
import time
import numpy as np


def orthogonal_rows(rng,k,n):
    q,_=np.linalg.qr(rng.normal(size=(n,k)));return q.T*np.sqrt(n)


def quadratic_design(S):
    k=S.shape[1];pairs=[(i,j) for i in range(k) for j in range(i,k)]
    D=np.column_stack([np.ones(len(S)),S,*[S[:,i]*S[:,j]*(1 if i==j else 2) for i,j in pairs]])
    return D,pairs


def acquire_geometry(native,c,rng):
    n=native.n;k=c['probe_dimensions'];frame=rng.normal(size=n);frame*=c['frame_rms']/np.sqrt(np.mean(frame*frame))
    U=orthogonal_rows(rng,k,n);S=rng.normal(size=(c['source_samples'],k))/np.sqrt(k)
    native.stage='unlabeled-geometry';X=frame[None]+c['source_radius']*(S@U);Y=native.run(X)
    D,pairs=quadratic_design(S);coeff=np.linalg.lstsq(D,Y.astype(float),rcond=1e-10)[0]
    linear=coeff[1:1+k];_,_,v=np.linalg.svd(linear,full_matrices=False);N=v[:k].T
    Q=np.zeros((k,k,n))
    for row,(i,j) in enumerate(pairs):Q[i,j]=coeff[1+k+row];Q[j,i]=Q[i,j]
    flat=Q.reshape(k*k,n);flat=flat-(flat@N)@N.T
    _,singular,V=np.linalg.svd(flat,full_matrices=False);q=V[0];q*=1 if q[np.argmax(abs(q))]>=0 else -1
    H=np.einsum('ijn,n->ij',Q,q);values,vectors=np.linalg.eigh(H)
    a=(vectors[:,-1]+vectors[:,0])/np.sqrt(2);b=(vectors[:,-1]-vectors[:,0])/np.sqrt(2)
    ports=np.stack([a@U,b@U]);random_ports=U[:2].copy()
    SV=rng.normal(size=(c['validation_samples'],k))/np.sqrt(k);DV,_=quadratic_design(SV)
    native.stage='geometry-validation';YV=native.run(frame[None]+c['source_radius']*(SV@U))
    prediction=DV@coeff;error=float(np.sum((prediction-YV)**2)/max(np.sum((YV-YV.mean(axis=0))**2),1e-30))
    return dict(frame=frame,U=U,source_coordinates=S,source_responses=Y,validation_coordinates=SV,validation_responses=YV,
                quadratic_coefficients=coeff,curvature_singular_values=singular,eigenvalues=values,learned_ports=ports,
                random_ports=random_ports,geometry_relative_mse=error)


def acquire_channel(native,frame,ports,c):
    grid=np.linspace(-1,1,c['calibration_grid_size']);a,b=np.meshgrid(grid,grid,indexing='ij');ab=np.column_stack([a.ravel(),b.ravel()])
    V=np.vander(grid,N=len(grid),increasing=True);P,_=np.linalg.qr(V)
    # Full tensor-product mode decomposition of the observed factorial responses.
    basis=np.column_stack([np.outer(P[:,i],P[:,j]).ravel() for i in range(len(grid)) for j in range(len(grid))])
    desired=1*len(grid)+1;records=[];candidates=[]
    check=np.array([[.25,.25],[.25,-.5],[-.5,.75],[.75,.5],[-.75,-.25],[.5,-.75]],dtype=float)
    for epsilon in c['amplitudes']:
        native.stage='unlabeled-channel';Y=native.run(frame[None]+epsilon*(ab@ports))
        modes=basis.T@Y.astype(float);signal=modes[desired]
        nuisance=np.delete(modes,[0,desired],axis=0)
        _,s,vt=np.linalg.svd(nuisance,full_matrices=False)
        rank=int(np.sum(s>max(s[0]*1e-7,1e-9))) if len(s) else 0
        Q=vt[:rank].T;q=signal-(signal@Q)@Q.T;norm=np.linalg.norm(q)
        if norm<1e-12:
            records.append(dict(epsilon=epsilon,emitted=False,reason='no_independent_joint_response'));continue
        q/=norm
        # Affine output calibration for the selected interaction mode. No model update.
        y=Y.astype(float)@q;A=np.column_stack([np.ones(len(ab)),ab[:,0]*ab[:,1]])
        offset,gain=np.linalg.lstsq(A,y,rcond=1e-12)[0]
        native.stage='channel-consistency';YC=native.run(frame[None]+epsilon*(check@ports));actual=(YC.astype(float)@q-offset)/gain
        # Supplied bilinear-isolation prior, checked on fresh source stimuli.
        homogeneity=float(np.sqrt(np.mean((actual-check[:,0]*check[:,1])**2)))
        rec=dict(epsilon=epsilon,emitted=bool(abs(gain)>=c['minimum_coupling_signal']),gain=float(gain),
                 source_grid_rmse=float(np.sqrt(np.mean(((y-offset)/gain-ab[:,0]*ab[:,1])**2))),
                 source_consistency_rmse=homogeneity,nuisance_rank=rank)
        records.append(rec);candidates.append((rec,dict(frame=frame.copy(),ports=ports.copy(),readout=q,offset=float(offset),gain=float(gain),epsilon=epsilon)))
    eligible=[v for v in candidates if v[0]['emitted'] and v[0]['source_consistency_rmse']<=c['homogeneity_tolerance']]
    if not eligible:return None,records
    # Largest measured usable signal; test examples have not yet been drawn.
    selected=max(eligible,key=lambda v:abs(v[0]['gain']))
    return selected[1],records


def apply(native,interface,pairs):
    X=interface['frame'][None]+interface['epsilon']*(np.asarray(pairs)@interface['ports'])
    Y=native.run(X);result=(Y.astype(float)@interface['readout']-interface['offset'])/interface['gain']
    return result,Y
