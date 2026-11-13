"""Neural-only interface diagnostics and outcome-trained cross-chart selection."""
import numpy as np

FEATURES=['fraction','absolute_fraction','base_state_rank','state_rank_change','base_observation_rank','observation_rank_change','state_variance_logratio','increment_power_logratio','trajectory_cosine','transport_holdout_logerror','transport_identity_logdistance','transport_logcondition','transport_error_drift','linear_history_readability_change','nonlinear_history_readability_change','base_linear_readability','base_nonlinear_readability','state_autocorrelation_change','observation_skew_change','observation_kurtosis_change']
def scale(X):
 X=np.asarray(X,float);return X.mean(0),np.maximum(X.std(0),1e-8)
def normal(X,mu,sd):return (np.asarray(X,float)-mu)/sd
def fit_interface(base,new,ridge=.001):
 bm,bs=scale(base);nm,ns=scale(new);X=normal(new,nm,ns);Y=normal(base,bm,bs);a=ridge*len(X);T=np.linalg.solve(X.T@X+a*np.eye(X.shape[1]),X.T@Y+a*np.eye(X.shape[1]));return dict(transport=T,new_feature_mean=nm,new_feature_scale=ns,old_feature_mean=bm,old_feature_scale=bs)
def apply_interface(X,t):return normal(X,t['new_feature_mean'],t['new_feature_scale'])@t['transport']
def rank(X):
 X=np.asarray(X,float);X=X-X.mean(0);e=np.linalg.eigvalsh(X.T@X/len(X));return float(e.sum()**2/max(e@e,1e-30))
def readability(X,Y):
 N=len(X)//2;mu,sd=scale(X[:N]);A=normal(X[:N],mu,sd);B=normal(X[N:],mu,sd);ym=Y[:N].mean(0);W=np.linalg.solve(A.T@A+.01*N*np.eye(X.shape[1]),A.T@(Y[:N]-ym));pred=B@W+ym;rr=1-np.sum((pred-Y[N:])**2,0)/np.maximum(np.sum((Y[N:]-Y[N:].mean(0))**2,0),1e-30);return float(np.maximum(rr,0).mean())
def source_features(base,new,drive,start,fraction,seed=117001):
 B=np.asarray(base,float);X=np.asarray(new,float);N=len(B);half=N//2
 face=fit_interface(B,X);bf=fit_interface(B[:half],X[:half]);V=apply_interface(X[half:],bf);Y=normal(B[half:],bf['old_feature_mean'],bf['old_feature_scale']);err=np.mean((V-Y)**2);trainerr=np.mean((apply_interface(X[:half],bf)-normal(B[:half],bf['old_feature_mean'],bf['old_feature_scale']))**2)
 ix=np.arange(start,start+N);history=np.stack([drive[ix-l] for l in range(16)],1);rng=np.random.default_rng(seed);W=rng.normal(size=(16,32))/4;bias=rng.uniform(-np.pi,np.pi,32);fourier=np.cos(history@W*2+bias)
 bl=readability(B,history);bn=readability(B,fourier);xl=readability(X,history);xn=readability(X,fourier)
 def variance(A):return max(float(np.mean((A-A.mean(0))**2)),1e-20)
 def increments(A):return max(float(np.mean(np.diff(A[:,:8],axis=0)**2)),1e-20)
 def ac(A):
  Z=normal(A,*scale(A));return float(np.mean(Z[1:,:8]*Z[:-1,:8]))
 def moments(A):
  Z=np.clip(normal(A,*scale(A)),-20,20);return float(np.mean(abs(np.mean(Z**3,0)))),float(np.mean(np.mean(Z**4,0)))
 bsk,bku=moments(B);xsk,xku=moments(X);ZB=normal(B,*scale(B));ZX=normal(X,*scale(X));cos=float(np.mean(ZB*ZX));T=face['transport'];sv=np.linalg.svd(T,compute_uv=False)
 f=np.array([fraction,abs(fraction),rank(B[:,:8])/8,(rank(X[:,:8])-rank(B[:,:8]))/8,rank(ZB)/40,(rank(ZX)-rank(ZB))/40,np.log(variance(X[:,:8])/variance(B[:,:8])),np.log(increments(X)/increments(B)),cos,np.log1p(err),np.log1p(np.linalg.norm(T-np.eye(40))/np.sqrt(40)),np.log1p(sv[0]/max(sv[-1],1e-8)),np.log1p(err)-np.log1p(trainerr),xl-bl,xn-bn,bl,bn,ac(X)-ac(B),xsk-bsk,np.log1p(xku)-np.log1p(bku)])
 assert np.isfinite(f).all();return f,face

def fit_policy(X,Y,bandwidth,ridge):
 mu,sd=scale(X);Z=np.clip(normal(X,mu,sd),-8,8);d=((Z[:,None]-Z[None])**2).sum(2);nonzero=d[d>1e-12];width=max(float(np.median(nonzero))*bandwidth**2,1e-6);K=np.exp(-d/(2*width));ym=Y.mean(0);dual=np.linalg.solve(K+ridge*len(X)*np.eye(len(X)),Y-ym)
 return dict(feature_mean=mu,feature_scale=sd,training_features=Z,dual=dual,target_mean=ym,width=np.array(width),ridge=np.array(ridge),bandwidth=np.array(bandwidth))
def predict(X,p):
 Z=np.clip(normal(X,p['feature_mean'],p['feature_scale']),-8,8);d=((Z[:,None]-p['training_features'][None])**2).sum(2);return np.exp(-d/(2*float(p['width'])))@p['dual']+p['target_mean']
def choose(pred,c):
 eligible=[i for i,(gain,retention) in enumerate(pred) if gain>=c['quality_acceptance'] and retention>=c['retention_acceptance']]
 return max(eligible,key=lambda i:(pred[i,0],-abs(c['signed_action_fractions'][i]),-i))+1 if eligible else 0
