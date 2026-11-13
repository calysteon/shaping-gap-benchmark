"""Shared observation transport; unchanged numerical implementation."""
import numpy as np

def scale(X):
 X=np.asarray(X,float);return X.mean(0),np.maximum(X.std(0),1e-8)

def normal(X,mu,sd):return (np.asarray(X,float)-mu)/sd

def fit_interface(base,new,ridge=.001):
 bm,bs=scale(base);nm,ns=scale(new);X=normal(new,nm,ns);Y=normal(base,bm,bs);a=ridge*len(X);T=np.linalg.solve(X.T@X+a*np.eye(X.shape[1]),X.T@Y+a*np.eye(X.shape[1]));return dict(transport=T,new_feature_mean=nm,new_feature_scale=ns,old_feature_mean=bm,old_feature_scale=bs)

def apply_interface(X,t):return normal(X,t['new_feature_mean'],t['new_feature_scale'])@t['transport']
