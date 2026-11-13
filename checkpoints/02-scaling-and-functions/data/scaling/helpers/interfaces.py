"""Generic integrable quadratic effect interfaces and an architectural reference.

Quadratic/Hermite regression is conventional. Native E/K observations supervise
the interface; no original model parameter is updated. Inference differentiates
the retained polynomial and integrates its conditioning along an action direction.
"""
import numpy as np


def feature_matrix(u):
    n,d=u.shape
    return np.column_stack([np.ones(n),u,.5*u*u]+[u[:,i]*u[:,j] for i in range(d) for j in range(i+1,d)])


def derivative_features(u):
    n,d=u.shape;p=1+2*d+d*(d-1)//2
    out=np.zeros((n,d,p));out[:,:,1:d+1]=np.eye(d)[None]
    for j in range(d):out[:,j,1+d+j]=u[:,j]
    k=1+2*d
    for i in range(d):
        for j in range(i+1,d):out[:,i,k]=u[:,j];out[:,j,k]=u[:,i];k+=1
    return out


class QuadraticInterface:
    def __init__(self,c,L,H,shape):
        self.c,self.L,self.H,self.shape=c,L,H,tuple(shape)
        self.hdiag=np.diagonal(H,axis1=1,axis2=2).reshape(*self.shape,-1)

    @classmethod
    def fit(cls,data,radius,with_gradients,ridge=1e-6,gradient_weight=1.):
        z=data['z'];shape=data['y'].shape[1:];y=data['y'].reshape(len(z),-1)
        mean=y.mean(0);scale=np.maximum(y.std(0),1e-4);yn=(y-mean)/scale
        u=z/radius;X=feature_matrix(u);gram=X.T@X/len(z);rhs=X.T@yn/len(z)
        if with_gradients:
            D=derivative_features(u).reshape(-1,X.shape[1])
            labels=data['g'].reshape(len(z)*z.shape[1],-1)*radius/scale
            gram+=gradient_weight*(D.T@D)/len(D);rhs+=gradient_weight*(D.T@labels)/len(D)
        penalty=np.eye(X.shape[1])*ridge;penalty[0,0]=0
        beta=np.linalg.solve(gram+penalty,rhs)*scale
        beta[0]+=mean;d=z.shape[1]
        L=beta[1:d+1].T/radius
        H=np.zeros((y.shape[1],d,d))
        for j in range(d):H[:,j,j]=beta[1+d+j]/radius**2
        k=1+2*d
        for i in range(d):
            for j in range(i+1,d):H[:,i,j]=H[:,j,i]=beta[k]/radius**2;k+=1
        return cls(beta[0],L,H,shape)

    def evaluate(self,z,derivatives=True):
        Hz=np.einsum('oij,j->oi',self.H,z,optimize=True)
        y=(self.c+self.L@z+.5*(Hz@z)).reshape(self.shape)
        return (y,(self.L+Hz).reshape(*self.shape,-1)) if derivatives else y

    def forecast(self,z,q,variant='full',permutation=None):
        y,gradient=self.evaluate(z);a=y[:,0];e=y[:,1:]
        if variant=='no_conditioning':conditional=np.broadcast_to(y[None],(len(a),*y.shape)).copy()
        else:
            grad=gradient.transpose(2,0,1);diagonal=self.hdiag.transpose(2,0,1)
            if permutation is not None:grad=grad[permutation];diagonal=diagonal[permutation]
            conditional=y[None]+a[:,None,None]*grad+.5*a[:,None,None]**2*diagonal
        return a,q[None]+e,q[None,None]+e[:,None]+conditional[:,:,1:],conditional[:,:,0]

    def save(self,path):np.savez_compressed(path,c=self.c,L=self.L,H=self.H,shape=np.array(self.shape))
    def payload(self):return self.c.nbytes+self.L.nbytes+self.H.nbytes


class AnalyticReference:
    """Known RMSNorm/SwiGLU structure, calibrated native projection rows.

This reference receives architecture formulas unavailable to the generic
regression learner. It is reported explicitly rather than mistaken for learning.
"""
    def __init__(self,base,linear,gram,eps,gate0,gateD,up0,upD,q0,qD):
        self.base,self.linear,self.gram,self.eps=base,linear,gram,eps
        self.gate0,self.gateD,self.up0,self.upD,self.q0,self.qD=gate0,gateD,up0,upD,q0,qD

    def values(self,z):
        z=np.atleast_2d(z)
        r=np.sqrt(np.maximum(self.base+2*z@self.linear+np.einsum('ni,ij,nj->n',z,self.gram,z,optimize=True)+self.eps,1e-12))
        g=(self.gate0[None]+z@self.gateD.T)/r[:,None]
        u=(self.up0[None]+z@self.upD.T)/r[:,None]
        amp=(g/(1+np.exp(np.clip(-g,-60,60))))*u
        q=(self.q0[None]+z@self.qD.T)/r[:,None]
        return amp,q

    def forecast(self,z,q,variant='full',permutation=None):
        amp,_=self.values(z);amp=amp[0];d=len(amp)
        zA=np.repeat(z[None],d,axis=0);zA[np.arange(d),np.arange(d)]+=amp
        ampB,qA=self.values(zA)
        zAB=np.repeat(zA[:,None,:],d,axis=1)
        zAB[:,np.arange(d),np.arange(d)]+=ampB
        _,qAB=self.values(zAB.reshape(d*d,d));qAB=qAB.reshape(d,d,-1)
        # Anchor forecasts on actual observed q to avoid inherited constant
        # calibration roundoff being counted as an effect.
        _,qbase=self.values(z)
        return amp,q[None]+qA-qbase,q[None,None]+qAB-qbase,ampB

    def save(self,path):np.savez_compressed(path,**self.__dict__)
    def payload(self):return sum(v.nbytes if isinstance(v,np.ndarray) else 8 for v in self.__dict__.values())
