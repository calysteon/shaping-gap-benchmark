"""Contract-specialized pair evaluation over acquired native response rows.

The architectural form is supplied, as in the first paper's structured reference.
All coefficient rows/directions are calibrated from frozen native observations.
Pair evaluation uses response moments instead of allocating candidate states.
"""
import numpy as np
import time
from base_planner import Planner,losses


def silu(x):return x/(1+np.exp(np.clip(-x,-60,60)))


class Bank:
    def __init__(self,path,k):
        ts=time.perf_counter()
        with np.load(path) as a:
            self.D=a['D'][:,:k].copy();self.anchors=a['anchors'].copy();self.eps=float(a['eps'])
            self.gram=self.D.T@self.D;self.P=np.linalg.solve(self.gram,self.D.T)
            self.diag=np.diag(self.gram).copy();self.k=k;self.n=self.D.shape[0]
            self.physical_preparation_seconds=time.perf_counter()-ts;ts=time.perf_counter()
            self.G=a['G'][:k].copy();self.U=a['U'][:k].copy();self.Q=a['Q'].copy()
            self.GD=self.G@self.D;self.UD=self.U@self.D;self.QD=self.Q@self.D
            self.behavior_preparation_seconds=time.perf_counter()-ts

    def payload(self):return sum(v.nbytes for v in self.__dict__.values() if isinstance(v,np.ndarray))

    def workspace_bound(self,rows,m,direct=False):
        # Conservative explicit-array reservation, including elementwise temporaries.
        return int(rows*self.k*(4*self.n+16*m+32)*8 if direct else
                   rows*self.k*(16*m+32)*8+32*self.k*8)

    def values(self,x,scope):
        radius=np.sqrt(np.dot(x,x)/self.n+self.eps)
        return silu(self.G@x/radius)*(self.U@x/radius),self.Q[scope]@x/radius

    def tiles(self,x,observed,scope,mask,rows,direct=False):
        x=np.asarray(x,dtype=np.float64);Q=self.Q[scope][mask];QD=self.QD[scope][mask]
        a,qbase=self.values(x,scope);qbase=qbase[mask];shift=observed[mask]-qbase
        x2=float(x@x);xD=x@self.D;gx=self.G@x;ux=self.U@x;qx=Q@x
        all_ids=np.arange(self.k)
        for start in range(0,self.k,rows):
            ids=all_ids[start:start+rows];t=a[ids]
            if direct:
                xa=x[None]+t[:,None]*self.D[:,ids].T
                ra=np.sqrt(np.einsum('an,an->a',xa,xa)/self.n+self.eps)
                tb=silu((xa@self.G.T)/ra[:,None])*((xa@self.U.T)/ra[:,None])
                xab=xa[:,None,:]+tb[:,:,None]*self.D.T[None]
                sqab=np.einsum('abn,abn->ab',xab,xab)/self.n
                qab=(xab@Q.T)/np.sqrt(sqab+self.eps)[:,:,None]
                qa=(xa@Q.T)/ra[:,None];sqa=np.einsum('an,an->a',xa,xa)/self.n
            else:
                sqa=(x2+2*t*xD[ids]+t*t*self.diag[ids])/self.n
                ra=np.sqrt(np.maximum(sqa+self.eps,1e-12))
                gb=(gx[None]+t[:,None]*self.GD[:,ids].T)/ra[:,None]
                ub=(ux[None]+t[:,None]*self.UD[:,ids].T)/ra[:,None]
                tb=silu(gb)*ub
                sqab=sqa[:,None]+(2*tb*(xD[None]+t[:,None]*self.gram[ids])+tb*tb*self.diag[None])/self.n
                qa=(qx[None]+t[:,None]*QD[:,ids].T)/ra[:,None]
                qab=(qx[None,None]+t[:,None,None]*QD[:,ids].T[:,None,:]+tb[:,:,None]*QD.T[None])/np.sqrt(np.maximum(sqab+self.eps,1e-12))[:,:,None]
            yield dict(ids=ids,amplitude=a,successor_amplitude=tb,q1=qa+shift,q2=qab+shift,
                       rms1=np.sqrt(np.maximum(sqa,0)),rms2=np.sqrt(np.maximum(sqab,0)))

    def legal(self,tile,z,c):
        ids=tile['ids'];a=tile['amplitude'];tb=tile['successor_amplitude']
        t=a[ids];root=np.sqrt(self.diag/self.n)
        legal1=(np.isfinite(t)&(abs(t)<=c['max_atom_amplitude'])&
                (abs(t)*root[ids]<=c['max_write_rms'])&(tile['rms1']<=c['max_live_rms'])&
                (abs(z[ids]+t)<=c['domain_radius']+1e-6))
        second=z[None]+tb
        second[np.arange(len(ids)),ids]+=t
        legal2=(legal1[:,None]&np.isfinite(tb)&(abs(tb)<=c['max_atom_amplitude'])&
                (abs(tb)*root[None]<=c['max_write_rms'])&(tile['rms2']<=c['max_live_rms'])&
                (abs(second)<=c['domain_radius']+1e-6))
        return legal1,legal2

    def rank(self,x,q,scope,mask,z,contract,c,rows,direct=False,pair=True):
        k=self.k;score=np.empty(k);single=np.empty(k);partner=np.full(k,-1,dtype=int);legal=np.zeros(k,dtype=bool)
        if not pair:
            x=np.asarray(x,dtype=np.float64);a,qbase=self.values(x,scope)
            sq=(x@x+2*a*(x@self.D)+a*a*self.diag)/self.n
            q1=(self.Q[scope][mask]@x+a[:,None]*self.QD[scope][mask].T)/np.sqrt(np.maximum(sq+self.eps,1e-12))[:,None]
            q1+=q[mask]-qbase[mask]
            legal=(np.isfinite(a)&(abs(a)<=c['max_atom_amplitude'])&(abs(a)*np.sqrt(self.diag/self.n)<=c['max_write_rms'])&
                   (np.sqrt(np.maximum(sq,0))<=c['max_live_rms'])&(abs(z+a)<=c['domain_radius']+1e-6))
            single=losses(q1,contract['lower'],contract['upper'])
            order=sorted(range(k),key=lambda j:(not bool(legal[j]),float(single[j]),j))
            return dict(order=order,score=single,single=single,partner=partner,amplitude=a,legal=legal)
        amplitude=None
        for tile in self.tiles(x,q,scope,mask,rows,direct):
            ids=tile['ids'];amplitude=tile['amplitude'];ok1,ok2=self.legal(tile,z,c)
            one=losses(tile['q1'],contract['lower'],contract['upper']);single[ids]=one;legal[ids]=ok1
            if pair:
                two=np.where(ok2,losses(tile['q2'],contract['lower'],contract['upper']),np.inf)
                best=np.argmin(two,axis=1);two_best=two[np.arange(len(ids)),best]
                stop=one<=two_best;score[ids]=np.minimum(one,two_best);partner[ids]=np.where(stop,-1,best)
            else:score[ids]=one
        order=sorted(range(k),key=lambda j:(not bool(legal[j]),float(score[j]),float(single[j]),j))
        return dict(order=order,score=score,single=single,partner=partner,amplitude=amplitude,legal=legal)


class ScaledPlanner(Planner):
    def __init__(self,native,scope,bank,units,method,c,emit,scope_id):
        super().__init__(native,scope,bank.D,bank.P,units,bank,method,c,emit)
        self.scope_id=scope_id

    def propose(self,node,contract):
        if self.method=='native_local':return super().propose(node,contract)
        import time
        ts=time.perf_counter();self.cache={}
        pair=(len(node['graph'])+1<self.c['max_depth'] and self.c['query_budget_lpe']-self.used()>=8)
        r=self.model.rank(node['x'],node['q'],self.scope_id,contract['mask'],self.coordinate(node['x']),contract,self.c,self.c['factored_tile_rows'],pair=pair)
        self.proposals+=self.model.k*self.model.k if pair else self.model.k
        self.planning_seconds+=time.perf_counter()-ts
        return [dict(unit_index=j,predicted_amp=float(r['amplitude'][j]),predicted_score=float(r['score'][j]),
                     successor_class=int(r['partner'][j])) for j in r['order']]

    def result(self,node,branches,depth):
        r=super().result(node,branches,depth)
        w=self.model.workspace_bound(self.c['factored_tile_rows'],self.c['contract_size']) if self.method!='native_local' else 8*self.model.k*self.model.k
        bound=self.model.payload()+w+8*1024**2
        assert bound<=self.c['controller_payload_cap_bytes'],bound
        r.update(interface_payload_bytes=self.model.payload(),forecast_workspace_bound_bytes=w,
                 controller_array_bound_bytes=bound)
        return r
