"""One native beam search; guidance alone varies across interface controls."""
import time
import numpy as np
from substrate import array_sha
from native import work
from geometry_backend import Geometry


class Exhausted(Exception):pass


def losses(q,lower,upper):
    e=np.maximum(lower-q,0)+np.maximum(q-upper,0)
    return np.sum(e*e,axis=-1)


def metrics(q,lower,upper,target_count):
    e=np.maximum(lower-q,0)+np.maximum(q-upper,0)
    return dict(loss=float(e@e),satisfied=bool(np.all(e<=1e-5)),
        protected_violation=float(e[target_count:].max(initial=0)),max_violation=float(e.max()))


class Planner:
    def __init__(self,native,scope,D,P,units,model,method,c,emit):
        self.native,self.scope,self.D,self.P,self.units=native,scope,D,P,units
        self.model,self.method,self.c,self.emit=model,method,c,emit
        self.start_work=work(native.counts);self.cache={};self.geometry=Geometry(native,{})
        self.cache_peak=0;self.planning_seconds=0.;self.proposals=0
        self.permutation=np.random.default_rng(c['corruption_seed']).permutation(len(units))

    def used(self):return 1+work(self.native.counts)-self.start_work
    def require(self,n):
        if self.used()+n>self.c['query_budget_lpe']:raise Exhausted()

    def coordinate(self,x):return self.P@(x-self.scope['anchor'])

    def predicted_legal(self,z,amp):
        z=np.asarray(z)
        return (np.max(abs(z),axis=-1)<=self.c['domain_radius']+1e-6)&np.isfinite(z).all(axis=-1)&np.isfinite(amp)&(abs(amp)<=self.c['max_atom_amplitude'])

    def propose(self,node,contract):
        ts=time.perf_counter();self.cache={}
        x=node['x'];q=node['q'];z=self.coordinate(x);d=len(self.units)
        mask=contract['mask'];lower=contract['lower'];upper=contract['upper']
        if self.method=='native_local':
            self.require(2*len({l for l,j in self.units})+1+2*len(mask))
            readout=dict(layer=self.scope['readout']['layer'],indices=[self.scope['readout']['indices'][i] for i in mask])
            _,jac=self.geometry.qj(x,readout)
            amps=[]
            for layer,j in self.units:
                if layer not in self.cache:self.cache[layer]=self.native.activity(x,layer)
                amps.append(float(self.cache[layer][j]))
            amps=np.array(amps)
            q1=q[mask][None]+((jac@self.D)*amps[None]).T
            score=losses(q1,lower,upper);partner=np.full(d,-1,dtype=int);mod=np.zeros((d,len(mask)))
        else:
            allow_pair=(len(node['graph'])+1<self.c['max_depth'] and self.c['query_budget_lpe']-self.used()>=8)
            if self.method=='differential_one' or not allow_pair:
                if self.method=='analytic_pair':
                    a0,qbase=self.model.values(z);amps=a0[0]
                    za=np.repeat(z[None],d,axis=0);za[np.arange(d),np.arange(d)]+=amps
                    _,qa=self.model.values(za);q1=q[None]+qa-qbase
                else:
                    y=self.model.evaluate(z,derivatives=False);amps=y[:,0]
                    q1=q[None]+y[:,1:]
                score=losses(q1[:,mask],lower,upper)
                partner=np.full(d,-1,dtype=int);mod=np.zeros((d,len(mask)))
            else:
                variant='no_conditioning' if self.method=='differential_no_conditioning' else 'full'
                perm=self.permutation if self.method=='differential_corrupt' else None
                amps,q1,pair,ampB=self.model.forecast(z,q,variant,perm)
                zA=np.repeat(z[None],d,axis=0);zA[np.arange(d),np.arange(d)]+=amps
                zAB=np.repeat(zA[:,None],d,axis=1);zAB[:,np.arange(d),np.arange(d)]+=ampB
                pair_legal=self.predicted_legal(zAB,ampB)&self.predicted_legal(zA,amps)[:,None]
                pair_loss=losses(pair[:,:,mask],lower,upper)
                pair_loss=np.where(pair_legal,pair_loss,np.inf)
                partner=np.argmin(pair_loss,axis=1);score=pair_loss[np.arange(d),partner]
                single_score=losses(q1[:,mask],lower,upper)
                stop=single_score<=score
                score=np.minimum(score,single_score)
                # Keep the learned/contextual successor class and the forecast
                # composed effect, not a fixed semantic menu or class quota.
                mod=(pair[np.arange(d),partner]-q1-q1[partner]+q)[:,mask]
                mod[stop]=0.;partner[stop]=-1
            self.proposals+=d*d if self.method!='differential_one' and allow_pair else d
        zA=np.repeat(z[None],d,axis=0);zA[np.arange(d),np.arange(d)]+=amps
        legal=self.predicted_legal(zA,amps)
        single=losses(q1 if self.method=='native_local' else q1[:,mask],lower,upper)
        ranked=sorted(range(d),key=lambda j:(not bool(legal[j]),float(score[j]),float(single[j]),j))
        self.cache_peak=max(self.cache_peak,sum(a.nbytes for a in self.cache.values()))
        assert self.cache_peak<=self.c['activity_workspace_cap_bytes']
        self.planning_seconds+=time.perf_counter()-ts
        return [dict(unit_index=int(j),predicted_amp=float(amps[j]),predicted_score=float(score[j]),
            successor_class=int(partner[j]),predicted_conditioning=mod[j].tolist()) for j in ranked]

    def execute(self,node,action,contract,node_id):
        column=action['unit_index'];layer,unit=self.units[column]
        self.require(2 if layer in self.cache else 4)
        if layer not in self.cache:self.cache[layer]=self.native.activity(node['x'],layer)
        amp=float(self.cache[layer][unit]);self.cache_peak=max(self.cache_peak,sum(a.nbytes for a in self.cache.values()))
        if abs(amp)>self.c['max_atom_amplitude']:
            self.emit(dict(kind='branch',node=node_id,parent=node['id'],action=action,amplitude=amp,legal=False,reason='amplitude',used_lpe=self.used()))
            return None
        write=self.native.write(layer,unit,amp);x=node['x']+write;z=self.coordinate(x)
        legal=bool(np.isfinite(x).all() and np.max(abs(z))<=self.c['domain_radius']+1e-6 and
            np.sqrt(np.mean(write*write))<=self.c['max_write_rms'] and np.sqrt(np.mean(x*x))<=self.c['max_live_rms'])
        if not legal:
            self.emit(dict(kind='branch',node=node_id,parent=node['id'],action=action,amplitude=amp,legal=False,reason='geometry',used_lpe=self.used(),write_sha256=array_sha(write)))
            return None
        q=self.native.consumer(x,self.scope['readout']['layer'],self.scope['readout']['indices'])
        m=metrics(q[contract['mask']],contract['lower'],contract['upper'],self.c['target_count'])
        binding=dict(unit_index=column,layer=layer,unit=unit,port='live')
        child=dict(id=node_id,x=x,q=q,graph=node['graph']+[binding],metrics=m)
        self.emit(dict(kind='branch',node=node_id,parent=node['id'],action=action,amplitude=amp,legal=True,
            before_sha256=array_sha(node['x']),write_sha256=array_sha(write),after_sha256=array_sha(x),
            q_sha256=array_sha(q),q=q.tolist(),metrics=m,used_lpe=self.used()))
        return child

    def search(self,x,q,contract):
        initial=dict(id=0,x=x,q=q,graph=[],metrics=metrics(q[contract['mask']],contract['lower'],contract['upper'],self.c['target_count']))
        frontier=[initial];best=initial;serial=0;exhausted=False;visited={array_sha(x)}
        for depth in range(1,self.c['max_depth']+1):
            children=[]
            for parent in frontier:
                try:actions=self.propose(parent,contract)
                except Exhausted:exhausted=True;break
                for action in actions[:self.c['children_per_parent']]:
                    serial+=1
                    try:child=self.execute(parent,action,contract,serial)
                    except Exhausted:serial-=1;exhausted=True;break
                    if child is None:continue
                    signature=array_sha(child['x'])
                    if signature in visited:continue
                    visited.add(signature);children.append(child)
                    if (child['metrics']['loss'],len(child['graph']))<(best['metrics']['loss'],len(best['graph'])):best=child
                    if child['metrics']['satisfied']:return self.result(child,serial,depth)
                if exhausted:break
            if exhausted or not children:break
            children.sort(key=lambda n:(n['metrics']['loss'],n['id']));frontier=children[:self.c['beam_width']]
        return self.result(best,serial,depth)

    def result(self,node,branches,depth):
        return dict(graph=node['graph'],metrics=node['metrics'],final_sha256=array_sha(node['x']),
            final_q=node['q'],branches=branches,depth_searched=depth,query_lpe=self.used(),
            compatibility_pairs_scored=self.proposals,planning_seconds=self.planning_seconds,
            activity_workspace_peak_bytes=self.cache_peak,final_x=node['x'])
