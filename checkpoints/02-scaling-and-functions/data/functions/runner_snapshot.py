"""P49: one neural program per complete external Boolean specification."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import resource
import shutil
import time
import numpy as np
from native import Native,work,difference
from substrate import array_sha
from factored import Bank

HERE=Path(__file__).resolve().parent;LAB=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def specification(name,bits):
    if name.startswith('and'):return np.all(bits,axis=1)
    if name.startswith('or'):return np.any(bits,axis=1)
    if name.startswith('xor'):return np.sum(bits,axis=1)%2==1
    if name=='majority3':return np.sum(bits,axis=1)>=2
    if name=='mux3':return np.where(bits[:,0],bits[:,1],bits[:,2]).astype(bool)
    raise ValueError(name)


def encoding(bank,seed,width,c):
    # Neither frame/directions nor decoder depends on the requested function.
    Q=bank.Q[1,0];rng=np.random.default_rng(seed)
    frame=rng.normal(size=bank.n);frame-=Q*(frame@Q)/(Q@Q)
    frame*=c['frame_rms']/np.sqrt(np.mean(frame*frame))
    columns=[]
    for _ in range(3):
        v=rng.normal(size=bank.n);v-=Q*(v@Q)/(Q@Q);v-=frame*(v@frame)/(frame@frame)
        for u in columns:v-=u*(v@u)/(u@u)
        v*=c['input_rms']/np.sqrt(np.mean(v*v));columns.append(v)
    bits=((np.arange(2**width)[:,None]>>np.arange(width))&1).astype(bool)
    x=(frame[None]+(2*bits.astype(float)-1)@np.array(columns[:width])).astype(np.float32)
    return bits,x,frame,np.array(columns)


def quality(q,labels,margin):
    value=np.asarray(q)[:,0];sign=np.where(labels,1.,-1.);error=np.maximum(margin-sign*value,0)
    return dict(loss=float(error@error),robust_satisfied=bool(np.all(error<=1e-5)),
                boolean_correct=bool(np.array_equal(value>0,labels)),correct_inputs=int(np.sum((value>0)==labels)),
                inputs=len(labels),min_signed_margin=float(np.min(sign*value)),max_violation=float(error.max()))


class FunctionPlanner:
    def __init__(self,native,bank,units,scope,roots,labels,method,c,emit):
        self.native,self.bank,self.units,self.scope=native,bank,units,scope
        self.roots,self.labels,self.method,self.c,self.emit=roots,labels,method,c,emit
        self.start=work(native.counts);self.cache={};self.planning_seconds=0.;self.proposals=0
        n=len(labels);m=c['output_margin']
        self.contracts=[dict(mask=np.array([0]),lower=np.array([m if label else -np.inf]),upper=np.array([np.inf if label else -m])) for label in labels]
        self.array_bound=bank.payload()+n*bank.workspace_bound(c['factored_tile_rows'],1)+8*1024**2
        assert self.array_bound<=c['controller_payload_cap_bytes']

    def used(self):return len(self.labels)+work(self.native.counts)-self.start
    def rank(self,node):
        ts=time.perf_counter();self.cache={};k=self.bank.k;inputs=len(self.labels)
        z=(self.bank.P@(node['x']-self.roots).T).T
        paired=self.method=='paired' and len(node['graph'])+1<self.c['max_depth']
        singles=np.zeros(k);scores=np.empty(k);legal=np.ones(k,dtype=bool);partners=np.full(k,-1,dtype=int)
        if not paired:
            for i in range(inputs):
                r=self.bank.rank(node['x'][i],node['q'][i],1,[0],z[i],self.contracts[i],self.c,self.c['factored_tile_rows'],pair=False)
                singles+=r['single'];legal&=r['legal']
            scores=singles
        else:
            generators=[self.bank.tiles(node['x'][i],node['q'][i],1,[0],self.c['factored_tile_rows']) for i in range(inputs)]
            for tiles in zip(*generators):
                ids=tiles[0]['ids'];pair_score=np.zeros((len(ids),k));pair_ok=np.ones((len(ids),k),dtype=bool);one=np.zeros(len(ids));first_ok=np.ones(len(ids),dtype=bool)
                for i,tile in enumerate(tiles):
                    ok1,ok2=self.bank.legal(tile,z[i],self.c);first_ok&=ok1;pair_ok&=ok2
                    sign=1. if self.labels[i] else -1.
                    one+=np.maximum(self.c['output_margin']-sign*tile['q1'][:,0],0)**2
                    pair_score+=np.maximum(self.c['output_margin']-sign*tile['q2'][:,:,0],0)**2
                pair_score=np.where(pair_ok,pair_score,np.inf);best=np.argmin(pair_score,axis=1);two=pair_score[np.arange(len(ids)),best]
                scores[ids]=np.minimum(one,two);singles[ids]=one;legal[ids]=first_ok;partners[ids]=np.where(one<=two,-1,best)
        self.proposals+=inputs*(k*k if paired else k);self.planning_seconds+=time.perf_counter()-ts
        order=sorted(range(k),key=lambda j:(not bool(legal[j]),float(scores[j]),float(singles[j]),j))
        return [dict(unit_index=j,predicted_score=float(scores[j]),successor_class=int(partners[j])) for j in order[:self.c['children_per_parent']]]

    def execute(self,parent,action,node_id):
        j=action['unit_index'];layer,unit=self.units[j];inputs=len(self.labels)
        needed=sum(2 if (i,layer) in self.cache else 4 for i in range(inputs))
        if self.used()+needed>self.c['query_budget_lpe']:return None,True
        frames=[];readouts=[]
        for i,x in enumerate(parent['x']):
            key=(i,layer)
            if key not in self.cache:self.cache[key]=self.native.activity(x,layer)
            amp=float(self.cache[key][unit])
            if abs(amp)>self.c['max_atom_amplitude']:
                self.emit(dict(kind='branch',node=node_id,parent=parent['id'],action=action,legal=False,reason='amplitude',input=i,amplitude=amp));return None,False
            w=self.native.write(layer,unit,amp);y=x+w
            legal=(np.isfinite(y).all() and np.sqrt(np.mean(w*w))<=self.c['max_write_rms'] and
                   np.sqrt(np.mean(y*y))<=self.c['max_live_rms'] and np.max(abs(self.bank.P@(y-self.roots[i])))<=self.c['domain_radius']+1e-6)
            if not legal:
                self.emit(dict(kind='branch',node=node_id,parent=parent['id'],action=action,legal=False,reason='geometry',input=i));return None,False
            frames.append(y);readouts.append(self.native.consumer(y,self.scope['layer'],self.scope['indices']))
        assert sum(v.nbytes for v in self.cache.values())<=self.c['activity_workspace_cap_bytes']
        frames=np.array(frames);readouts=np.array(readouts);m=quality(readouts,self.labels,self.c['output_margin'])
        child=dict(id=node_id,x=frames,q=readouts,graph=parent['graph']+[dict(unit_index=j,layer=layer,unit=unit,port='live')],metrics=m)
        self.emit(dict(kind='branch',node=node_id,parent=parent['id'],action=action,legal=True,activation_sha256=array_sha(frames),q=readouts.tolist(),metrics=m))
        return child,False

    def search(self,q):
        root=dict(id=0,x=self.roots.copy(),q=q.copy(),graph=[],metrics=quality(q,self.labels,self.c['output_margin']))
        frontier=[root];best=root;serial=0;exhausted=False;visited={array_sha(root['x'])}
        for depth in range(1,self.c['max_depth']+1):
            children=[]
            for parent in frontier:
                for action in self.rank(parent):
                    serial+=1;child,exhausted=self.execute(parent,action,serial)
                    if exhausted:serial-=1;break
                    if child is None:continue
                    h=array_sha(child['x'])
                    if h in visited:continue
                    visited.add(h);children.append(child)
                    if (child['metrics']['loss'],len(child['graph']))<(best['metrics']['loss'],len(best['graph'])):best=child
                    if child['metrics']['robust_satisfied']:return self.result(child,serial)
                if exhausted:break
            if exhausted or not children:break
            children.sort(key=lambda v:(v['metrics']['loss'],v['id']));frontier=children[:self.c['beam_width']]
        return self.result(best,serial)

    def result(self,node,branches):
        return dict(graph=node['graph'],metrics=node['metrics'],branches=branches,query_lpe=self.used(),planning_seconds=self.planning_seconds,
                    predicted_case_compositions=self.proposals,controller_array_bound_bytes=self.array_bound,
                    final_sha256=array_sha(node['x']),final_q=node['q'].tolist(),final_x=node['x'])


def main():
    plan=json.loads((HERE/'protocol-functions.json').read_text());c=plan['planned_config']['config']
    source=LAB/c['source_npz'];assert sha(source)==c['source_sha256'];source_run=source.parent
    out=LAB/'results/p49-external-functions-001';out.mkdir(exist_ok=False);(out/'helpers').mkdir();(out/'cases').mkdir()
    shutil.copy2(__file__,out/'runner_snapshot.py');shutil.copy2(HERE/'protocol-functions.json',out/'protocol.json')
    helpers=['substrate.py','geometry_backend.py','native.py','interfaces.py','base_planner.py','factored.py']
    for name in helpers:shutil.copy2(HERE/name,out/'helpers'/name)
    dump(out/'manifest.json',dict(plan_id=plan['id'],config=c,created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),helper_sha256={name:sha(HERE/name) for name in helpers}))
    bank=Bank(source,512);units=[tuple(v) for v in json.loads((source_run/'units.json').read_text())]
    parent=LAB/c['source_directory'];scope=json.loads((parent/'scopes/1/domain.json').read_text())['readout']
    receipt=json.loads((source_run/'checkpoint.json').read_text());dump(out/'checkpoint.json',receipt)
    raw=(out/'runs.jsonl').open('x');native=None;started=time.perf_counter()
    def emit(v):
        if 'stage' not in v:v['stage']=native.stage if native is not None else 'setup'
        raw.write(json.dumps(v,allow_nan=False)+'\n');raw.flush()
    try:
        native=Native(receipt['local_path'],emit);assert native.before[0]==c['parameter_sha256'];results=[];case=0
        for seed in c['encoder_seeds']:
            for name in c['functions']:
                width=2 if name.endswith('2') else 3;bits,X,frame,directions=encoding(bank,seed,width,c);labels=specification(name,bits)
                np.savez_compressed(out/'cases'/f't{case}.npz',roots=X,bits=bits,labels=labels,frame=frame,directions=directions)
                native.stage=f't{case}-initial';before=dict(native.counts);ts=time.perf_counter()
                q=np.array([native.consumer(x,scope['layer'],scope['indices']) for x in X])
                initial=dict(native_work=difference(before,native.counts),seconds=time.perf_counter()-ts,metrics=quality(q,labels,c['output_margin']))
                for method in c['methods'][case%2:]+c['methods'][:case%2]:
                    native.stage=f't{case}-{method}';before=dict(native.counts);ts=time.perf_counter()
                    emit(dict(kind='case_start',case=case,function=name,seed=seed,method=method,bits=bits.tolist(),labels=labels.tolist(),root_sha256=array_sha(X)))
                    planner=FunctionPlanner(native,bank,units,scope,X,labels,method,c,emit);r=planner.search(q);final=r.pop('final_x')
                    search_seconds=time.perf_counter()-ts;search_work=difference(before,native.counts);before=dict(native.counts);ts=time.perf_counter();replay=X.copy()
                    for action in r['graph']:
                        for i,x in enumerate(replay):
                            h=native.activity(x,action['layer']);replay[i]+=native.write(action['layer'],action['unit'],float(h[action['unit']]))
                    qr=np.array([native.consumer(x,scope['layer'],scope['indices']) for x in replay])
                    assert np.array_equal(replay,final) and np.array_equal(qr,np.array(r['final_q'],dtype=np.float32))
                    r.update(case=case,function=name,seed=seed,method=method,inputs=len(bits),initial=initial,search_seconds=search_seconds,search_native_work=search_work,
                             deploy_seconds=time.perf_counter()-ts,deploy_native_work=difference(before,native.counts),native_deploy_exact=True)
                    emit(dict(kind='case_result',**r));results.append(r)
                    print(json.dumps(dict(case=case,function=name,seed=seed,method=method,metrics=r['metrics'],program_length=len(r['graph']))),flush=True)
                case+=1;dump(out/'progress.json',dict(completed_targets=case,total_targets=24))
        assert native.parameter_digest()==native.before and sha(source)==c['source_sha256']
        dump(out/'summary.json',dict(config=c,results=results,targets=case,weights_unchanged=True,source_unchanged=True,native_work_total=native.counts,
             wall_seconds=time.perf_counter()-started,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mlx_peak_bytes=native.mx.get_peak_memory()))
        print(json.dumps(dict(complete=True,targets=case,rows=len(results))),flush=True)
    except Exception as error:dump(out/'failure.json',dict(type=type(error).__name__,message=str(error)));raise
    finally:
        raw.close()
        if native is not None:native.close()

if __name__=='__main__':main()
