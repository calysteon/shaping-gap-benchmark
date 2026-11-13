"""Native source acquisition and transparent linear-projection query accounting."""
import numpy as np
from substrate import Substrate
from geometry_backend import Geometry
from interfaces import AnalyticReference


PROJECTION_WEIGHTS={
    'producer_gate_projection':1,'producer_up_projection':1,'consumer_gate_projection':1,
    'atom_down_projection':1,'interface_basis_probe':1,
    'geometry_producer_gate':1,'geometry_producer_up':1,
    'geometry_consumer_coordinate_vjp':1,'geometry_producer_coordinate_vjp':2,
    'analytic_pre_projection':1,'analytic_pre_vjp':1}


def work(counts):return sum(PROJECTION_WEIGHTS.get(k,0)*v for k,v in counts.items())
def difference(a,b):return {k:b.get(k,0)-a.get(k,0) for k in sorted(set(a)|set(b))}


class Native(Substrate):
    def pre_row(self,x,layer,unit,kind):
        mx=self.mx;l=self.layers[layer]
        def f(z):
            normalized=l.post_attention_layernorm(z[None]);self.meter('analytic_pre_norm',layer)
            value=(l.mlp.gate_proj if kind=='gate' else l.mlp.up_proj)(normalized)
            self.meter('analytic_pre_projection',layer)
            return value[0,unit]
        value,grad=mx.value_and_grad(f)(mx.array(x,dtype=mx.float32));mx.eval(value,grad)
        self.meter('analytic_pre_vjp',layer)
        return float(value),np.asarray(grad).astype(np.float64)


def state(anchor,D,z):return anchor+np.asarray(D@z,dtype=np.float32)


def collect(native,scope,D,P,units,zs,derivatives,emit,label):
    geometry=Geometry(native,{}) if derivatives else None
    X=[];Z=[];Y=[];G=[];before=dict(native.counts)
    for index,z in enumerate(zs):
        native.stage=f'{label}-{index}';x=state(scope['anchor'],D,z)
        zactual=P@(x-scope['anchor']);cs=scope['readout']
        if derivatives:q,jac=geometry.qj(x,cs)
        else:q=native.consumer(x,cs['layer'],cs['indices']);jac=None
        activities={};values=[];gradients=[]
        for column,(layer,unit) in enumerate(units):
            if layer not in activities:activities[layer]=native.activity(x,layer)
            amp=float(activities[layer][unit]);w=native.write(layer,unit,amp);after=x+w
            if derivatives:
                qa,ja=geometry.qj(after,cs)
                _,g=geometry.producer_gradient(x,layer,unit);gD=g@D
                K=(ja-jac)@D+np.outer(ja@D[:,column],gD)
                gradients.append(np.column_stack([gD,K.T]))
            else:qa=native.consumer(after,cs['layer'],cs['indices'])
            values.append(np.r_[amp,qa.astype(np.float64)-q])
        X.append(x);Z.append(zactual);Y.append(values)
        if derivatives:G.append(np.array(gradients).transpose(1,0,2))
        emit(dict(kind='source_context',label=label,index=index,derivatives=derivatives))
        if (index+1)%32==0:print(f'{label}: {index+1} contexts',flush=True)
    result=dict(x=np.array(X),z=np.array(Z),y=np.array(Y))
    if derivatives:result['g']=np.array(G)
    return result,difference(before,native.counts)


def analytic_reference(native,scope,D,units):
    x=scope['anchor'];d=native.width;eps=float(native.config['rms_norm_eps'])
    r=np.sqrt(float(np.mean(x.astype(np.float64)**2))+eps)
    xD=x.astype(np.float64)@D
    g0=[];gd=[];u0=[];ud=[]
    for layer,unit in units:
        for kind,base,rows in [('gate',g0,gd),('up',u0,ud)]:
            value,gradient=native.pre_row(x,layer,unit,kind)
            base.append(r*value);rows.append(r*(gradient@D)+value*xD/(d*r))
    geometry=Geometry(native,{})
    q,jac=geometry.qj(x,scope['readout'])
    qD=r*(jac@D)+np.outer(q,xD)/(d*r)
    return AnalyticReference(float(np.mean(x.astype(np.float64)**2)),xD/d,D.T@D/d,eps,
        np.array(g0),np.array(gd),np.array(u0),np.array(ud),r*q,qD)
