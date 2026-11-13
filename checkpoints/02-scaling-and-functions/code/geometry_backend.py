"""Native readout Jacobians and live producer partials."""
import numpy as np
from substrate import array_sha

class Geometry:
    def __init__(self,native,c):
        self.native,self.c=native,c;self.qj_cache={};self.grad_cache={};self.ports={}
        self.qj_hits=0

    def qj(self,x,cs):
        key=(array_sha(x),cs['layer'],tuple(cs['indices']))
        if key in self.qj_cache:self.qj_hits+=1;return self.qj_cache[key]
        mx=self.native.mx;q=self.native.consumer(x,cs['layer'],cs['indices']);rows=[]
        for i in range(len(cs['indices'])):
            f=lambda z:self.native.consumer_mx(z,cs['layer'],cs['indices'])[i]
            g=mx.grad(f)(mx.array(x,dtype=mx.float32));mx.eval(g)
            self.native.meter('geometry_consumer_coordinate_vjp',cs['layer'])
            rows.append(np.asarray(g).copy())
        value=(q,np.array(rows,dtype=np.float64));self.qj_cache[key]=value
        return value

    def producer_gradient(self,x,layer,unit):
        key=(array_sha(x),layer,unit)
        if key in self.grad_cache:return self.grad_cache[key]
        mx=self.native.mx;l=self.native.layers[layer]
        def amplitude(z):
            z=l.post_attention_layernorm(z[None]);self.native.meter('geometry_producer_norm',layer)
            g=l.mlp.gate_proj(z);u=l.mlp.up_proj(z)
            self.native.meter('geometry_producer_gate',layer);self.native.meter('geometry_producer_up',layer)
            return self.native.swiglu(g,u)[0,unit]
        value,g=mx.value_and_grad(amplitude)(mx.array(x,dtype=mx.float32));mx.eval(value,g)
        self.native.meter('geometry_producer_coordinate_vjp',layer)
        result=(float(value),np.asarray(g).astype(np.float64));self.grad_cache[key]=result
        return result

    def direction(self,layer,unit):
        key=(layer,unit)
        if key not in self.ports:self.ports[key]=self.native.acquire_direction(layer,unit).astype(np.float64)
        return self.ports[key]

    def execute(self,x,root,binding):
        source=root if binding['port']=='root' else x
        activity=self.native.activity(source,binding['layer'])
        return x+self.native.write(binding['layer'],binding['unit'],float(activity[binding['unit']]))
