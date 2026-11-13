"""Complete frozen transformer field on continuous input states, no token I/O."""
from pathlib import Path
import hashlib
import os
import time
import numpy as np


class NativeField:
    def __init__(self,path,emit,tokens=2,batch_size=32):
        os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1')
        import mlx.core as mx
        from mlx_lm.utils import load_model
        from mlx_lm.models.base import create_attention_mask
        from mlx.utils import tree_flatten
        self.mx=mx;self.mask=create_attention_mask;self.emit=emit;self.stage='load'
        self.model,self.config=load_model(Path(path),trust_remote_code=False)
        self.model.eval();self.model.freeze();assert not tree_flatten(self.model.trainable_parameters())
        self.tokens=tokens;self.width=self.config['hidden_size'];self.n=tokens*self.width;self.batch=batch_size
        self.layers=len(self.model.model.layers);self.counts=dict(batches=0,sequences=0,tokens=0,attention_block_tokens=0,mlp_block_tokens=0)
        self.seconds=0.;self.before=self.digest()

    def digest(self):
        from mlx.utils import tree_flatten
        h=hashlib.sha256();size=0
        for name,a in tree_flatten(self.model.parameters()):
            raw=np.asarray(a.view(self.mx.uint8));h.update(name.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(raw.tobytes());size+=raw.nbytes
        return h.hexdigest(),size

    def run(self,X):
        X=np.asarray(X,dtype=np.float32).reshape(-1,self.tokens,self.width);outputs=[];began=time.perf_counter()
        for start in range(0,len(X),self.batch):
            h=self.mx.array(X[start:start+self.batch]);b=len(h);mask=self.mask(h,None)
            for layer in self.model.model.layers:h=layer(h,mask,None)
            y=self.model.model.norm(h).astype(self.mx.float32);self.mx.eval(y)
            outputs.append(np.asarray(y).reshape(b,self.n).copy())
            self.counts['batches']+=1;self.counts['sequences']+=b;self.counts['tokens']+=b*self.tokens
            self.counts['attention_block_tokens']+=b*self.tokens*self.layers;self.counts['mlp_block_tokens']+=b*self.tokens*self.layers
            self.emit(dict(kind='native_field',stage=self.stage,batch=b,tokens=b*self.tokens,layers=self.layers,attention_block_tokens=b*self.tokens*self.layers,mlp_block_tokens=b*self.tokens*self.layers))
        self.seconds+=time.perf_counter()-began
        result=np.concatenate(outputs);assert np.isfinite(result).all();return result
