"""Direct, parameter-backed neural fragments. No tokenizer or model forward.

A SwiGLU atom is the original gate/up scalar at one native neuron followed by
its original down-projection connection. Module boundaries and residual addition
are supplied physical syntax; no semantic role is assigned to a neuron.
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np


def array_sha(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


class Substrate:
    def __init__(self, checkpoint, emit):
        os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
        global mx
        import mlx.core as mx
        from mlx_lm.utils import load_model
        from mlx.utils import tree_flatten
        from mlx_lm.models.activations import swiglu
        self.mx, self.swiglu, self.emit = mx, swiglu, emit
        self.model, self.config = load_model(Path(checkpoint), trust_remote_code=False)
        self.model.eval(); self.model.freeze()
        assert not tree_flatten(self.model.trainable_parameters())
        self.layers = self.model.model.layers
        self.width = self.config['hidden_size']; self.hidden = self.config['intermediate_size']
        self.counts = {}; self.serial = 0; self.stage = 'setup'; self.guards = []
        self.before = self.parameter_digest()
        # An accidental return to whole-model, transformer-block, attention or
        # token I/O execution aborts. The loaded model only owns frozen arrays.
        for cls in {type(self.model), type(self.model.model), type(self.layers[0]),
                    type(self.layers[0].self_attn), type(self.model.model.embed_tokens)}:
            old = cls.__call__
            def forbidden(*args, _name=cls.__name__, **kwargs):
                raise RuntimeError('forbidden_original_execution:'+_name)
            cls.__call__ = forbidden
            self.guards.append((cls, '__call__', old))
        cls = type(self.model.model.embed_tokens)
        if hasattr(cls, 'as_linear'):
            old = cls.as_linear
            def forbidden_head(*args, **kwargs):
                raise RuntimeError('forbidden_original_unembedding')
            cls.as_linear = forbidden_head
            self.guards.append((cls, 'as_linear', old))

    def close(self):
        for cls, name, call in self.guards:
            setattr(cls, name, call)
        self.guards = []

    def parameter_digest(self):
        from mlx.utils import tree_flatten
        h = hashlib.sha256(); size = 0
        for name, a in tree_flatten(self.model.parameters()):
            raw = np.asarray(a.view(mx.uint8))
            h.update(name.encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
            h.update(raw.tobytes()); size += raw.nbytes
        return h.hexdigest(), size

    def meter(self, operation, layer, vectors=1, **details):
        self.serial += 1
        self.counts[operation] = self.counts.get(operation, 0)+vectors
        self.emit(dict(kind='primitive', serial=self.serial, stage=self.stage,
                       operation=operation, layer=int(layer), vectors=int(vectors), **details))

    def activity(self, x, layer):
        """The original input-dependent hidden activations, not fitted offsets."""
        l = self.layers[layer]; z = l.post_attention_layernorm(mx.array(x[None], dtype=mx.float32))
        self.meter('producer_norm', layer)
        a = self.swiglu(l.mlp.gate_proj(z), l.mlp.up_proj(z))
        self.meter('producer_gate_projection', layer); self.meter('producer_up_projection', layer)
        mx.eval(a)
        return np.asarray(a[0]).copy()

    def down(self, layer, hidden, purpose):
        a = self.layers[layer].mlp.down_proj(mx.array(hidden[None], dtype=mx.float32))
        mx.eval(a); self.meter(purpose, layer)
        return np.asarray(a[0]).copy()

    def write(self, layer, neuron, amplitude):
        hidden = np.zeros(self.hidden, dtype=np.float32); hidden[neuron] = amplitude
        return self.down(layer, hidden, 'atom_down_projection')

    def acquire_direction(self, layer, neuron):
        hidden = np.zeros(self.hidden, dtype=np.float32); hidden[neuron] = 1.
        return self.down(layer, hidden, 'interface_basis_probe')

    def down_adjoint(self, layer, gradient):
        """Native quantized matrix adjoint: one score per physical output port.

        This is known sensitivity machinery, not a learned semantic interface.
        Its work is charged separately from useful atom execution.
        """
        cot = mx.array(gradient[None], dtype=mx.float32)
        fn = lambda h: mx.sum(self.layers[layer].mlp.down_proj(h)*cot)
        g = mx.grad(fn)(mx.zeros((1, self.hidden), dtype=mx.float32))
        mx.eval(g); self.meter('selector_down_adjoint', layer)
        return np.asarray(g[0]).copy()

    def consumer_mx(self, x, layer, indices):
        l = self.layers[layer]
        q = l.mlp.gate_proj(l.post_attention_layernorm(x[None]))[0]
        self.meter('consumer_norm', layer); self.meter('consumer_gate_projection', layer)
        return q[mx.array(indices)]

    def consumer(self, x, layer, indices):
        q = self.consumer_mx(mx.array(x, dtype=mx.float32), layer, indices)
        mx.eval(q); return np.asarray(q).copy()

    def loss_gradient(self, x, contract):
        lower = mx.array(contract.lower); upper = mx.array(contract.upper)
        def objective(z):
            q = self.consumer_mx(z, contract.layer, contract.indices)
            lo = mx.maximum(lower-q, 0); hi = mx.maximum(q-upper, 0)
            return mx.sum(lo*lo+hi*hi)
        loss, g = mx.value_and_grad(objective)(mx.array(x, dtype=mx.float32))
        mx.eval(loss, g); self.meter('consumer_vjp', contract.layer)
        return float(loss), np.asarray(g).copy()

    def fidelity_check(self, x, layer, neurons):
        """Check decomposition against the native masked-hidden computation."""
        a = self.activity(x, layer)
        mask = np.zeros_like(a); mask[list(neurons)] = a[list(neurons)]
        direct = self.down(layer, mask, 'fidelity_masked_down')
        pieces = [self.write(layer, j, float(a[j])) for j in neurons]
        composed = np.sum(pieces, axis=0)
        difference = float(np.max(np.abs(direct-composed)))
        return dict(layer=layer, neurons=list(neurons), max_abs_error=difference,
                    passed=bool(np.allclose(direct, composed, rtol=2e-5, atol=2e-5)))
