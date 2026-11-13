"""Independent native execution and Boolean specification check for P49.

Does not import the constructor, its quality function, or its specification.
"""
from pathlib import Path
import hashlib
import argparse
from trace_io import open_trace
import json
import time
import numpy as np
from substrate import Substrate, array_sha
from factored import Bank


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def truth(name, bits):
    result = []
    for row in bits:
        b = [bool(v) for v in row]
        if name == 'and2': value = b[0] and b[1]
        elif name == 'or2': value = b[0] or b[1]
        elif name == 'xor2': value = b[0] != b[1]
        elif name == 'and3': value = b[0] and b[1] and b[2]
        elif name == 'or3': value = b[0] or b[1] or b[2]
        elif name == 'xor3': value = (b[0] != b[1]) != b[2]
        elif name == 'majority3': value = (b[0] and b[1]) or (b[0] and b[2]) or (b[1] and b[2])
        elif name == 'mux3': value = b[1] if b[0] else b[2]
        else: raise ValueError(name)
        result.append(value)
    return np.array(result)


def main():
    here = Path(__file__).resolve().parent
    lab = here.parent
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--run',type=Path,default=lab/'data/functions');parser.add_argument('--source',type=Path,default=lab/'data/scaling');parser.add_argument('--output',type=Path,default=lab/'runs/functions-verification.json');args=parser.parse_args();out=args.run
    summary = json.loads((out / 'summary.json').read_text())
    c = summary['config']
    manifest = json.loads((out / 'manifest.json').read_text())
    assert sha(out / 'runner_snapshot.py') == manifest['source_sha256']
    for name, digest in manifest['helper_sha256'].items():
        assert sha(out / 'helpers' / name) == digest
        assert sha(here / name) == digest
    source = args.source / 'source.npz'
    assert sha(source) == c['source_sha256']
    bank = Bank(source, 512)
    units = [tuple(v) for v in json.loads((source.parent / 'units.json').read_text())]
    scope = json.loads((lab / c['source_directory'] / 'scopes/1/domain.json').read_text())['readout']
    counts = {}
    starts = {}
    for line in open_trace(out):
        row = json.loads(line)
        if row['kind'] == 'primitive':
            op = row['operation']; counts[op] = counts.get(op, 0) + row['vectors']
        elif row['kind'] == 'case_start': starts[row['stage']] = row
    assert counts == summary['native_work_total']
    native = Substrate(args.checkpoint, lambda r: None)
    started = time.perf_counter()
    encodings = {}; results = []; scalar_inputs = 0
    try:
        assert native.before[0] == c['parameter_sha256']
        for r in summary['results']:
            case = np.load(out / 'cases' / f't{r["case"]}.npz')
            roots, bits, labels = case['roots'], case['bits'], case['labels']
            assert np.array_equal(labels, truth(r['function'], bits))
            # Exhaust the finite domain exactly once, with no missing/duplicate input.
            assert len(bits) == 2 ** bits.shape[1]
            assert len({tuple(b) for b in bits}) == len(bits)
            key = r['seed'], bits.shape[1]
            if key in encodings: assert np.array_equal(roots, encodings[key])
            else: encodings[key] = roots
            start = starts[f't{r["case"]}-{r["method"]}']
            assert array_sha(roots) == start['root_sha256']
            # Check the supplied generic encoder algebra, independent of target labels.
            expected = (case['frame'][None] + (2 * bits.astype(float) - 1) @ case['directions'][:bits.shape[1]]).astype(np.float32)
            assert np.array_equal(roots, expected)
            replay = roots.copy()
            for action in r['graph']:
                j = action['unit_index']; layer, unit = units[j]
                assert (layer, unit) == (action['layer'], action['unit'])
                for i, x in enumerate(replay):
                    activity = native.activity(x, layer)
                    amp = float(activity[unit])
                    assert abs(amp) <= c['max_atom_amplitude']
                    hidden = np.zeros(native.hidden, dtype=np.float32)
                    hidden[unit] = activity[unit]
                    write = native.down(layer, hidden, 'verification_masked_down')
                    y = x + write
                    assert np.sqrt(np.mean(write * write)) <= c['max_write_rms']
                    assert np.sqrt(np.mean(y * y)) <= c['max_live_rms']
                    assert np.max(abs(bank.P @ (y - roots[i]))) <= c['domain_radius'] + 1e-6
                    replay[i] = y
            q = np.array([native.consumer(x, scope['layer'], scope['indices']) for x in replay])
            assert array_sha(replay) == r['final_sha256']
            assert np.array_equal(q, np.asarray(r['final_q'], dtype=np.float32))
            sign = 2 * labels.astype(int) - 1
            margin = sign * q[:, 0]
            error = np.maximum(c['output_margin'] - margin, 0)
            metrics = dict(loss=float(error @ error), robust_satisfied=bool(np.all(error <= 1e-5)),
                           boolean_correct=bool(np.array_equal(q[:, 0] > 0, labels)),
                           correct_inputs=int(np.sum((q[:, 0] > 0) == labels)), inputs=len(labels),
                           min_signed_margin=float(np.min(margin)), max_violation=float(error.max()))
            assert metrics == r['metrics']
            results.append(dict(case=r['case'], method=r['method'], function=r['function'], seed=r['seed'],
                                program_length=len(r['graph']), **metrics))
            scalar_inputs += len(bits)
        assert native.parameter_digest() == native.before
        assert sha(source) == c['source_sha256']
        report = dict(passed=True, selected_graphs_native_bit_exact=len(results),
                      exhaustive_input_executions=scalar_inputs, distinct_specifications=24,
                      independent_boolean_specifications_match=True, task_independent_encodings_match=True,
                      all_intermediate_physical_bounds_pass=True, primitive_ledger_matches=True,
                      weights_unchanged=True, source_unchanged=True,
                      verification_seconds=time.perf_counter() - started, native_work=native.counts, results=results)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)
    finally:
        native.close()


if __name__ == '__main__':
    main()
