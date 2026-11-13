"""Deterministic P48/P49 analysis; no model calls or outcome selection."""
from collections import Counter
from pathlib import Path
import csv
import argparse
import hashlib
import json
import os
import shutil
import sys
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parent / '.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from native import PROJECTION_WEIGHTS

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parent.parent
OUT = LAB / 'runs/analysis'
WEIGHTS = dict(PROJECTION_WEIGHTS, selector_down_adjoint=1, consumer_vjp=1,
               fidelity_masked_down=1, verification_masked_down=1)


def load(p): return json.loads(p.read_text())
def dump(p, v): p.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')
def work(v): return sum(WEIGHTS.get(k, 0) * n for k, n in v.items())
def add(*vectors):
    c = Counter()
    for v in vectors: c.update(v)
    return dict(c)
def total(rows, key): return add(*(r[key] for r in rows))
def selected_check(rows, joint=False):
    n = sum((r['inputs'] if joint else 1) * len(r['graph']) for r in rows)
    q = sum(r['inputs'] if joint else 1 for r in rows)
    return dict(producer_norm=n, producer_gate_projection=n, producer_up_projection=n,
                verification_masked_down=n, consumer_norm=q, consumer_gate_projection=q)
def table(p, rows):
    with p.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT);OUT=parser.parse_args().output
    OUT.mkdir(parents=True, exist_ok=True)
    a = LAB / 'data/scaling'
    b = LAB / 'data/functions'
    s, t = load(a / 'summary.json'), load(b / 'summary.json')
    va, vb = load(a / 'native-verification.json'), load(b / 'native-verification.json')
    assert va['passed'] and vb['passed']
    hist = load(ROOT / 'data/accounting/ledger.json')['historical_inventory_discovery']
    assert work(hist) == 3199
    old = load(LAB / 'data/inherited/summary.json')
    domain = add(*(r['domain']['native_work'] for r in old['source_costs']))
    costs = load(a / 'source-costs.json')
    source_validation = load(a / 'source-validation.json')
    def source(k, behavior):
        rows = costs['components'][:k]
        return add(hist, domain, *(r['physical_native_work'] for r in rows),
                   *([costs['common_readouts']['native_work']] + [r['behavior_native_work'] for r in rows] if behavior else []))
    ledger = dict(cost_unit='linear-projection-equivalent count, not FLOPs, joules or equal-cost hardware operations',
                  weights=WEIGHTS, historical_inventory=hist, inherited_domain_setup=domain,
                  historical_inventory_wall_seconds=load(ROOT / 'data/accounting/history-summary.json')['wall_seconds'],
                  inherited_domain_setup_wall_seconds=sum(r['domain']['seconds'] for r in old['source_costs']),
                  scopes={'acquisition_construction_deployment': 'Historical inventory, inherited domain setup, calibrated source rows/directions, all attempted native branches, initial readouts and one native deployment per selected graph.',
                          'with_selected_program_verification': 'Previous scope plus independent native execution of every selected graph.',
                          'research_diagnostics': 'Source validation, microbenchmark native probes, and sampled-branch audit retained separately; not silently assigned zero.',
                          'maintenance': 'No updates, deletions, eviction or refitting in these two studies; zero maintenance events. This does not evaluate a continual-maintenance algorithm.',
                          'wall': 'Complete experiment wall and measured components separately. Shared startup, serialization and unallocated driver overhead are not attributed to either arm. No energy measurement.'},
                  scaling=[], functions=[])
    scaling = []; micro = []
    for k in s['config']['bank_sizes']:
        prep = next(r for r in s['preparation'] if r['bank_size'] == k)
        c = costs['prefix'][str(k)]
        for method in s['config']['methods']:
            rows = [r for r in s['results'] if r['bank_size'] == k and r['method'] == method]
            behavior = method == 'factored_pair'
            initial = add(*(r['generation']['native_work'] for r in rows))
            search = total(rows, 'search_native_work'); deploy = total(rows, 'deploy_native_work')
            acquisition = source(k, behavior)
            charged = add(acquisition, initial, search, deploy)
            checked = selected_check(rows)
            assert work(add(initial, search)) == sum(r['query_lpe'] for r in rows)
            measured_wall = c['physical_seconds'] + prep['physical_seconds'] + sum(r['search_seconds'] + r['deploy_seconds'] + r['generation']['seconds'] for r in rows)
            if behavior: measured_wall += c['behavior_seconds'] + prep['behavior_seconds']
            row = dict(bank_size=k, method=method, solved=sum(r['metrics']['satisfied'] for r in rows), contracts=len(rows),
                       acquisition_lpe=work(acquisition), construction_lpe=work(add(initial, search)), deployment_lpe=work(deploy),
                       acquisition_construction_deployment_lpe=work(charged), independent_selected_verification_lpe=work(checked),
                       with_selected_program_verification_lpe=work(add(charged, checked)),
                       measured_algorithm_component_seconds=measured_wall,
                       retained_bank_bytes=prep['payload_bytes'], controller_array_bound_bytes=max(r['controller_array_bound_bytes'] for r in rows))
            scaling.append(row)
            ledger['scaling'].append(dict(**row, vectors=dict(acquisition=acquisition, initial=initial, search=search, deployment=deploy, verification=checked)))
        mr = [r for r in s['micro'] if r['bank_size'] == k]
        direct = float(np.median([r['seconds']['direct'] for r in mr]))
        factor = float(np.median([r['seconds']['factored'] for r in mr]))
        micro.append(dict(bank_size=k, contexts=len(mr), direct_median_seconds=direct, factored_median_seconds=factor,
                          median_ratio=direct/factor, first_six_agree=all(r['first_six_choices_equal'] for r in mr),
                          max_abs_discrepancy=max(v for r in mr for v in r['max_abs_discrepancy'].values())))
    all_checks = selected_check(s['results'])
    branch_audit = {op: n-all_checks.get(op, 0) for op, n in va['native_work'].items()}
    assert all(n >= 0 for n in branch_audit.values())
    ledger['scaling_research_diagnostics'] = dict(source_validation=source_validation,
        micro_native_probes=dict(consumer_norm=len(s['micro']), consumer_gate_projection=len(s['micro'])),
        sampled_branch_audit_vector=branch_audit, sampled_branch_audit_lpe=work(branch_audit),
        complete_run_native_vector=s['native_work_total'], independent_verification_native_vector=va['native_work'],
        complete_run_wall_seconds=s['wall_seconds'], independent_verification_seconds=va['verification_seconds'],
        process_peak_rss_bytes=s['process_peak_rss_bytes'], mlx_peak_bytes=s['mlx_peak_bytes'])
    functions = []
    by_function = []
    for method in t['config']['methods']:
        rows = [r for r in t['results'] if r['method'] == method]
        acquisition = source(512, True)
        initial = add(*(r['initial']['native_work'] for r in rows))
        search = total(rows, 'search_native_work'); deploy = total(rows, 'deploy_native_work')
        checked = selected_check(rows, joint=True)
        assert work(add(initial, search)) == sum(r['query_lpe'] for r in rows)
        charged = add(acquisition, initial, search, deploy)
        row = dict(method=method, exact_functions=sum(r['metrics']['boolean_correct'] for r in rows),
                   robust_functions=sum(r['metrics']['robust_satisfied'] for r in rows), targets=len(rows),
                   acquisition_lpe=work(acquisition), construction_lpe=work(add(initial, search)), deployment_lpe=work(deploy),
                   acquisition_construction_deployment_lpe=work(charged), independent_selected_verification_lpe=work(checked),
                   with_selected_program_verification_lpe=work(add(charged, checked)),
                   search_seconds=sum(r['search_seconds'] for r in rows), deployment_seconds=sum(r['deploy_seconds'] for r in rows),
                   median_program_length=float(np.median([len(r['graph']) for r in rows])),
                   maximum_controller_array_bound_bytes=max(r['controller_array_bound_bytes'] for r in rows))
        functions.append(row)
        ledger['functions'].append(dict(**row, vectors=dict(acquisition=acquisition, initial=initial, search=search, deployment=deploy, verification=checked)))
        for name in t['config']['functions']:
            rs = [r for r in rows if r['function'] == name]
            by_function.append(dict(function=name, method=method, exact=sum(r['metrics']['boolean_correct'] for r in rs),
                                    robust=sum(r['metrics']['robust_satisfied'] for r in rs), encoders=len(rs),
                                    construction_lpe=sum(r['query_lpe'] for r in rs), minimum_margin=min(r['metrics']['min_signed_margin'] for r in rs)))
    ledger['functions_research_diagnostics'] = dict(complete_run_native_vector=t['native_work_total'],
        independent_verification_native_vector=vb['native_work'], complete_run_wall_seconds=t['wall_seconds'],
        independent_verification_seconds=vb['verification_seconds'], process_peak_rss_bytes=t['process_peak_rss_bytes'],
        mlx_peak_bytes=t['mlx_peak_bytes'], source_admission_reuses_P48_validation=True,
        source_preparation_wall='P49 bank preparation is inside complete wall; it was not separately instrumented. P48 preparation timings must not be presented as a P49 wall measurement.')
    summary = dict(scaling=scaling, micro=micro, functions=functions, by_function=by_function,
                   verified_programs=va['selected_graphs_native_bit_exact']+vb['selected_graphs_native_bit_exact'],
                   weights_unchanged=s['weights_unchanged'] and t['weights_unchanged'],
                   source_unchanged=s['source_unchanged'] and t['source_unchanged'])
    dump(OUT / 'summary.json', summary); dump(OUT / 'cost-ledger.json', ledger)
    table(OUT / 'scaling.csv', scaling); table(OUT / 'functions.csv', functions); table(OUT / 'functions-by-target.csv', by_function)
    table(OUT / 'forecast-microbenchmark.csv', micro)
    for p, label in [(a, 'scaling'), (b, 'functions')]:
        shutil.copy2(p / 'native-verification.json', OUT / f'{label}-verification.json')
        shutil.copy2(p / 'protocol.json', OUT / f'{label}-protocol.json')
    shutil.copy2(LAB / 'theory/response-moments.txt', OUT / 'response-moment-theory.txt')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method, label, color in [('native_local', 'Native local', '#54637b'), ('factored_pair', 'Acquired interfaces + pair forecast', '#087e8b')]:
        rs = [r for r in scaling if r['method'] == method]; ks = [r['bank_size'] for r in rs]
        axs[0,0].plot(ks, [r['solved'] for r in rs], 'o-', label=label, color=color)
        axs[0,1].plot(ks, [r['with_selected_program_verification_lpe'] for r in rs], 'o-', color=color)
    axs[0,0].set(title='Common 72 contracts; 512 LPE/search', ylabel='Contracts satisfied', ylim=(0, 76))
    axs[0,0].legend(fontsize=8)
    axs[0,1].set(title='Acquisition + construction + deployment + verification', ylabel='Native projection-equivalent count')
    axs[1,0].plot([r['bank_size'] for r in micro], [r['direct_median_seconds']*1000 for r in micro], 'o-', label='Direct activation-space evaluation', color='#b85f18')
    axs[1,0].plot([r['bank_size'] for r in micro], [r['factored_median_seconds']*1000 for r in micro], 'o-', label='Response moments', color='#087e8b')
    axs[1,0].set(title='Exploratory forecast timing: 3 contexts; fixed order', ylabel='Median milliseconds (log scale)', yscale='log')
    axs[1,0].legend(fontsize=8)
    rs = [r for r in scaling if r['method'] == 'factored_pair']
    axs[1,1].plot([r['bank_size'] for r in rs], [r['retained_bank_bytes']/2**20 for r in rs], 'o-', label='Retained arrays')
    axs[1,1].plot([r['bank_size'] for r in rs], [r['controller_array_bound_bytes']/2**20 for r in rs], 'o-', label='Controller array reservation')
    axs[1,1].axhline(64, linestyle='--', color='gray', label='Controller limit')
    axs[1,1].set(title='Explicit arrays; full process peak = 585 MiB', ylabel='MiB', ylim=(0, 70)); axs[1,1].legend(fontsize=8)
    for ax in axs.flat: ax.set_xlabel('Available frozen components'); ax.set_xticks([22,128,256,512]); ax.grid(alpha=.15)
    fig.savefig(OUT / 'scaling.png', dpi=170); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(2); names = ['One-step', 'Paired']
    axs[0].bar(x-.16, [r['exact_functions'] for r in functions], .3, label='Exact Boolean truth table', color='#54637b')
    axs[0].bar(x+.16, [r['robust_functions'] for r in functions], .3, label='All outputs meet margin 0.025', color='#087e8b')
    axs[0].set(xticks=x, xticklabels=names, ylim=(0,30), yticks=[0,6,12,18,24], ylabel='Functions / 24', title='One program across every input'); axs[0].legend(fontsize=8)
    axs[1].bar(x, [r['with_selected_program_verification_lpe'] for r in functions], color=['#54637b','#087e8b'])
    axs[1].set(xticks=x, xticklabels=names, ylabel='Native projection-equivalent count', title='Acquisition through independent verification')
    fig.savefig(OUT / 'functions.png', dpi=170); plt.close(fig)
    print(json.dumps(dict(scaling=scaling, functions=functions, verified_programs=summary['verified_programs']), indent=2))


if __name__ == '__main__': main()
