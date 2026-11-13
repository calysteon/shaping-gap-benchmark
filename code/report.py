"""Evidence-derived ledger, central tables and cumulative acquisition-cost curves."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import numpy as np
from native import work

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE.parent/'runs/.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def add(*xs):return {k:sum(x.get(k,0) for x in xs) for k in sorted({k for x in xs for k in x})}
def legacy_work(x):return work(x)+sum(x.get(k,0) for k in ['selector_down_adjoint','consumer_vjp','fidelity_masked_down'])


LABELS={'native_local':'Native local','value_same_pair':'Endpoint,32 contexts','value_budget_pair':'Endpoint,query-matched',
 'differential_pair':'Differential interfaces','differential_one':'Differential,one-step','differential_no_conditioning':'Conditioning removed',
 'differential_corrupt':'Conditioning permuted','differential_restored':'Restored','analytic_pair':'Structured reference'}


def main():
    root=HERE.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=root/'data/benchmark')
    parser.add_argument('--source',type=Path,default=root/'data/source')
    parser.add_argument('--output',type=Path,default=root/'runs/report')
    args=parser.parse_args();run=args.run;parent=args.source
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    paper=out;(paper/'figures').mkdir(exist_ok=True)
    s=json.loads((run/'summary.json').read_text());p46=json.loads((parent/'summary.json').read_text())
    verify=json.loads((root/'data/benchmark/native-verification.json').read_text());assert verify['passed']
    timing=json.loads((root/'data/benchmark/source-timing.json').read_text())
    hist=json.loads((root/'data/accounting/history-summary.json').read_text())
    history=json.loads((root/'data/accounting/ledger.json').read_text())['historical_inventory_discovery']
    costs=s['source_costs'];domain=add(*(v['domain']['native_work'] for v in costs))
    common_native=add(history,s['common_acquisition']['native_work'],domain)
    common_lpe=legacy_work(common_native)
    generation={r['task']:r['generation']['seconds'] for r in s['results']}
    recorded_active=sum(r['search_seconds']+r['deploy_seconds'] for r in s['results'])+sum(generation.values())
    shared_overhead=max(0.,s['wall_seconds']-recorded_active)
    common_seconds=hist['wall_seconds']+s['common_acquisition']['seconds']+sum(v['domain']['seconds'] for v in costs)+shared_overhead
    startup={};ledgers={};curves={};paired={}
    ref={r['task']:r for r in s['results'] if r['method']=='differential_pair'}
    for method in s['config']['methods']:
        source={};source_seconds=fit_seconds=0.
        if method.startswith('differential'):
            source=add(*(v['EK_source']['native_work'] for v in costs))
            source_seconds=sum(v['EK_source']['seconds'] for v in costs)
            fit_seconds=sum(v['fitting_seconds']['differential'] for v in costs)
        elif method=='value_budget_pair':
            source=add(*(v['endpoint_source']['native_work'] for v in costs))
            source_seconds=sum(v['endpoint_source']['seconds'] for v in costs)
            fit_seconds=sum(v['fitting_seconds']['value_budget'] for v in costs)
        elif method=='value_same_pair':
            source=add(*(v['native_work'] for v in timing['timings']))
            source_seconds=sum(v['seconds'] for v in timing['timings'])
            fit_seconds=sum(v['fitting_seconds']['value_same'] for v in costs)
        elif method=='analytic_pair':
            source=add(*(v['analytic']['native_work'] for v in costs))
            source_seconds=sum(v['analytic']['seconds'] for v in costs)
        start_lpe=common_lpe+work(source);start_seconds=common_seconds+source_seconds+fit_seconds
        startup[method]=dict(native_lpe=start_lpe,active_seconds=start_seconds)
        rows=sorted([r for r in s['results'] if r['method']==method],key=lambda r:r['task'])
        q=np.array([r['query_lpe']+work(r['deploy_native_work']) for r in rows])
        t=np.array([r['search_seconds']+r['deploy_seconds']+r['generation']['seconds'] for r in rows])
        solved=np.array([r['metrics']['satisfied'] for r in rows],dtype=int)
        curves[method]=dict(lpe=np.r_[start_lpe,start_lpe+np.cumsum(q)].tolist(),
            seconds=np.r_[start_seconds,start_seconds+np.cumsum(t)].tolist(),solved=np.r_[0,np.cumsum(solved)].tolist())
        native=add(common_native,source,*(r['search_native_work'] for r in rows),
            *(r['deploy_native_work'] for r in rows),*(r['generation']['native_work'] for r in rows))
        assert legacy_work(native)==curves[method]['lpe'][-1]
        ledgers[method]=dict(label=LABELS[method],solved=int(solved.sum()),conditions=len(rows),
            source_native_lpe=work(source),source_seconds=source_seconds,fit_seconds=fit_seconds,
            full_native_lpe=curves[method]['lpe'][-1],full_active_seconds=curves[method]['seconds'][-1],
            native_primitive_ledger=native,source_native_work=source,
            online_query_lpe=sum(r['query_lpe'] for r in rows),deployment_lpe=sum(work(r['deploy_native_work']) for r in rows),
            parameter_payload_bytes=max(r['payload_bytes'] for r in rows))
        wins=[r['task'] for r in rows if ref[r['task']]['metrics']['satisfied'] and not r['metrics']['satisfied']]
        losses=[r['task'] for r in rows if not ref[r['task']]['metrics']['satisfied'] and r['metrics']['satisfied']]
        changed=[r['task'] for r in rows if ref[r['task']]['graph']!=r['graph']]
        paired[method]=dict(differential_wins=wins,differential_losses=losses,changed_graphs=changed)
    baseline=ledgers['native_local'];structured=ledgers['analytic_pair'];values=ledgers['value_same_pair']
    research=dict(development_run_seconds=p46['wall_seconds'],final_run_seconds=s['wall_seconds'],
        source_validation_work=add(*(v['validation']['native_work'] for v in costs)),
        source_validation_seconds=sum(v['validation']['seconds'] for v in costs),
        independent_verification_seconds=verify['verification_seconds'],independent_verification_work=verify['native_work'],
        source_timing_replay_work=timing['native_work_total'],source_timing_replay_seconds=sum(v['seconds'] for v in timing['timings']),
        finite_difference_sweep_work=json.loads((root/'data/benchmark/source-fd-convergence.json').read_text())['native_work'],
        first_verification_attempt='Passed graph/branch replay, failed the small-step finite-difference assertion; duration not separately captured. Source labels/models/results unchanged.')
    ledger=dict(common_native=common_native,common_lpe=common_lpe,common_active_seconds=common_seconds,
        final_shared_model_loading_and_driver_overhead_seconds=shared_overhead,
        methodology='Algorithm lifecycle includes common historical inventory, physical ports/domain setup, required source acquisition/fit, full search and all selected-graph deployment. Scientific validation/replay costs are separately enumerated. LPE counts linear queries/transposes, not hardware FLOPs; active wall records CPU planning and logging too.',
        endpoint32_wall_method='Measured by separate exact source-only replay; no proportional timing estimates.',
        methods=ledgers,research=research)
    dump(out/'lifecycle-ledger.json',ledger);dump(out/'curves.json',curves);dump(out/'paired-outcomes.json',paired)
    analysis=dict(conditions=288,domains=3,methods=ledgers,paired=paired,
        structured_vs_native=dict(additional_satisfied=structured['solved']-baseline['solved'],
            relative_solve_gain=structured['solved']/baseline['solved']-1,
            native_lpe_reduction=1-structured['full_native_lpe']/baseline['full_native_lpe']),
        endpoint32_vs_native=dict(additional_satisfied=values['solved']-baseline['solved'],
            native_lpe_reduction=1-values['full_native_lpe']/baseline['full_native_lpe']),
        derivative_validation=s['validation'],source_k_checks=verify['direct_autodiff_checks_passed'],
        original_fd_passes=verify['original_fd_checks_passed'],all_selected_programs_replayed=verify['selected_graphs_native_bit_exact'])
    dump(out/'analysis.json',analysis)
    # A fixed-stream budget curve counts every unattempted condition as unsolved.
    # It is throughput on the same ordered stream, not a per-task success claim.
    budget_comparisons={}
    for axis in ['lpe','seconds']:
        limit=min(curves[m][axis][-1] for m in curves)
        budget_comparisons[axis]=dict(common_completed_budget=limit,
            successes={m:int(np.array(v['solved'])[np.searchsorted(np.array(v[axis]),limit,side='right')-1]) if limit>=v[axis][0] else 0 for m,v in curves.items()})
    dump(out/'budget-comparisons.json',budget_comparisons)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    shown=['native_local','value_same_pair','value_budget_pair','differential_pair','differential_no_conditioning','analytic_pair']
    colors=['#56616B','#3476A3','#D18B33','#087F82','#8B66A5','#326D3B']
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.3));fig.subplots_adjust(left=.07,right=.98,bottom=.28,top=.89,wspace=.25)
    for ax,axis,scale,title,xlabel in [(axes[0],'lpe',1000,'Native-query lifecycle','Charged native work (10³ LPE)'),
                                    (axes[1],'seconds',1,'Measured active-wall lifecycle','Charged active time (seconds)')]:
        for m,color in zip(shown,colors):ax.step(np.array(curves[m][axis])/scale,curves[m]['solved'],where='post',label=LABELS[m],color=color,lw=1.8)
        ax.set(title=title,xlabel=xlabel,ylabel='Satisfied conditions out of 288');ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
    fig.suptitle('Acquisition is charged before later construction begins',weight='bold',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(paper/'figures'/f'lifecycle.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.7,3.8));fig.subplots_adjust(left=.07,right=.98,bottom=.28,top=.85)
    ms=[m for m in s['config']['methods'] if m!='differential_restored']
    names=['Native\nlocal','Endpoint\n32','Endpoint\nmatched','Differential\npair','Differential\none-step','No\nconditioning','Permuted\nconditioning','Structured\nreference']
    ax.bar(np.arange(len(ms)),[ledgers[m]['solved'] for m in ms],color=['#65737C','#3476A3','#D18B33','#087F82','#67A5A5','#8B66A5','#AF8DBA','#326D3B'])
    ax.set(xticks=np.arange(len(ms)),xticklabels=names,ylabel='Satisfied conditions out of 288',ylim=(0,145))
    for j,m in enumerate(ms):ax.text(j,ledgers[m]['solved']+2,str(ledgers[m]['solved']),ha='center')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_title('Same search engine and 512-LPE online envelope',weight='bold')
    for ext in ['png','pdf']:fig.savefig(paper/'figures'/f'outcomes.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    print(json.dumps(dict(conditions=288,methods=ledgers),indent=2))

if __name__=='__main__':main()
